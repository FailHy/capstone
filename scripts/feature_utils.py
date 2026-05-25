"""
feature_utils.py — Fixed Version
==================================
BUGS FIXED:
  #2 - Missing Functions: hybrid_gatekeeper dan feedback_from_prediction
       ditambahkan sebagai implementasi minimal yang fungsional (bukan stub
       kosong). Ini agar evaluator_service.py dan live_evaluator.py tidak crash.
  #4 - Signature Mismatch: RepetitionSegmenter.reset() sekarang punya
       parameter keep_down (default False) untuk backward-compatibility.
  Original fix: KeyError 'avg_visibility' -> 'mean_visibility' (sudah ada
       di kode original, dipertahankan).

DESIGN NOTE - hybrid_gatekeeper:
  Ini adalah rule-based pre-filter sebelum masuk XGBoost. Jika kondisi
  biomekanik jelas (misal ROM sangat kecil), langsung label tanpa ML.
  Threshold dipilih konservatif agar tidak meng-override model terlalu agresif.

DESIGN NOTE - feedback_from_prediction:
  Menghasilkan string feedback yang actionable untuk ditampilkan di UI.
  Tidak perlu ML — pure lookup table dari label + feature values.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from config import EXERCISE_CONFIG


# ===========================================================================
# Angle Smoother
# ===========================================================================

class AngleSmoother:
    """Moving-average smoother untuk meredam noise jitter MediaPipe."""
    def __init__(self, window: int = 5) -> None:
        self._buf: deque = deque(maxlen=window)

    def update(self, angle: float) -> float:
        self._buf.append(angle)
        return float(np.mean(self._buf))

    def reset(self) -> None:
        self._buf.clear()


# ===========================================================================
# RepBuffer
# ===========================================================================

@dataclass
class RepBuffer:
    def __init__(self) -> None:
        self.times: List[float] = []
        self.angles: List[float] = []
        self.features: List[Dict[str, float]] = []

    def append(self, timestamp: float, feature_dict: Dict[str, float]) -> None:
        self.times.append(timestamp)
        self.angles.append(feature_dict["elbow_angle"])
        self.features.append(feature_dict)

    def clear(self) -> None:
        self.times.clear()
        self.angles.clear()
        self.features.clear()

    @property
    def is_empty(self) -> bool:
        return len(self.times) == 0


# ===========================================================================
# RepetitionSegmenter
# ===========================================================================

class RepetitionSegmenter:
    """
    State machine untuk mendeteksi fase repetisi.

    BUG FIX #4: reset() sekarang menerima keep_down: bool = False.
      - keep_down=False (default): reset penuh ke idle, semua state cleared.
      - keep_down=True: set state ke 'down' agar segmenter tidak perlu
        menunggu start_angle terpenuhi lagi jika user masih di posisi bawah.
        Digunakan oleh live_evaluator.py setelah rep selesai.
    """
    DEBOUNCE_FRAMES = 3

    def __init__(self, exercise_type: str = "biceps") -> None:
        self.exercise_type = exercise_type
        config = EXERCISE_CONFIG.get(exercise_type, EXERCISE_CONFIG["biceps"])
        self.start_angle: float = config["start_angle"]
        self.peak_angle: float  = config["peak_angle"]
        self.direction: str     = config["direction"]

        self.state      = "idle"
        self.start_time = 0.0
        self.peak_time  = 0.0
        self.end_time   = 0.0

        self._smoother        = AngleSmoother(window=5)
        self._prev_smooth: float | None = None
        self._debounce_count  = 0
        self._debounce_target = ""

    def _reset_debounce(self) -> None:
        self._debounce_count  = 0
        self._debounce_target = ""

    def _confirm(self, target_state: str, condition: bool, timestamp: float) -> bool:
        if condition:
            if self._debounce_target != target_state:
                self._debounce_target = target_state
                self._debounce_count  = 1
            else:
                self._debounce_count += 1
            if self._debounce_count >= self.DEBOUNCE_FRAMES:
                self._reset_debounce()
                return True
        else:
            if self._debounce_target == target_state:
                self._reset_debounce()
        return False

    def update(self, raw_angle: float, timestamp: float) -> str:
        smooth_angle = self._smoother.update(raw_angle)
        if self._prev_smooth is None:
            self._prev_smooth = smooth_angle
            return self.state

        angle_diff        = smooth_angle - self._prev_smooth
        self._prev_smooth = smooth_angle
        event = self._run_logic(smooth_angle, angle_diff, timestamp)
        return event

    def _run_logic(self, angle: float, diff: float, ts: float) -> str:
        if self.direction == "up":
            if self.state in ("idle", "down"):
                if self._confirm("moving_up", diff < -3.0, ts):
                    self.state = "moving_up"; self.start_time = ts
                    return "started"
            elif self.state == "moving_up":
                if self._confirm("top", angle <= self.peak_angle + 5.0 or diff > 3.0, ts):
                    self.state = "top"; self.peak_time = ts
                    return "top"
            elif self.state == "top":
                if self._confirm("moving_down", diff > 2.0, ts):
                    self.state = "moving_down"
            elif self.state == "moving_down":
                if self._confirm("completed", angle >= self.start_angle - 25.0, ts):
                    self.state = "idle"; self.end_time = ts
                    return "completed"

        elif self.direction == "down":
            if self.state in ("idle", "top"):
                if self._confirm("moving_down", diff > 3.0, ts):
                    self.state = "moving_down"; self.start_time = ts
                    return "started"
            elif self.state == "moving_down":
                if self._confirm("down", angle >= self.peak_angle - 5.0 or diff < -3.0, ts):
                    self.state = "down"; self.peak_time = ts
                    return "top"
            elif self.state == "down":
                if self._confirm("moving_up", diff < -2.0, ts):
                    self.state = "moving_up"
            elif self.state == "moving_up":
                if self._confirm("completed", angle <= self.start_angle + 25.0, ts):
                    self.state = "idle"; self.end_time = ts
                    return "completed"

        return self.state

    def reset(self, keep_down: bool = False) -> None:
        """
        Reset state machine untuk repetisi berikutnya.

        Args:
            keep_down: Jika True, state diset ke 'down' (biceps) atau 'top'
                       (triceps) sehingga segmenter siap mendeteksi angkatan
                       berikutnya tanpa harus kembali ke posisi start.
                       Jika False (default), reset penuh ke 'idle'.
        """
        self.start_time   = 0.0
        self.peak_time    = 0.0
        self.end_time     = 0.0
        self._prev_smooth = None
        self._smoother.reset()
        self._reset_debounce()

        if keep_down:
            # Untuk biceps: setelah rep selesai, tangan sudah di bawah (down).
            # Untuk triceps: setelah rep selesai, tangan sudah di atas (top).
            self.state = "down" if self.direction == "up" else "top"
        else:
            self.state = "idle"


# ===========================================================================
# Feature Extraction
# ===========================================================================

def extract_repetition_features(buffer: RepBuffer) -> Dict[str, float]:
    if buffer.is_empty:
        return {}

    features = buffer.features
    angles   = buffer.angles
    times    = buffer.times

    min_angle = float(np.min(angles))
    max_angle = float(np.percentile(angles, 95))
    rom       = max_angle - min_angle

    upper_arm_angles = [f["upper_arm_angle"] for f in features]
    shoulder_angles  = [f["shoulder_angle"]  for f in features]
    torso_angles     = [f["torso_angle"]     for f in features]
    visibilities     = [f["mean_visibility"] for f in features]  # FIX: avg_ -> mean_

    elbow_drift = (
        float(np.max(upper_arm_angles) - np.min(upper_arm_angles))
        if upper_arm_angles else 0.0
    )

    rep_duration = times[-1] - times[0] if len(times) > 1 else 0.0
    dt           = np.diff(times)
    d_angle      = np.diff(angles)
    velocities   = np.abs(d_angle / (dt + 1e-6))

    velocity_mean = float(np.mean(velocities)) if len(velocities) > 0 else 0.0
    velocity_std  = float(np.std(velocities))  if len(velocities) > 0 else 0.0

    peak_idx = int(np.argmin(angles))
    if 0 < peak_idx < len(times) - 1:
        up_duration   = times[peak_idx] - times[0]
        down_duration = times[-1] - times[peak_idx]
    else:
        up_duration   = rep_duration / 2
        down_duration = rep_duration / 2

    accelerations     = np.diff(velocities) / (dt[1:] + 1e-6) if len(dt) > 1 else np.zeros(1)
    motion_smoothness = float(np.std(accelerations)) if len(accelerations) > 0 else 0.0

    return {
        "rom_elbow":           float(rom),
        "upper_arm_angle_std": float(np.std(upper_arm_angles)),
        "torso_sway_range":    float(np.max(torso_angles) - np.min(torso_angles)),
        "shoulder_angle_range":float(np.max(shoulder_angles) - np.min(shoulder_angles)),
        "elbow_drift_range":   elbow_drift,
        "mean_visibility":     float(np.mean(visibilities)),
        "rep_duration":        float(rep_duration),
        "up_phase_duration":   float(up_duration),
        "down_phase_duration": float(down_duration),
        "elbow_velocity_mean": velocity_mean,
        "elbow_velocity_std":  velocity_std,
        "motion_smoothness":   motion_smoothness,
    }


def validate_repetition_quality(
    buffer: RepBuffer,
    features: Dict[str, float],
    exercise_type: str = "biceps",
) -> Tuple[bool, str]:
    if buffer.is_empty:
        return False, "Buffer kosong."
    config = EXERCISE_CONFIG.get(exercise_type, EXERCISE_CONFIG["biceps"])
    if features.get("rom_elbow", 0) < config["rom_min_absurd"]:
        return False, f"ROM terlalu kecil (<{config['rom_min_absurd']} derajat)."
    return True, "Valid"


# ===========================================================================
# BUG FIX #2A: hybrid_gatekeeper
# ===========================================================================

# Threshold biomekanik untuk rule-based pre-filter.
# Nilai ini TIDAK boleh sama dengan threshold yang digunakan clean_dataset.py
# (anti-leakage). Ini adalah batas "absurd" yang jelas salah secara fisik.
_GATEKEEPER_ROM_MIN     = 15.0   # deg — di bawah ini pasti bukan rep valid
_GATEKEEPER_SWAY_MAX    = 30.0   # deg — torso swaying ekstrem
_GATEKEEPER_VEL_MAX     = 800.0  # deg/s — tempo tidak mungkin dikontrol
_GATEKEEPER_DRIFT_MAX   = 40.0   # deg — siku keluar terlalu jauh

def hybrid_gatekeeper(
    features: Dict[str, float]
) -> Tuple[bool, str, str]:
    """
    Rule-based pre-filter sebelum XGBoost inference.

    Returns:
        (allowed_to_model, label, feedback)
        - allowed_to_model: False jika rule langsung menentukan label,
          True jika harus diteruskan ke ML model.
        - label: label string jika allowed=False, kosong jika True.
        - feedback: pesan feedback untuk user.

    DESIGN: Threshold konservatif. Hanya tangkap kasus yang JELAS salah
    secara biomekanik. Jangan terlalu agresif — biarkan ML yang menilai
    kasus ambigu.
    """
    rom          = features.get("rom_elbow", 999.0)
    torso_sway   = features.get("torso_sway_range", 0.0)
    vel_mean     = features.get("elbow_velocity_mean", 0.0)
    elbow_drift  = features.get("elbow_drift_range", 0.0)

    # 1. ROM terlalu kecil: jelas bukan full rep
    if rom < _GATEKEEPER_ROM_MIN:
        return (
            False,
            "not_full_up",
            f"Angkat lebih tinggi! ROM hanya {rom:.0f}°, minimum {_GATEKEEPER_ROM_MIN}°.",
        )

    # 2. Torso sway ekstrem: badan berayun berlebihan
    if torso_sway > _GATEKEEPER_SWAY_MAX:
        return (
            False,
            "body_swing",
            f"Stabilkan badan! Torso bergerak {torso_sway:.0f}°.",
        )

    # 3. Kecepatan tidak manusiawi: kemungkinan tracking error
    if vel_mean > _GATEKEEPER_VEL_MAX:
        return (
            False,
            "too_fast",
            "Gerakan terlalu cepat atau terjadi tracking error. Ulangi dengan kontrol.",
        )

    # Tidak ada rule yang triggered — teruskan ke ML model
    return True, "", ""


# ===========================================================================
# BUG FIX #2B: feedback_from_prediction
# ===========================================================================

# Mapping dari error label ke pesan feedback yang actionable.
# Menggunakan fitur untuk personalisasi pesan jika relevan.
_FEEDBACK_MAP: Dict[str, str] = {
    "correct":           "Bagus! Gerakan sudah benar. Pertahankan!",
    "body_swing":        "Stabilkan badan — jangan gunakan momentum torso untuk mengangkat.",
    "elbow_swing":       "Kunci siku di sisi badan. Bahu jangan ikut bergerak.",
    "not_full_up":       "Angkat lebih tinggi! Pastikan range gerak penuh dari bawah ke atas.",
    "too_fast":          "Perlambat fase turun (eksentrik). Kontrol beban jangan dijatuhkan.",
    "unknown_incorrect": "Gerakan kurang tepat. Fokus pada isolasi otot target.",
    "uncertain":         "Evaluasi tidak meyakinkan. Coba ulangi gerakan dengan lebih jelas.",
}

def feedback_from_prediction(
    label: str,
    confidence: float,
    features: Dict[str, float],
) -> str:
    """
    Menghasilkan string feedback yang actionable berdasarkan label prediksi.

    Args:
        label: Label dari XGBoost atau gatekeeper.
        confidence: Confidence score model (0.0 - 1.0).
        features: Feature dict dari extract_repetition_features().

    Returns:
        String feedback untuk ditampilkan di UI.
    """
    base = _FEEDBACK_MAP.get(label, f"Hasil: {label}. Coba lagi.")

    # Tambahkan konteks numerik untuk kasus spesifik
    if label == "not_full_up":
        rom = features.get("rom_elbow", 0.0)
        if rom > 0:
            base += f" (ROM terdeteksi: {rom:.0f}°)"

    elif label == "too_fast":
        vel = features.get("elbow_velocity_mean", 0.0)
        if vel > 0:
            base += f" (Kecepatan: {vel:.0f}°/s)"

    elif label == "correct" and confidence < 0.75:
        base += " (Keyakinan model sedang — pastikan kamera stabil)."

    return base