"""Feature extraction, repetition segmentation, and feedback rules."""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from config import EXERCISE_CONFIG


class AngleSmoother:
    """Moving-average smoother untuk meredam noise jitter MediaPipe."""
    def __init__(self, window: int = 5) -> None:
        self._buf: deque = deque(maxlen=window)

    def update(self, angle: float) -> float:
        self._buf.append(angle)
        return float(np.mean(self._buf))

    def reset(self) -> None:
        self._buf.clear()


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


class RepetitionSegmenter:
    """
    State machine untuk mendeteksi fase repetisi dengan:
    1. Smoothing sinyal sudut (AngleSmoother) — eliminasi jitter MediaPipe.
    2. Debouncing multi-frame — cegah transisi state akibat noise sesaat.
    3. Threshold delta konservatif — tidak terlalu sensitif.
    4. Toleransi sudut akhir lebih lebar — tidak macet di moving_down.
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

        print(
            f"[SM] state={self.state:<12} raw={raw_angle:6.1f}° "
            f"smooth={smooth_angle:6.1f}° diff={angle_diff:+6.2f}° "
            f"evt={event if event != self.state else '-':<10} "
            f"dbc={self._debounce_count}/{self.DEBOUNCE_FRAMES}"
        )
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

    def reset(self) -> None:
        self.state        = "idle"
        self.start_time   = 0.0
        self.peak_time    = 0.0
        self.end_time     = 0.0
        self._prev_smooth = None
        self._smoother.reset()
        self._reset_debounce()


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

    # -------------------------------------------------------------------------
    # ROOT CAUSE FIX — KeyError 'avg_visibility'
    #
    # pose_utils.compute_frame_angles() mengembalikan key "mean_visibility",
    # BUKAN "avg_visibility". Itulah satu-satunya penyebab seluruh error di log.
    #
    # Sebelumnya (kode asli):
    #   visibilities = [f["avg_visibility"] for f in features]  # ← CRASH
    #
    # Sesudahnya (perbaikan):
    #   visibilities = [f["mean_visibility"] for f in features]  # ← BENAR
    #
    # Referensi pose_utils.py baris compute_frame_angles():
    #   "mean_visibility": arm_landmarks.mean_visibility(),
    #   "min_visibility":  arm_landmarks.min_visibility(),
    # -------------------------------------------------------------------------
    upper_arm_angles = [f["upper_arm_angle"] for f in features]
    shoulder_angles  = [f["shoulder_angle"]  for f in features]
    torso_angles     = [f["torso_angle"]     for f in features]
    visibilities     = [f["mean_visibility"] for f in features]  # FIX: avg_ → mean_

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