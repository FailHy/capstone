"""Feature extraction, repetition segmentation, and feedback rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    CONFIDENCE_THRESHOLD,
    ELBOW_COMPLETE_ANGLE,
    ELBOW_DOWN_ANGLE,
    ELBOW_START_ANGLE,
    ELBOW_UP_ANGLE,
    FEATURE_COLUMNS,
    MAX_ABSURD_ANGLE_RANGE,
    MAX_ABSURD_ROM,
    MAX_ABSURD_STD,
    MAX_REP_DURATION_SECONDS,
    MIN_ABSURD_ROM,
    MIN_REP_DURATION_SECONDS,
    MIN_VALID_FRAMES_PER_REP,
    RULE_MAX_ELBOW_DRIFT,
    RULE_MAX_SHOULDER_RANGE,
    RULE_MAX_TORSO_SWAY,
    RULE_MAX_UPPER_ARM_STD,
    RULE_MIN_ROM_ELBOW,
    VISIBILITY_THRESHOLD,
)


@dataclass
class RepBuffer:
    """Stores per-frame values for one repetition."""

    timestamps: List[float] = field(default_factory=list)
    elbow_angles: List[float] = field(default_factory=list)
    shoulder_angles: List[float] = field(default_factory=list)
    upper_arm_angles: List[float] = field(default_factory=list)
    torso_angles: List[float] = field(default_factory=list)
    elbow_x: List[float] = field(default_factory=list)
    elbow_y: List[float] = field(default_factory=list)
    shoulder_x: List[float] = field(default_factory=list)
    shoulder_y: List[float] = field(default_factory=list)
    wrist_x: List[float] = field(default_factory=list)
    wrist_y: List[float] = field(default_factory=list)
    visibilities: List[float] = field(default_factory=list)

    def clear(self) -> None:
        for key in list(self.__dict__.keys()):
            getattr(self, key).clear()

    def append(self, timestamp: float, frame_features: Dict[str, float]) -> None:
        self.timestamps.append(float(timestamp))
        self.elbow_angles.append(float(frame_features["elbow_angle"]))
        self.shoulder_angles.append(float(frame_features["shoulder_angle"]))
        self.upper_arm_angles.append(float(frame_features["upper_arm_angle"]))
        self.torso_angles.append(float(frame_features["torso_angle"]))
        self.elbow_x.append(float(frame_features["elbow_x"]))
        self.elbow_y.append(float(frame_features["elbow_y"]))
        self.shoulder_x.append(float(frame_features["shoulder_x"]))
        self.shoulder_y.append(float(frame_features["shoulder_y"]))
        self.wrist_x.append(float(frame_features["wrist_x"]))
        self.wrist_y.append(float(frame_features["wrist_y"]))
        self.visibilities.append(float(frame_features["mean_visibility"]))

    def __len__(self) -> int:
        return len(self.timestamps)


@dataclass
class RepetitionSegmenter:
    state: str = "idle"
    start_time: Optional[float] = None
    top_time: Optional[float] = None
    down_start_time: Optional[float] = None

    def reset(self, keep_down: bool = False) -> None:
        self.state = "down" if keep_down else "idle"
        self.start_time = None
        self.top_time = None
        self.down_start_time = None

    def update(self, elbow_angle: float, timestamp: float) -> str:
        angle = float(elbow_angle)
        t = float(timestamp)

        if self.state == "idle":
            if angle >= ELBOW_DOWN_ANGLE:
                self.state = "down"
                return "ready"
            if angle < ELBOW_START_ANGLE:
                self.state = "moving_up"
                self.start_time = t
                return "started"
            return "idle"

        if self.state == "down":
            if angle < ELBOW_START_ANGLE:
                self.state = "moving_up"
                self.start_time = t
                return "started"
            return "down"

        if self.state == "moving_up":
            if angle <= ELBOW_UP_ANGLE:
                self.state = "up"
                self.top_time = t
                return "top"
            if angle >= ELBOW_COMPLETE_ANGLE and self.start_time is not None:
                # The user returned to the bottom without reaching the target
                # top angle. Keep it as a valid completed rep candidate so
                # half-rep hard negatives can be collected and labeled.
                if (t - self.start_time) >= MIN_REP_DURATION_SECONDS:
                    self.state = "completed"
                    return "completed_partial"
                self.reset(keep_down=True)
                return "false_start"
            return "moving_up"

        if self.state == "up":
            if angle > ELBOW_UP_ANGLE + 10.0:
                self.state = "moving_down"
                self.down_start_time = t
                return "moving_down"
            return "up"

        if self.state == "moving_down":
            if angle >= ELBOW_COMPLETE_ANGLE:
                self.state = "completed"
                return "completed"
            return "moving_down"

        if self.state == "completed":
            return "completed"

        raise RuntimeError(f"Unknown repetition state: {self.state}")


def percentile_range(values: List[float], low: float = 5.0, high: float = 95.0) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, high) - np.percentile(arr, low))


def robust_percentile(values: List[float], percentile: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, percentile))


def _safe_velocity(angle_values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    if len(angle_values) < 2:
        return np.array([], dtype=float)
    dt = np.diff(timestamps)
    da = np.diff(angle_values)
    valid = dt > 1e-6
    if not np.any(valid):
        return np.array([], dtype=float)
    return np.abs(da[valid] / dt[valid])


def _motion_smoothness(angle_values: np.ndarray, timestamps: np.ndarray) -> float:
    """Return a simple jerk-like score; lower means smoother movement."""
    if len(angle_values) < 4:
        return 0.0
    velocity = _safe_velocity(angle_values, timestamps)
    if len(velocity) < 2:
        return 0.0
    dt = np.diff(timestamps[: len(velocity) + 1])
    valid = dt[1:] > 1e-6 if len(dt) > 1 else np.array([], dtype=bool)
    if len(valid) == 0 or not np.any(valid):
        return 0.0
    acceleration = np.diff(velocity)[valid]
    return float(np.mean(np.abs(acceleration)))


def _elbow_drift_range(buffer: RepBuffer) -> float:
    elbow_x = np.asarray(buffer.elbow_x, dtype=float)
    elbow_y = np.asarray(buffer.elbow_y, dtype=float)
    shoulder_x = np.asarray(buffer.shoulder_x, dtype=float)
    shoulder_y = np.asarray(buffer.shoulder_y, dtype=float)

    if len(elbow_x) == 0:
        return float("nan")

    rel = np.column_stack([elbow_x - shoulder_x, elbow_y - shoulder_y])
    start = rel[0]
    dist = np.linalg.norm(rel - start, axis=1)
    return float(np.percentile(dist, 95) - np.percentile(dist, 5))


def extract_repetition_features(buffer: RepBuffer) -> Dict[str, float]:
    """Extract robust per-repetition features.

    Uses percentile ranges instead of raw min/max to reduce MediaPipe jitter.
    """
    if len(buffer) == 0:
        return {col: float("nan") for col in FEATURE_COLUMNS}

    t = np.asarray(buffer.timestamps, dtype=float)
    elbow = np.asarray(buffer.elbow_angles, dtype=float)

    rep_duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    top_idx = int(np.argmin(elbow)) if len(elbow) else 0
    up_phase_duration = float(max(0.0, t[top_idx] - t[0])) if len(t) else 0.0
    down_phase_duration = float(max(0.0, t[-1] - t[top_idx])) if len(t) else 0.0

    velocity = _safe_velocity(elbow, t)
    elbow_velocity_mean = float(np.mean(velocity)) if len(velocity) else 0.0
    elbow_velocity_std = float(np.std(velocity)) if len(velocity) else 0.0

    features = {
        "rom_elbow": percentile_range(buffer.elbow_angles),
        "upper_arm_angle_std": float(np.nanstd(np.asarray(buffer.upper_arm_angles, dtype=float))),
        "torso_sway_range": percentile_range(buffer.torso_angles),
        "shoulder_angle_range": percentile_range(buffer.shoulder_angles),
        "rep_duration": rep_duration,
        "up_phase_duration": up_phase_duration,
        "down_phase_duration": down_phase_duration,
        "elbow_velocity_mean": elbow_velocity_mean,
        "elbow_velocity_std": elbow_velocity_std,
        "motion_smoothness": _motion_smoothness(elbow, t),
        "elbow_drift_range": _elbow_drift_range(buffer),
        "mean_visibility": float(np.mean(buffer.visibilities)) if buffer.visibilities else 0.0,
    }

    return {col: round(float(features[col]), 6) for col in FEATURE_COLUMNS}


def validate_repetition_quality(buffer: RepBuffer, features: Dict[str, float]) -> Tuple[bool, str]:
    """Data-quality validation only. Does not label correct/incorrect form."""
    if len(buffer) < MIN_VALID_FRAMES_PER_REP:
        return False, f"too_few_frames:{len(buffer)}"

    if features.get("mean_visibility", 0.0) < VISIBILITY_THRESHOLD:
        return False, f"low_visibility:{features.get('mean_visibility', 0.0):.3f}"

    duration = features.get("rep_duration", 0.0)
    if duration < MIN_REP_DURATION_SECONDS:
        return False, f"too_fast:{duration:.3f}"
    if duration > MAX_REP_DURATION_SECONDS:
        return False, f"too_slow:{duration:.3f}"

    rom = features.get("rom_elbow", float("nan"))
    if not np.isfinite(rom) or rom < MIN_ABSURD_ROM or rom > MAX_ABSURD_ROM:
        return False, f"absurd_rom:{rom:.3f}"

    for col in ["torso_sway_range", "shoulder_angle_range"]:
        val = features.get(col, float("nan"))
        if not np.isfinite(val) or val < 0 or val > MAX_ABSURD_ANGLE_RANGE:
            return False, f"absurd_{col}:{val:.3f}"

    upper_std = features.get("upper_arm_angle_std", float("nan"))
    if not np.isfinite(upper_std) or upper_std < 0 or upper_std > MAX_ABSURD_STD:
        return False, f"absurd_upper_arm_angle_std:{upper_std:.3f}"

    return True, "ok"

# hybrid gatekeeper untuk antisipator
def hybrid_gatekeeper(features: Dict[str, float]) -> Tuple[bool, str, str]:
    """Return (allowed_to_model, label, feedback).
    Penyelarasan aturan heuristik dengan terminologi pakar.
    """
    if features["rom_elbow"] < RULE_MIN_ROM_ELBOW:
        return False, "incorrect", "Not full up: Gerakan belum penuh, rentangkan tangan lebih jauh."

    if features["torso_sway_range"] > RULE_MAX_TORSO_SWAY:
        return False, "incorrect", "Body swing: Badan ikut mengayun. Kunci postur Anda."

    # Gabungan evaluasi bahu dan lengan atas untuk mendeteksi siku yang maju-mundur
    if features["shoulder_angle_range"] > RULE_MAX_SHOULDER_RANGE or features["upper_arm_angle_std"] > RULE_MAX_UPPER_ARM_STD:
        return False, "incorrect", "Elbow swing: Siku bergeser karena bahu tidak dikunci."

    if features["elbow_drift_range"] > RULE_MAX_ELBOW_DRIFT:
        return False, "incorrect", "Elbow swing: Siku berpindah dari sisi tubuh."

    return True, "unknown", ""

def feedback_from_prediction(label: str, confidence: float, features: Dict[str, float]) -> str:
    """Convert ML output into user-facing feedback."""
    if confidence < CONFIDENCE_THRESHOLD:
        return "Gerakan belum dapat dipastikan; ulangi dengan posisi kamera dan tubuh lebih jelas."

    if label == "correct":
        return "Good form. Gerakan terlihat stabil."

    # If the model predicts incorrect but no rule fired, choose the most likely
    # explanation by feature priority.
    _, _, rule_feedback = hybrid_gatekeeper(features)
    if rule_feedback:
        return rule_feedback

    return "Gerakan kurang tepat; ulangi lebih stabil dan kontrol tempo gerakan."
