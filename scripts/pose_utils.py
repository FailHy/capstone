"""Pose and geometry helper functions.

This module is intentionally free from dataset and model logic. It only handles
landmark extraction, angle calculation, visibility validation, and small utility
functions used by collection and live inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

from config import VISIBILITY_THRESHOLD


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float
    visibility: float = 1.0

    def as_xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


@dataclass(frozen=True)
class ArmLandmarks:
    shoulder: Point2D
    elbow: Point2D
    wrist: Point2D
    hip: Point2D

    def mean_visibility(self) -> float:
        return float(
            np.mean(
                [
                    self.shoulder.visibility,
                    self.elbow.visibility,
                    self.wrist.visibility,
                    self.hip.visibility,
                ]
            )
        )

    def min_visibility(self) -> float:
        return float(
            min(
                self.shoulder.visibility,
                self.elbow.visibility,
                self.wrist.visibility,
                self.hip.visibility,
            )
        )


def calculate_angle(a: Iterable[float], b: Iterable[float], c: Iterable[float]) -> float:
    """Return the angle ABC in degrees, clamped to [0, 180]."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    c_arr = np.asarray(c, dtype=float)

    radians = np.arctan2(c_arr[1] - b_arr[1], c_arr[0] - b_arr[0]) - np.arctan2(
        a_arr[1] - b_arr[1], a_arr[0] - b_arr[0]
    )
    angle = abs(np.degrees(radians))
    if angle > 180.0:
        angle = 360.0 - angle
    return float(angle)


def angle_with_y_axis(p_top: Iterable[float], p_bottom: Iterable[float]) -> float:
    """Return absolute angle between a segment and the vertical y-axis."""
    top = np.asarray(p_top, dtype=float)
    bottom = np.asarray(p_bottom, dtype=float)
    dx = bottom[0] - top[0]
    dy = bottom[1] - top[1]
    return float(np.degrees(np.arctan2(abs(dx), abs(dy))))


def _landmark_to_point(landmark) -> Point2D:
    return Point2D(
        x=float(landmark.x),
        y=float(landmark.y),
        visibility=float(getattr(landmark, "visibility", 1.0)),
    )


def get_arm_landmarks(landmarks, mp_pose, arm: str, mirrored: bool = True) -> ArmLandmarks:
    """Extract shoulder, elbow, wrist, and hip landmarks for a selected arm.

    Args:
        landmarks: MediaPipe landmark list.
        mp_pose: mediapipe.solutions.pose module.
        arm: "left" or "right" from the user's perspective.
        mirrored: True if the frame was flipped before pose processing. This is
            consistent with the original project code.
    """
    if arm not in {"left", "right"}:
        raise ValueError("arm must be 'left' or 'right'")

    # If the image is mirrored before MediaPipe inference, MediaPipe's RIGHT
    # landmarks visually correspond to the user's left arm and vice versa.
    if mirrored:
        mp_side = "RIGHT" if arm == "left" else "LEFT"
    else:
        mp_side = "LEFT" if arm == "left" else "RIGHT"

    lm = mp_pose.PoseLandmark
    shoulder = _landmark_to_point(landmarks[getattr(lm, f"{mp_side}_SHOULDER").value])
    elbow = _landmark_to_point(landmarks[getattr(lm, f"{mp_side}_ELBOW").value])
    wrist = _landmark_to_point(landmarks[getattr(lm, f"{mp_side}_WRIST").value])
    hip = _landmark_to_point(landmarks[getattr(lm, f"{mp_side}_HIP").value])
    return ArmLandmarks(shoulder=shoulder, elbow=elbow, wrist=wrist, hip=hip)


def arm_visibility_ok(arm_landmarks: ArmLandmarks, threshold: float = VISIBILITY_THRESHOLD) -> bool:
    """Check whether all required landmarks are visible enough."""
    return arm_landmarks.min_visibility() >= threshold


def compute_frame_angles(arm_landmarks: ArmLandmarks) -> Dict[str, float]:
    """Compute per-frame biomechanical angles for a selected arm."""
    shoulder = arm_landmarks.shoulder.as_xy()
    elbow = arm_landmarks.elbow.as_xy()
    wrist = arm_landmarks.wrist.as_xy()
    hip = arm_landmarks.hip.as_xy()

    return {
        "elbow_angle": calculate_angle(shoulder, elbow, wrist),
        "shoulder_angle": calculate_angle(hip, shoulder, elbow),
        "upper_arm_angle": angle_with_y_axis(shoulder, elbow),
        "torso_angle": angle_with_y_axis(shoulder, hip),
        "elbow_x": float(arm_landmarks.elbow.x),
        "elbow_y": float(arm_landmarks.elbow.y),
        "shoulder_x": float(arm_landmarks.shoulder.x),
        "shoulder_y": float(arm_landmarks.shoulder.y),
        "wrist_x": float(arm_landmarks.wrist.x),
        "wrist_y": float(arm_landmarks.wrist.y),
        "mean_visibility": arm_landmarks.mean_visibility(),
        "min_visibility": arm_landmarks.min_visibility(),
    }


def get_elbow_angle_for_arm(landmarks, mp_pose, arm: str, mirrored: bool = True) -> Optional[float]:
    """Convenience helper used for rep start detection."""
    try:
        arm_landmarks = get_arm_landmarks(landmarks, mp_pose, arm=arm, mirrored=mirrored)
        if not arm_visibility_ok(arm_landmarks):
            return None
        return compute_frame_angles(arm_landmarks)["elbow_angle"]
    except Exception:
        return None
