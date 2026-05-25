import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from collections import deque

from feature_utils import (
    RepetitionSegmenter, 
    RepBuffer, 
    extract_repetition_features, 
    validate_repetition_quality
)
from pose_utils import ArmLandmarks, Point2D, arm_visibility_ok, compute_frame_angles

class PredictionSmoother:
    """Small repetition-level majority vote to reduce flicker across reps."""
    def __init__(self, window_size: int = 3) -> None:
        self.window = deque(maxlen=window_size)

    def update(self, label: str) -> str:
        if label == "uncertain":
            return label
        self.window.append(label)
        if not self.window:
            return label
        labels = list(self.window)
        return max(set(labels), key=labels.count)

class ExerciseEvaluatorService:
    def __init__(self, exercise_type: str, model: Any, label_encoder: Any, feature_columns: list):
        self.exercise_type = exercise_type
        self.model = model
        self.label_encoder = label_encoder
        self.feature_columns = feature_columns

        # Isolated state for THIS specific user session
        self.segmenter = RepetitionSegmenter(exercise_type=exercise_type)
        self.buffer = RepBuffer()
        self.smoother = PredictionSmoother(window_size=3)
        self.rep_count = 0

    def _parse_raw_landmarks(self, landmarks_dict: Dict[str, Dict[str, float]]) -> ArmLandmarks:
        """Converts raw JSON payload from Mobile/Frontend into PoseUtils objects."""
        def to_point(key: str) -> Point2D:
            data = landmarks_dict.get(key, {})
            return Point2D(
                x=float(data.get("x", 0.0)),
                y=float(data.get("y", 0.0)),
                visibility=float(data.get("visibility", 0.0))
            )
            
        return ArmLandmarks(
            shoulder=to_point("shoulder"),
            elbow=to_point("elbow"),
            wrist=to_point("wrist"),
            hip=to_point("hip")
        )

    def process_frame(self, timestamp: float, landmarks_dict: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Main entry point for incoming frame data."""
        arm_landmarks = self._parse_raw_landmarks(landmarks_dict)

        if not arm_visibility_ok(arm_landmarks):
            return {"status": "tracking", "state": self.segmenter.state, "message": "low_visibility"}

        frame_features = compute_frame_angles(arm_landmarks)
        event = self.segmenter.update(frame_features["elbow_angle"], timestamp)

        # Collect data if in active movement phase
        if event in {"started", "moving_up", "top", "up", "moving_down", "completed", "completed_partial", "down"}:
            self.buffer.append(timestamp, frame_features)

        # Evaluate rep upon completion
        if event in {"completed", "completed_partial"}:
            features = extract_repetition_features(self.buffer)
            is_valid, reason = validate_repetition_quality(self.buffer, features, self.exercise_type)

            # Reset local state for next rep
            self.buffer.clear()
            # Equivalent to reset(keep_down=True) used in live_evaluator.py
            self.segmenter.reset()

            if not is_valid:
                return {"status": "rejected", "reason": reason}

            # Run Inference
            self.rep_count += 1
            label, confidence = self._predict(features)
            smoothed_label = self.smoother.update(label)

            return {
                "status": "success",
                "rep_count": self.rep_count,
                "prediction": label,
                "smoothed_prediction": smoothed_label,
                "confidence": confidence,
                "features": features
            }

        return {"status": "tracking", "state": self.segmenter.state}

    def _predict(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Runs the XGBoost model natively on backend memory."""
        # Optional: Attempt to use the hybrid gatekeeper if it exists in feature_utils
        try:
            from feature_utils import hybrid_gatekeeper
            allowed_to_model, rule_label, rule_feedback = hybrid_gatekeeper(features)
            if not allowed_to_model:
                return rule_label, 1.0
        except ImportError:
            pass # Graceful fallback if gatekeeper isn't fully implemented in utils yet

        # Align features perfectly to model schema
        X_live = pd.DataFrame([{col: float(features.get(col, 0.0)) for col in self.feature_columns}])

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_live)[0]
            pred_idx = int(np.argmax(proba))
            confidence = float(np.max(proba))
        else:
            pred_idx = int(self.model.predict(X_live)[0])
            confidence = 1.0

        if self.label_encoder is not None:
            label = str(self.label_encoder.inverse_transform([pred_idx])[0])
        else:
            label = "correct" if pred_idx == 0 else "incorrect"

        return label, confidence