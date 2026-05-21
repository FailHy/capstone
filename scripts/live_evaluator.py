"""Real-time biceps evaluator with hybrid rule-based + XGBoost inference."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

try:
    import pygame
except Exception:  # pragma: no cover - depends on local audio backend
    pygame = None

from config import (
    CONFIDENCE_THRESHOLD,
    ELBOW_START_ANGLE,
    FEATURE_COLUMNS_PATH,
    LABEL_ENCODER_PATH,
    LIVE_PREDICTION_LOG,
    MODEL_PKL_PATH,
    REJECTED_SAMPLE_LOG,
    VISIBILITY_THRESHOLD,
)
from feature_utils import (
    RepBuffer,
    RepetitionSegmenter,
    extract_repetition_features,
    feedback_from_prediction,
    hybrid_gatekeeper,
    validate_repetition_quality,
)
from pose_utils import arm_visibility_ok, compute_frame_angles, get_arm_landmarks


def init_audio() -> None:
    if pygame is None:
        return
    try:
        pygame.mixer.init()
    except Exception:
        pass


def play_beep(is_correct: bool) -> None:
    if pygame is None:
        return
    try:
        pygame.mixer.music.stop()
        sample_rate = 44100
        duration = 0.25
        frequency = 880 if is_correct else 220
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = (np.sin(frequency * 2 * np.pi * t) * 32767).astype(np.int16)
        stereo_wave = np.column_stack((wave, wave))
        pygame.sndarray.make_sound(stereo_wave).play()
    except Exception:
        pass


def load_artifacts(model_path: Path, feature_path: Path, label_encoder_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    with model_path.open("rb") as f:
        model = pickle.load(f)

    if not feature_path.exists():
        raise FileNotFoundError(f"Feature schema not found: {feature_path}")
    with feature_path.open("r") as f:
        feature_columns = json.load(f)

    label_encoder = None
    if label_encoder_path.exists():
        with label_encoder_path.open("rb") as f:
            label_encoder = pickle.load(f)

    return model, feature_columns, label_encoder


def choose_active_arm(left_angle: Optional[float], right_angle: Optional[float]) -> Optional[str]:
    candidates = []
    if left_angle is not None and left_angle < ELBOW_START_ANGLE:
        candidates.append((left_angle, "left"))
    if right_angle is not None and right_angle < ELBOW_START_ANGLE:
        candidates.append((right_angle, "right"))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def put_text(frame, text: str, y: int, color=(255, 255, 255), scale: float = 0.58) -> None:
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def log_csv(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def predict_label(model, label_encoder, feature_columns: List[str], features: Dict[str, float]) -> Tuple[str, float]:
    X_live = pd.DataFrame([{col: float(features[col]) for col in feature_columns}])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_live)[0]
        pred_idx = int(np.argmax(proba))
        confidence = float(np.max(proba))
    else:
        pred_idx = int(model.predict(X_live)[0])
        confidence = 1.0

    if label_encoder is not None:
        label = str(label_encoder.inverse_transform([pred_idx])[0])
    else:
        label = "correct" if pred_idx == 0 else "incorrect"

    return label, confidence


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


def evaluate_rep(model, label_encoder, feature_columns: List[str], features: Dict[str, float]) -> Tuple[str, float, str, str]:
    allowed_to_model, rule_label, rule_feedback = hybrid_gatekeeper(features)
    if not allowed_to_model:
        return rule_label, 1.0, rule_feedback, "rule_gatekeeper"

    label, confidence = predict_label(model, label_encoder, feature_columns, features)
    if confidence < CONFIDENCE_THRESHOLD:
        return "uncertain", confidence, feedback_from_prediction(label, confidence, features), "ml_low_confidence"

    feedback = feedback_from_prediction(label, confidence, features)
    return label, confidence, feedback, "xgboost"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time movement evaluator.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=MODEL_PKL_PATH)
    parser.add_argument("--features", type=Path, default=FEATURE_COLUMNS_PATH)
    parser.add_argument("--label-encoder", type=Path, default=LABEL_ENCODER_PATH)
    parser.add_argument("--subject-id", default="live_user")
    parser.add_argument("--session-id", default=datetime.utcnow().strftime("live_%Y%m%d_%H%M%S"))
    parser.add_argument("--exercise-type", choices=["biceps", "triceps"], default="biceps")
    parser.add_argument("--mirrored", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_audio()
    model, feature_columns, label_encoder = load_artifacts(args.model, args.features, args.label_encoder)
    print(f"INFO: loaded model = {args.model}")
    print(f"INFO: feature schema = {feature_columns}")

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    draw_spec_landmark = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
    draw_spec_connection = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    segmenter = RepetitionSegmenter()
    buffer = RepBuffer()
    smoother = PredictionSmoother(window_size=3)
    active_arm: Optional[str] = None
    counter = 0
    status = "Waiting for rep..."
    last_label = "waiting"
    last_feedback = ""
    last_confidence = 0.0
    last_source = "none"
    prev_time = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            if args.mirrored:
                frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            left_angle = None
            right_angle = None

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                try:
                    left_lm = get_arm_landmarks(landmarks, mp_pose, "left", mirrored=args.mirrored)
                    right_lm = get_arm_landmarks(landmarks, mp_pose, "right", mirrored=args.mirrored)
                    if arm_visibility_ok(left_lm):
                        left_angle = compute_frame_angles(left_lm)["elbow_angle"]
                    if arm_visibility_ok(right_lm):
                        right_angle = compute_frame_angles(right_lm)["elbow_angle"]

                    if active_arm is None:
                        selected = choose_active_arm(left_angle, right_angle)
                        if selected is not None:
                            active_arm = selected
                            buffer.clear()
                            segmenter.reset(keep_down=True)
                            status = f"Recording {active_arm.upper()}"

                    if active_arm is not None:
                        arm_lm = get_arm_landmarks(landmarks, mp_pose, active_arm, mirrored=args.mirrored)
                        if not arm_visibility_ok(arm_lm, threshold=VISIBILITY_THRESHOLD):
                            status = "Low visibility: adjust camera/body"
                        else:
                            frame_features = compute_frame_angles(arm_lm)
                            event = segmenter.update(frame_features["elbow_angle"], now)

                            if event in {"started", "moving_up", "top", "up", "moving_down", "completed", "completed_partial", "down"}:
                                buffer.append(now, frame_features)

                            if event == "false_start":
                                log_csv(
                                    REJECTED_SAMPLE_LOG,
                                    {
                                        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                                        "session_id": args.session_id,
                                        "subject_id": args.subject_id,
                                        "reason": "false_start",
                                        "active_arm": active_arm,
                                    },
                                )
                                buffer.clear()
                                active_arm = None
                                status = "False start. Try again."

                            if event in {"completed", "completed_partial"}:
                                features = extract_repetition_features(buffer)
                                is_valid, reason = validate_repetition_quality(buffer, features)
                                counter += 1
                                if is_valid:
                                    label, confidence, feedback, source = evaluate_rep(
                                        model, label_encoder, feature_columns, features
                                    )
                                    smoothed_label = smoother.update(label)
                                    last_label = smoothed_label
                                    last_confidence = confidence
                                    last_feedback = feedback
                                    last_source = source
                                    status = feedback

                                    print(f"\nREP {counter} | label={label} | smoothed={smoothed_label} | confidence={confidence:.3f} | source={source}")
                                    for col in feature_columns:
                                        print(f"  {col}: {features[col]:.4f}")

                                    log_row = {
                                        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                                        "subject_id": args.subject_id,
                                        "session_id": args.session_id,
                                        "exercise_type": args.exercise_type,
                                        "rep_index": counter,
                                        "active_arm": active_arm,
                                        "prediction": label,
                                        "smoothed_prediction": smoothed_label,
                                        "confidence": round(confidence, 6),
                                        "source": source,
                                        "feedback": feedback,
                                        **features,
                                    }
                                    log_csv(LIVE_PREDICTION_LOG, log_row)
                                    threading.Thread(target=play_beep, args=(smoothed_label == "correct",), daemon=True).start()
                                else:
                                    status = f"Dropped: {reason}"
                                    log_csv(
                                        REJECTED_SAMPLE_LOG,
                                        {
                                            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                                            "subject_id": args.subject_id,
                                            "session_id": args.session_id,
                                            "reason": reason,
                                            "active_arm": active_arm,
                                            **features,
                                        },
                                    )

                                buffer.clear()
                                segmenter.reset(keep_down=True)
                                active_arm = None

                except Exception as exc:
                    status = f"Pose/eval error: {exc}"

                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    draw_spec_landmark,
                    draw_spec_connection,
                )
            else:
                status = "No pose detected"

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            # UI overlay.
            cv2.rectangle(frame, (0, 0), (width, 130), (0, 0, 0), -1)
            label_color = (0, 255, 0) if last_label == "correct" else (0, 0, 255) if last_label == "incorrect" else (0, 255, 255)
            put_text(frame, f"FPS: {fps:.1f} | Reps: {counter} | Arm: {active_arm} | State: {segmenter.state}", 25)
            put_text(frame, f"Prediction: {last_label} | Conf: {last_confidence:.2f} | Source: {last_source}", 55, label_color, 0.65)
            put_text(frame, f"Feedback: {last_feedback or status}", 85, label_color, 0.55)
            put_text(frame, "Press Q to exit", 115, (255, 255, 255), 0.50)

            if left_angle is not None:
                put_text(frame, f"Left elbow: {left_angle:.1f}", height - 45)
            if right_angle is not None:
                put_text(frame, f"Right elbow: {right_angle:.1f}", height - 20)

            cv2.imshow("Biceps Real-Time Hybrid Evaluator", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
