from __future__ import annotations

from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
REPORT_DIR = PROJECT_ROOT / "reports"

# Path untuk menyimpan visualisasi hasil training (Cek Overfitting)
CONFUSION_MATRIX_PATH = REPORT_DIR / "xgboost_confusion_matrix.png"
LEARNING_CURVE_PATH = REPORT_DIR / "xgboost_learning_curve.png"

RAW_DATASET_PATH = DATASET_DIR / "data_training_biceps.csv"
CLEAN_DATASET_PATH = DATASET_DIR / "data_biceps_clean.csv"
REJECTED_SAMPLE_LOG = LOG_DIR / "rejected_sample_log.csv"
LIVE_PREDICTION_LOG = LOG_DIR / "live_prediction_log.csv"

MODEL_PKL_PATH = MODEL_DIR / "biceps_xgboost_model.pkl"
MODEL_JSON_PATH = MODEL_DIR / "xgboost_biceps_model.json"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
TRAINING_METRICS_PATH = MODEL_DIR / "training_metrics.json"
MODEL_VERSION_PATH = MODEL_DIR / "model_version.txt"
# -----------------------------------------------------------------------------
# Dataset schema
# -----------------------------------------------------------------------------
METADATA_COLUMNS = [
    "sample_id", "timestamp", "subject_id", "session_id", 
    "camera_position", "lighting_condition", "exercise_type", "active_arm",
]

# Core four plus temporal and drift features. Keep this exact order for training
# and real-time inference.
FEATURE_COLUMNS = [
    "rom_elbow", "upper_arm_angle_std", "torso_sway_range", "shoulder_angle_range",
    "rep_duration", "up_phase_duration", "down_phase_duration",
    "elbow_velocity_mean", "elbow_velocity_std", "motion_smoothness",
    "elbow_drift_range", "mean_visibility",
]

TARGET_COLUMNS = ["label", "error_type", "notes"]
DATASET_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS

VALID_BINARY_LABELS = {"correct", "incorrect"}
VALID_ERROR_TYPES = {
    "correct",
    "body_swing",      # Badan ikut bergerak
    "elbow_swing",     # Siku maju mundur (bahu tidak dikunci)
    "not_full_up",     # ROM tidak penuh
    "too_fast",        # Tempo terlalu cepat
    "unknown_incorrect",
}

# -----------------------------------------------------------------------------
# Pose and repetition thresholds
# -----------------------------------------------------------------------------
VISIBILITY_THRESHOLD = 0.60
MIN_VALID_FRAMES_PER_REP = 8
MIN_REP_DURATION_SECONDS = 0.60
MAX_REP_DURATION_SECONDS = 8.00

ELBOW_DOWN_ANGLE = 145.0
ELBOW_START_ANGLE = 130.0
ELBOW_UP_ANGLE = 70.0
ELBOW_COMPLETE_ANGLE = 140.0

MIN_ABSURD_ROM = 5.0
MAX_ABSURD_ROM = 180.0
MAX_ABSURD_ANGLE_RANGE = 180.0
MAX_ABSURD_STD = 90.0

RULE_MIN_ROM_ELBOW = 105.0
RULE_MAX_UPPER_ARM_STD = 14.0
RULE_MAX_SHOULDER_RANGE = 35.0
RULE_MAX_TORSO_SWAY = 18.0
RULE_MAX_ELBOW_DRIFT = 0.30
CONFIDENCE_THRESHOLD = 0.70

# -----------------------------------------------------------------------------
# MediaPipe landmark names used by the project
# -----------------------------------------------------------------------------
REQUIRED_LANDMARKS = [
    "shoulder",
    "elbow",
    "wrist",
    "hip",
]
