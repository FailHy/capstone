from __future__ import annotations
from pathlib import Path

# =============================================================================
# FILE PATHS & DIRECTORIES
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
REPORT_DIR = PROJECT_ROOT / "reports"

# Ensure all directories exist
for d in [DATASET_DIR, MODEL_DIR, LOG_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset Paths
RAW_DATASET_PATH = DATASET_DIR / "data_training_harmonized.csv"
CLEAN_DATASET_PATH = DATASET_DIR / "data_clean.csv"

# Model Artifact Paths (Biceps)
MODEL_PKL_PATH = MODEL_DIR / "biceps_xgboost_model.pkl"
MODEL_JSON_PATH = MODEL_DIR / "xgboost_biceps_model.json"

# Model Artifact Paths (Triceps)
TRICEPS_MODEL_PKL_PATH = MODEL_DIR / "triceps_xgboost_model.pkl"
TRICEPS_MODEL_JSON_PATH = MODEL_DIR / "xgboost_triceps_model.json"

# Shared Artifacts
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
MODEL_VERSION_PATH = MODEL_DIR / "model_version.txt"
TRAINING_METRICS_PATH = MODEL_DIR / "training_metrics.json"

# Log Paths
LIVE_PREDICTION_LOG = LOG_DIR / "live_prediction_log.csv"
REJECTED_SAMPLE_LOG = LOG_DIR / "rejected_sample_log.csv"

# Report Paths
CONFUSION_MATRIX_PATH = REPORT_DIR / "xgboost_confusion_matrix.png"
LEARNING_CURVE_PATH = REPORT_DIR / "xgboost_learning_curve.png"

# =============================================================================
# EXERCISE BIOMECHANICS CONFIGURATION (HYBRID GATEKEEPER)
# =============================================================================
VISIBILITY_THRESHOLD = 0.5  # Minimum MediaPipe landmark visibility

# EXERCISE_CONFIG dict to handle multiple exercises dynamically
EXERCISE_CONFIG = {
    "biceps": {
        "start_angle": 130.0,  # Arms straight down
        "peak_angle": 70.0,    # Arms curled up
        "direction": "up",     # Concentric phase means angle decreases
        "rom_min_absurd": 5.0, # Less than 5 deg ROM is not a rep
    },
    "triceps": {
        "start_angle": 70.0,   # Arms curled up/forward
        "peak_angle": 150.0,   # Arms pushed down/back straight
        "direction": "down",   # Concentric phase means angle increases
        "rom_min_absurd": 5.0,
    }
}

# Legacy fallback (for backward compatibility if needed in other scripts)
ELBOW_START_ANGLE = 130.0
ELBOW_UP_ANGLE = 70.0
MIN_ABSURD_ROM = 5.0
MAX_ABSURD_ROM = 180.0         # ROM above 180 deg is physically impossible
MAX_ABSURD_ANGLE_RANGE = 360.0 # Angle range above 360 deg is absurd
MAX_ABSURD_STD = 180.0         # Std deviation above 180 deg is absurd

# -----------------------------------------------------------------------------
# MediaPipe landmark names used by the project
# -----------------------------------------------------------------------------
MP_LANDMARKS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================
FEATURE_COLUMNS = [
    "rom_elbow",
    "upper_arm_angle_std",
    "torso_sway_range",
    "shoulder_angle_range",
    "rep_duration",
    "up_phase_duration",
    "down_phase_duration",
    "elbow_velocity_mean",
    "elbow_velocity_std",
    "motion_smoothness",
    "elbow_drift_range",
    "mean_visibility",
]

DATASET_COLUMNS = [
    "sample_id",
    "timestamp",
    "subject_id",
    "session_id",
    "camera_position",
    "lighting_condition",
    "exercise_type",
    "active_arm",
] + FEATURE_COLUMNS + [
    "label",
    "error_type",
    "notes"
]

# Supported Target Labels
ALLOWED_LABELS = {
    "correct",         # Gerakan benar
    "body_swing",      # Badan ikut bergerak
    "elbow_swing",     # Siku maju mundur (bahu tidak dikunci)
    "not_full_up",     # ROM tidak penuh
    "too_fast",        # Tempo terlalu cepat
    "unknown_incorrect",
}

# Binary labels used by clean_dataset
VALID_BINARY_LABELS = {"correct", "incorrect"}

# Valid error type labels (superset of ALLOWED_LABELS)
VALID_ERROR_TYPES = ALLOWED_LABELS | {"unknown_incorrect"}

# =============================================================================
# COLUMN GROUPS (used by clean_dataset.py)
# =============================================================================
METADATA_COLUMNS = [
    "sample_id",
    "timestamp",
    "subject_id",
    "session_id",
    "camera_position",
    "lighting_condition",
    "exercise_type",
    "active_arm",
]

TARGET_COLUMNS = [
    "label",
    "error_type",
    "notes",
]