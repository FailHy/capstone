import pickle
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

# Assuming this runs where config.py is accessible in the python path
try:
    from config import MODEL_PKL_PATH, TRICEPS_MODEL_PKL_PATH, FEATURE_COLUMNS_PATH, LABEL_ENCODER_PATH
except ImportError:
    # Fallback paths if config.py is not in path
    MODEL_PKL_PATH = Path("models/biceps_xgboost_model.pkl")
    TRICEPS_MODEL_PKL_PATH = Path("models/triceps_xgboost_model.pkl")
    FEATURE_COLUMNS_PATH = Path("models/feature_columns_biceps.json")
    LABEL_ENCODER_PATH = Path("models/label_encoder.pkl")

class ModelLoader:
    _instance = None
    _models: Dict[str, Any] = {}
    _label_encoder = None
    _feature_columns = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_all()
        return cls._instance

    def _load_all(self) -> None:
        logging.info("Loading ML artifacts into memory...")
        
        # Load schema
        if FEATURE_COLUMNS_PATH.exists():
            with FEATURE_COLUMNS_PATH.open("r") as f:
                self._feature_columns = json.load(f)
        else:
            logging.warning(f"Feature schema not found at {FEATURE_COLUMNS_PATH}")
        
        # Load encoder
        if LABEL_ENCODER_PATH.exists():
            with LABEL_ENCODER_PATH.open("rb") as f:
                self._label_encoder = pickle.load(f)
                
        # Load specific exercise models
        if MODEL_PKL_PATH.exists():
            with MODEL_PKL_PATH.open("rb") as f:
                self._models["biceps"] = pickle.load(f)
                
        if TRICEPS_MODEL_PKL_PATH.exists():
            with TRICEPS_MODEL_PKL_PATH.open("rb") as f:
                self._models["triceps"] = pickle.load(f)

    def get_artifacts(self, exercise_type: str) -> Tuple[Any, Any, list]:
        """Returns (model, label_encoder, feature_columns)"""
        if exercise_type not in self._models:
            raise ValueError(f"Model for '{exercise_type}' is not loaded or does not exist.")
        return self._models[exercise_type], self._label_encoder, self._feature_columns