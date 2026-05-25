from pydantic import BaseModel, Field
from typing import Dict, Optional, Any

# --- Sub-models for Landmarks ---
class Point2D(BaseModel):
    x: float
    y: float
    visibility: float

class LandmarksPayload(BaseModel):
    shoulder: Point2D
    elbow: Point2D
    wrist: Point2D
    hip: Point2D

# --- API Request Models ---
class SessionStartRequest(BaseModel):
    user_id: str
    exercise_type: str = Field(default="biceps", description="Type of exercise, e.g., 'biceps' or 'triceps'")

class SessionEndRequest(BaseModel):
    session_id: str

class PredictRequest(BaseModel):
    session_id: str
    timestamp: float
    landmarks: LandmarksPayload

# --- API Response Models ---
class SessionStartResponse(BaseModel):
    session_id: str
    message: str

class PredictResponse(BaseModel):
    status: str
    state: Optional[str] = None
    rep_count: Optional[int] = None
    prediction: Optional[str] = None
    smoothed_prediction: Optional[str] = None
    confidence: Optional[float] = None
    message: Optional[str] = None
    features: Optional[Dict[str, Any]] = None

class SessionEndResponse(BaseModel):
    status: str
    total_reps: int
    exercise_type: str