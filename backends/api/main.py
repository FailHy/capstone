import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Ensure the root project paths are available for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "scripts"))

from backends.core.session_manager import SessionManager
from backends.api.schemas import (
    SessionStartRequest, SessionStartResponse,
    PredictRequest, PredictResponse,
    SessionEndRequest, SessionEndResponse
)

# Global Session Manager instance
session_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event to initialize ML models once at startup."""
    global session_manager
    print("🚀 Starting API and loading ML models into memory...")
    try:
        session_manager = SessionManager()
        print("✅ Models loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
    
    yield # App is running
    
    print("🛑 Shutting down API...")
    # Clean up all active sessions on shutdown
    if session_manager:
        session_manager._sessions.clear()

app = FastAPI(title="Exercise AI Backend", lifespan=lifespan)

# Allow Flutter app to connect from any origin (Update in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/session/start", response_model=SessionStartResponse)
async def start_session(request: SessionStartRequest):
    """Initializes a new stateful PredictionService for the user."""
    try:
        session_id = session_manager.create_session(
            user_id=request.user_id, 
            exercise_type=request.exercise_type
        )
        return SessionStartResponse(session_id=session_id, message="Session created successfully")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@app.post("/predict", response_model=PredictResponse)
async def predict_frame(request: PredictRequest):
    """
    Receives streaming landmarks.
    Returns 'tracking' normally, and 'success' when a rep is completed.
    """
    try:
        # Retrieve the specific PredictionService (ExerciseEvaluatorService) for this session
        evaluator = session_manager.get_session(request.session_id)
        
        # Convert Pydantic model to dictionary format expected by the core engine
        landmarks_dict = request.landmarks.model_dump()
        
        # Process the frame through the state machine
        result = evaluator.process_frame(request.timestamp, landmarks_dict)
        
        return PredictResponse(**result)

    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or expired")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/session/end", response_model=SessionEndResponse)
async def end_session(request: SessionEndRequest):
    """Deletes the session state and returns final workout metrics."""
    result = session_manager.delete_session(request.session_id)
    
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
        
    return SessionEndResponse(**result)