import uuid
from typing import Dict, Any

from .model_loader import ModelLoader
from .evaluator_service import ExerciseEvaluatorService

class SessionManager:
    def __init__(self):
        # In-memory store (dict) holding active sessions
        self._sessions: Dict[str, ExerciseEvaluatorService] = {}
        self.model_loader = ModelLoader()  # Automatically utilizes the singleton

    def create_session(self, user_id: str, exercise_type: str = "biceps") -> str:
        """Allocates a new ML service state for a user."""
        model, label_encoder, feature_columns = self.model_loader.get_artifacts(exercise_type)
        
        service = ExerciseEvaluatorService(
            exercise_type=exercise_type,
            model=model,
            label_encoder=label_encoder,
            feature_columns=feature_columns
        )
        
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = service
        return session_id

    def get_session(self, session_id: str) -> ExerciseEvaluatorService:
        """Retrieves active session logic processor."""
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id} not found or expired.")
        return self._sessions[session_id]

    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """Cleans up memory and returns final workout stats."""
        if session_id in self._sessions:
            service = self._sessions.pop(session_id)
            return {
                "status": "session_ended", 
                "total_reps": service.rep_count,
                "exercise_type": service.exercise_type
            }
        return {"status": "error", "message": "not_found"}