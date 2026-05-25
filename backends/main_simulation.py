import time
import sys
from pathlib import Path

# 1. Dapatkan lokasi folder utama (myproject)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. Dapatkan lokasi folder scripts (myproject/scripts)
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# 3. Daftarkan folder scripts ke dalam sistem Python agar bisa di-import
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(SCRIPTS_DIR))

from core.session_manager import SessionManager

def main():
    print("--- 1. Initializing System Engine ---")
    try:
        session_manager = SessionManager()
    except Exception as e:
        print(f"Failed to load models (Did you run train_model.py first?): {e}")
        return
        
    print("\n--- 2. Starting Mobile App Session ---")
    session_id = session_manager.create_session(user_id="user_123", exercise_type="biceps")
    print(f"Session Created: {session_id}")
    
    # Retrieve the stateful service for this session
    evaluator = session_manager.get_session(session_id)
    
    print("\n--- 3. Simulating WebSocket Stream ---")
    
    # Mock sequence: Arm goes from straight (140 deg) -> curled (60 deg) -> straight (140 deg)
    # Using spatial coordinates to trick the geometry calculations
    mock_frames = [
        # Frame 1: Arm Straight Down (~140+ deg)
        {"shoulder": {"x": 0.5, "y": 0.2, "visibility": 0.99}, "elbow": {"x": 0.5, "y": 0.5, "visibility": 0.99}, "wrist": {"x": 0.5, "y": 0.8, "visibility": 0.99}, "hip": {"x": 0.5, "y": 0.7, "visibility": 0.99}},
        # Frame 2: Arm Starting to Curl
        {"shoulder": {"x": 0.5, "y": 0.2, "visibility": 0.99}, "elbow": {"x": 0.5, "y": 0.5, "visibility": 0.99}, "wrist": {"x": 0.6, "y": 0.6, "visibility": 0.99}, "hip": {"x": 0.5, "y": 0.7, "visibility": 0.99}},
        # Frame 3: Peak Curl (~90 deg or less)
        {"shoulder": {"x": 0.5, "y": 0.2, "visibility": 0.99}, "elbow": {"x": 0.5, "y": 0.5, "visibility": 0.99}, "wrist": {"x": 0.7, "y": 0.4, "visibility": 0.99}, "hip": {"x": 0.5, "y": 0.7, "visibility": 0.99}},
        # Frame 4: Moving Down
        {"shoulder": {"x": 0.5, "y": 0.2, "visibility": 0.99}, "elbow": {"x": 0.5, "y": 0.5, "visibility": 0.99}, "wrist": {"x": 0.6, "y": 0.6, "visibility": 0.99}, "hip": {"x": 0.5, "y": 0.7, "visibility": 0.99}},
        # Frame 5: Arm Straight Down (Should trigger 'completed')
        {"shoulder": {"x": 0.5, "y": 0.2, "visibility": 0.99}, "elbow": {"x": 0.5, "y": 0.5, "visibility": 0.99}, "wrist": {"x": 0.5, "y": 0.8, "visibility": 0.99}, "hip": {"x": 0.5, "y": 0.7, "visibility": 0.99}},
    ]
    
    # We loop it multiple times to simulate the "debouncing" mechanism in RepetitionSegmenter
    # The segmenter requires DEBOUNCE_FRAMES (usually 3) frames in the same direction to confirm a state change
    expanded_frames = []
    for frame in mock_frames:
        expanded_frames.extend([frame] * 4) # Duplicate each posture 4 times
        
    start_time = time.time()
    
    for i, payload in enumerate(expanded_frames):
        timestamp = start_time + (i * 0.1) # Simulate 10 FPS
        
        # Core Backend Engine logic happens here:
        result = evaluator.process_frame(timestamp, payload)
        
        print(f"Frame {i:02d} | Status: {result['status']:<10} | State: {result.get('state', '')}")
        
        if result['status'] == 'success':
             print(f"\n🔥 REP COMPLETE! Rep Count: {result['rep_count']}")
             print(f"🤖 ML Evaluation: {result['prediction']} (Smoothed: {result['smoothed_prediction']})")
             print(f"📈 Confidence: {result['confidence']:.2f}\n")
             
        time.sleep(0.05) # Just for terminal readability
        
    print("\n--- 4. Ending Session ---")
    stats = session_manager.delete_session(session_id)
    print(f"Workout Stats: {stats}")

if __name__ == "__main__":
    main()