import sys
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

# Memastikan jalur akses proyek
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "scripts"))

from backends.core.session_manager import SessionManager
from backends.api.schemas import (
    SessionStartRequest, SessionStartResponse,
    SessionEndRequest, SessionEndResponse,
    PredictRequest  # Kita gunakan ini untuk memvalidasi JSON masuk dari WS
)

session_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global session_manager
    print("🚀 Memulai API dan memuat model ML...")
    session_manager = SessionManager()
    
    # Memulai task pembersihan sesi di latar belakang (Mencegah Memory Leak)
    # Catatan: Fungsi `cleanup_loop` harus sudah ada di SessionManager
    cleanup_task = asyncio.create_task(session_manager.cleanup_loop())
    
    yield
    
    print("🛑 Mematikan API...")
    cleanup_task.cancel()
    # Fungsi penutupan yang aman (harus diimplementasikan di SessionManager)
    await session_manager.shutdown()

app = FastAPI(title="Exercise AI Backend (Real-Time WebSocket)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/session/start", response_model=SessionStartResponse)
async def start_session(request: SessionStartRequest):
    """REST: Menginisialisasi sesi dan mengembalikan ID untuk digunakan dalam WebSocket."""
    try:
        # Asumsikan create_session sekarang asinkron karena lock di SessionManager
        session_id = await session_manager.create_session(
            user_id=request.user_id, 
            exercise_type=request.exercise_type
        )
        return SessionStartResponse(session_id=session_id, message="Sesi dibuat. Hubungkan ke WS.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WEBSOCKET: Endpoint aliran berfrekuensi tinggi (menggantikan /predict)."""
    await websocket.accept()
    
    # Verifikasi apakah sesi ada
    try:
        # Asumsikan get_session asinkron
        evaluator = await session_manager.get_session(session_id)
    except KeyError:
        await websocket.send_json({"error": "Sesi tidak valid atau telah berakhir"})
        await websocket.close(code=1008) # 1008: Policy Violation
        return

    last_state = None

    try:
        while True:
            # 1. Menerima payload JSON mentah
            raw_data = await websocket.receive_json()
            
            # 2. Sentuh sesi agar tidak dibersihkan oleh Garbage Collector (TTL)
            await session_manager.touch_session(session_id)
            
            try:
                # 3. Validasi bentuk data menggunakan skema Pydantic yang sudah ada
                validated_data = PredictRequest(session_id=session_id, **raw_data)
                landmarks_dict = validated_data.landmarks.model_dump()
                
                # 4. Offload pemrosesan ML (CPU-bound) ke thread terpisah!
                # SANGAT PENTING: Mencegah pemblokiran loop async untuk pengguna lain yang serentak.
                result = await asyncio.to_thread(
                    evaluator.process_frame, 
                    validated_data.timestamp, 
                    landmarks_dict
                )
                
                # 5. Smart Broadcasting: Hanya kirim data kembali jika sesuatu yang penting terjadi
                # (misal, Repetisi selesai, atau status pengguna berubah dari 'idle' -> 'moving_up')
                current_state = result.get("state")
                status = result.get("status")
                
                if status == "success" or current_state != last_state:
                    await websocket.send_json(result)
                    last_state = current_state
                    
            except ValidationError as ve:
                await websocket.send_json({"error": "Struktur payload tidak valid", "details": ve.errors()})
                
    except WebSocketDisconnect:
        print(f"⚠️ Klien terputus dari sesi {session_id}")
        # Catatan: Kita TIDAK langsung menghapus sesi. 
        # Klien mungkin kehilangan sinyal dan akan menyambung kembali.
        # Task TTL cleanup milik SessionManager akan menanganinya jika mereka tidak kembali.

@app.post("/session/end", response_model=SessionEndResponse)
async def end_session(request: SessionEndRequest):
    """REST: Menutup sesi secara paksa dan mengkompilasi statistik."""
    # Asumsikan delete_session asinkron
    result = await session_manager.delete_session(request.session_id)
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail="Sesi tidak ditemukan")
    return SessionEndResponse(**result)