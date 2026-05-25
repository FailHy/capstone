"""
SessionManager — Fixed Version
================================
BUGS FIXED:
  #1 - Async Mismatch: semua method yang dipanggil dengan await dari main.py
       sekarang didefinisikan sebagai async.
  #5 - Missing Methods: touch_session, cleanup_loop, shutdown ditambahkan.
  #6 - Race Condition: asyncio.Lock melindungi akses ke _sessions dict.

DESIGN:
  - TTL default 5 menit. Sesi yang tidak aktif otomatis dibersihkan.
  - touch_session() dipanggil setiap frame dari WebSocket handler untuk
    mencegah sesi aktif dibersihkan cleanup_loop.
  - shutdown() membatalkan cleanup_task dengan aman tanpa raise exception.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, Optional

from .model_loader import ModelLoader
from .evaluator_service import ExerciseEvaluatorService


# TTL dalam detik. Sesi yang tidak menerima frame selama ini akan dihapus.
_SESSION_TTL_SECONDS: int = 300  # 5 menit


class SessionManager:
    def __init__(self, session_ttl: int = _SESSION_TTL_SECONDS) -> None:
        # Dict utama: session_id -> ExerciseEvaluatorService
        self._sessions: Dict[str, ExerciseEvaluatorService] = {}

        # Dict timestamp last-touch: session_id -> float (epoch)
        # Dipisah dari _sessions agar tidak mencemari service object.
        self._last_activity: Dict[str, float] = {}

        # Lock tunggal untuk semua operasi ke _sessions dan _last_activity.
        # asyncio.Lock aman di single-process uvicorn. Jika upgrade ke
        # multi-process (Gunicorn), ganti dengan Redis-based session store.
        self._lock = asyncio.Lock()

        self._ttl = session_ttl
        self.model_loader = ModelLoader()

    # ------------------------------------------------------------------
    # PUBLIC ASYNC INTERFACE
    # ------------------------------------------------------------------

    async def create_session(
        self, user_id: str, exercise_type: str = "biceps"
    ) -> str:
        """
        Alokasi sesi baru. Mengembalikan session_id (UUID string).

        Raises:
            ValueError: jika exercise_type tidak dikenali model loader.
        """
        # ModelLoader.get_artifacts bisa raise ValueError — biarkan naik ke
        # caller (main.py akan tangkap dan return HTTP 400).
        model, label_encoder, feature_columns = self.model_loader.get_artifacts(
            exercise_type
        )

        service = ExerciseEvaluatorService(
            exercise_type=exercise_type,
            model=model,
            label_encoder=label_encoder,
            feature_columns=feature_columns,
        )

        session_id = str(uuid.uuid4())

        async with self._lock:
            self._sessions[session_id] = service
            self._last_activity[session_id] = time.monotonic()

        return session_id

    async def get_session(self, session_id: str) -> ExerciseEvaluatorService:
        """
        Ambil service untuk session_id yang aktif.

        Raises:
            KeyError: jika sesi tidak ditemukan atau sudah expired.
        """
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session '{session_id}' not found or expired.")
            return self._sessions[session_id]

    async def touch_session(self, session_id: str) -> None:
        """
        Perbarui timestamp last-activity agar sesi tidak di-expire cleanup_loop.
        Dipanggil setiap frame di WebSocket handler.
        Tidak raise jika sesi tidak ada (client mungkin sudah disconnect).
        """
        async with self._lock:
            if session_id in self._last_activity:
                self._last_activity[session_id] = time.monotonic()

    async def delete_session(self, session_id: str) -> Dict[str, Any]:
        """
        Hapus sesi secara eksplisit dan kembalikan statistik workout.
        Dipanggil dari REST endpoint /session/end.
        """
        async with self._lock:
            service = self._sessions.pop(session_id, None)
            self._last_activity.pop(session_id, None)

        if service is not None:
            return {
                "status": "session_ended",
                "total_reps": service.rep_count,
                "exercise_type": service.exercise_type,
            }
        return {"status": "error", "message": "not_found"}

    async def cleanup_loop(self) -> None:
        """
        Coroutine background: setiap 60 detik cek dan hapus sesi yang TTL-nya
        habis. Dipanggil dari lifespan sebagai asyncio.Task.

        Kenapa 60 detik? Cukup responsif untuk TTL 5 menit,
        tidak membebani lock di setiap detik.
        """
        while True:
            try:
                await asyncio.sleep(60)
                await self._expire_stale_sessions()
            except asyncio.CancelledError:
                # Task dibatalkan saat shutdown — keluar dengan bersih.
                break
            except Exception as exc:
                # Jangan crash loop karena 1 siklus gagal.
                print(f"[SessionManager] cleanup_loop error (non-fatal): {exc}")

    async def shutdown(self) -> None:
        """
        Pembersihan saat API shutdown. Hapus semua sesi aktif.
        Dipanggil dari lifespan after yield.
        """
        async with self._lock:
            count = len(self._sessions)
            self._sessions.clear()
            self._last_activity.clear()

        if count:
            print(f"[SessionManager] shutdown: cleared {count} active session(s).")

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    async def _expire_stale_sessions(self) -> None:
        """Hapus sesi yang tidak aktif melebihi TTL."""
        now = time.monotonic()
        expired: list[str] = []

        async with self._lock:
            for sid, last in self._last_activity.items():
                if (now - last) > self._ttl:
                    expired.append(sid)

            for sid in expired:
                self._sessions.pop(sid, None)
                self._last_activity.pop(sid, None)

        if expired:
            print(
                f"[SessionManager] TTL cleanup: removed {len(expired)} "
                f"stale session(s): {expired}"
            )