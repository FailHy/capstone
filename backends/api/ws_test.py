import asyncio
import websockets
import requests
import time
import json

BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"

async def test_websocket_flow():
    print("--- 1. Memulai Sesi (REST) ---")
    start_payload = {"user_id": "tester_ws_001", "exercise_type": "biceps"}
    response = requests.post(f"{BASE_URL}/session/start", json=start_payload)
    
    if response.status_code != 200:
        print("Gagal memulai sesi:", response.text)
        return
        
    session_id = response.json()["session_id"]
    print(f"✅ Sesi Berhasil Dibuat. ID: {session_id}\n")
    
    print("--- 2. Streaming via WebSocket ---")
    ws_endpoint = f"{WS_URL}/ws/{session_id}"
    
    try:
        # Membuka koneksi WebSocket
        async with websockets.connect(ws_endpoint) as ws:
            print("🔗 WebSocket Terhubung!\n")
            
            def build_frame(y_wrist):
                return {
                    "timestamp": time.time(),
                    "landmarks": {
                        "shoulder": {"x": 0.5, "y": 0.2, "visibility": 0.99},
                        "elbow":    {"x": 0.5, "y": 0.5, "visibility": 0.99},
                        "wrist":    {"x": 0.5, "y": y_wrist, "visibility": 0.99}, 
                        "hip":      {"x": 0.5, "y": 0.7, "visibility": 0.99}
                    }
                }

            async def send_and_listen(y_wrist, deskripsi):
                # Kirim data ke server
                frame = build_frame(y_wrist)
                await ws.send(json.dumps(frame))
                
                # Coba dengarkan balasan dari server (dengan timeout kecil agar tidak macet)
                try:
                    # WebSocket backend kita sekarang HANYA membalas jika ada perubahan state / rep selesai!
                    response_str = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    data = json.loads(response_str)
                    print(f"[{deskripsi:<15}] 📩 SERVER MEMBALAS: State -> {data.get('state')}")
                    if data.get('status') == 'success':
                        print(f"  🔥 REP SELESAI! Total: {data['rep_count']} | ML: {data['prediction']} (Conf: {data['confidence']:.2f})\n")
                except asyncio.TimeoutError:
                    # Server diam (artinya frame diterima tapi tidak ada perubahan state penting)
                    print(f"[{deskripsi:<15}] 📤 Frame terkirim (Server diam - hemat bandwidth)")
                
                await asyncio.sleep(0.05) # Jeda simulasi kamera

            # Simulasi Gerakan Curl Biceps Lengkap
            for _ in range(5): await send_and_listen(0.8, "Fase: Idle")
            for _ in range(5): await send_and_listen(0.6, "Fase: Angkat")
            for _ in range(5): await send_and_listen(0.3, "Fase: Puncak")
            for _ in range(5): await send_and_listen(0.6, "Fase: Turun")
            for _ in range(5): await send_and_listen(0.8, "Fase: Selesai")
            
    except Exception as e:
        print(f"❌ Error WebSocket: {e}")

    print("\n--- 3. Mengakhiri Sesi (REST) ---")
    end_payload = {"session_id": session_id}
    end_res = requests.post(f"{BASE_URL}/session/end", json=end_payload)
    print(f"✅ Sesi Ditutup. Hasil: {end_res.json()}")

if __name__ == "__main__":
    # Jalankan event loop asyncio
    asyncio.run(test_websocket_flow())