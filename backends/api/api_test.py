import requests
import time

BASE_URL = "http://localhost:8000"

def run_test():
    print("--- 1. Memulai Sesi ---")
    start_payload = {"user_id": "pail", "exercise_type": "biceps"}
    response = requests.post(f"{BASE_URL}/session/start", json=start_payload)
    
    if response.status_code != 200:
        print("Gagal memulai sesi:", response.text)
        return
        
    session_data = response.json()
    session_id = session_data["session_id"]
    print(f"✅ Sesi Berhasil Dibuat. ID: {session_id}\n")
    
    print("--- 2. Simulasi Mengirim Frame (Predict) ---")
    
    def send_frame(y_wrist, deskripsi):
        mock_payload = {
            "session_id": session_id,
            "timestamp": time.time(),
            "landmarks": {
                "shoulder": {"x": 0.5, "y": 0.2, "visibility": 0.99},
                "elbow":    {"x": 0.5, "y": 0.5, "visibility": 0.99},
                "wrist":    {"x": 0.5, "y": y_wrist, "visibility": 0.99}, 
                "hip":      {"x": 0.5, "y": 0.7, "visibility": 0.99}
            }
        }
        res = requests.post(f"{BASE_URL}/predict", json=mock_payload)
        data = res.json()
        
        if res.status_code != 200:
            print(f"{deskripsi:<20} | ❌ ERROR {res.status_code}: {data.get('detail', data)}")
            time.sleep(0.05)
            return
            
        print(f"{deskripsi:<20} | Status: {data['status']:<10} | State Mesin: {data.get('state', '')}")
        if data['status'] == 'success':
            print(f"\n  🔥 REP SELESAI! Total: {data['rep_count']}")
            print(f"  🤖 Prediksi ML: {data['prediction']} (Confident: {data.get('confidence', 0):.2f})\n")
        time.sleep(0.05) # Jeda simulasi kamera

    # Fase 1: Posisi Awal (Tangan Lurus Bawah)
    for _ in range(5): send_frame(0.8, "Fase: Idle")
    
    # Fase 2: Mulai Angkat Beban (Curl)
    for _ in range(5): send_frame(0.6, "Fase: Angkat")
    
    # Fase 3: Puncak Curl (Tangan Terlipat penuh)
    for _ in range(5): send_frame(0.3, "Fase: Puncak")
    
    # Fase 4: Turunkan Beban
    for _ in range(5): send_frame(0.6, "Fase: Turun")
    
    # Fase 5: Kembali Lurus ke Bawah (Akan men-trigger 'completed')
    for _ in range(5): send_frame(0.8, "Fase: Selesai Rep")
        
    print("\n--- 3. Mengakhiri Sesi ---")
    end_payload = {"session_id": session_id}
    end_res = requests.post(f"{BASE_URL}/session/end", json=end_payload)
    print(f"✅ Sesi Ditutup. Hasil: {end_res.json()}")

if __name__ == "__main__":
    run_test()