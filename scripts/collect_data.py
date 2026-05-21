from __future__ import annotations

import argparse
import csv
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import mediapipe as mp
import numpy as np

try:
    import pygame
except Exception:  # Bergantung pada audio backend lokal
    pygame = None

# Mengimpor variabel konfigurasi dan utilitas dari file proyek lainnya
from config import (
    DATASET_COLUMNS,
    ELBOW_START_ANGLE,
    RAW_DATASET_PATH,
    REJECTED_SAMPLE_LOG,
    VISIBILITY_THRESHOLD,
)
from feature_utils import RepBuffer, RepetitionSegmenter, extract_repetition_features, validate_repetition_quality
from pose_utils import arm_visibility_ok, compute_frame_angles, get_arm_landmarks

# -----------------------------------------------------------------------------
# MAPPING LABEL (Sesuai Validasi Pakar / SME)
# Format: Tombol: ("Label Biner", "Error Type", "Notes")
# -----------------------------------------------------------------------------
LABEL_KEYS = {
    ord("c"): ("correct", "correct", "good_form"),
    ord("1"): ("incorrect", "body_swing", "badan_bergerak"),
    ord("2"): ("incorrect", "elbow_swing", "siku_maju_mundur"),
    ord("3"): ("incorrect", "not_full_up", "rom_tidak_penuh"),
    ord("4"): ("incorrect", "too_fast", "tempo_terlalu_cepat"),
    ord("i"): ("incorrect", "unknown_incorrect", "bad_form_unspecified"),
}

# -----------------------------------------------------------------------------
# SETUP UI TERMINAL (Interactive Setup)
# -----------------------------------------------------------------------------
def get_user_metadata() -> dict:
    """Meminta input metadata dari user via terminal sebelum kamera terbuka."""
    print("="*45)
    print("  BICEPS DATA COLLECTOR - PENGATURAN SESI")
    print("="*45)
    
    # Pilih Subject ID (S01 - S10)
    while True:
        try:
            sub_id = int(input("Pilih Subject ID (1-10) : "))
            if 1 <= sub_id <= 10:
                subject_str = f"S{sub_id:02d}" # Format S01, S02, dst.
                break
            print("Harap masukkan angka 1 hingga 10.")
        except ValueError:
            print("Input tidak valid! Masukkan angka.")

    # Pilih Session ID (1 - 5)
    while True:
        try:
            ses_id = int(input("Pilih Session ID (1-5)  : "))
            if 1 <= ses_id <= 5:
                session_str = f"session_{ses_id:02d}" # Format session_01, dst.
                break
            print("Harap masukkan angka 1 hingga 5.")
        except ValueError:
            print("Input tidak valid! Masukkan angka.")

    # Pengaturan Default yang aman
    camera_pos = "side"
    lighting = "normal"
    exercise = "biceps"

    print("\n--- KONFIRMASI SESI ---")
    print(f"Subject : {subject_str}")
    print(f"Session : {session_str}")
    print("Menyiapkan kamera...\n")
    
    return {
        "subject_id": subject_str,
        "session_id": session_str,
        "camera_position": camera_pos,
        "lighting_condition": lighting,
        "exercise_type": exercise,
    }


# -----------------------------------------------------------------------------
# FUNGSI UTILITAS SISTEM
# -----------------------------------------------------------------------------
def init_audio() -> None:
    """Inisialisasi sistem audio untuk feedback suara."""
    if pygame is None:
        return
    try:
        pygame.mixer.init()
    except Exception:
        pass

def play_beep() -> None:
    """Memutar suara 'beep' pendek saat satu repetisi selesai dideteksi."""
    if pygame is None:
        return
    try:
        pygame.mixer.music.stop()
        sample_rate = 44100
        duration = 0.25
        frequency = 660 # Nada menengah yang nyaman
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = (np.sin(frequency * 2 * np.pi * t) * 32767).astype(np.int16)
        stereo_wave = np.column_stack((wave, wave))
        pygame.sndarray.make_sound(stereo_wave).play()
    except Exception:
        pass

def ensure_csv(path: Path) -> None:
    """Memastikan file CSV tujuan sudah ada dan memiliki header yang benar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DATASET_COLUMNS)
            writer.writeheader()

def next_sample_id(path: Path) -> int:
    """Mencari ID sampel terakhir di dataset agar tidak menimpa data lama."""
    if not path.exists(): 
        return 1
    try:
        import pandas as pd
        df = pd.read_csv(path)
        if "sample_id" not in df.columns or df.empty: 
            return 1
        return int(df["sample_id"].max()) + 1
    except Exception: 
        return 1

def append_row(path: Path, row: Dict[str, object]) -> None:
    """Menambahkan baris data fitur baru ke dalam dataset CSV."""
    ensure_csv(path)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_COLUMNS, extrasaction="ignore")
        writer.writerow(row)

def log_rejection(reason: str, row_data: Optional[dict] = None) -> None:
    """Mencatat repetisi yang dibuang atau dilewati (skip) ke file log khusus."""
    print(f"[REJECTED] {reason}")
    if row_data is None:
        return
    REJECTED_SAMPLE_LOG.parent.mkdir(parents=True, exist_ok=True)
    exists = REJECTED_SAMPLE_LOG.exists()
    with REJECTED_SAMPLE_LOG.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_data.keys()) + ["reject_reason"], extrasaction="ignore")
        if not exists:
            writer.writeheader()
        row_data["reject_reason"] = reason
        writer.writerow(row_data)

def choose_active_arm(left_angle: Optional[float], right_angle: Optional[float]) -> Optional[str]:
    """Menentukan lengan mana yang sedang mengangkat beban (kiri/kanan) berdasarkan sudut."""
    candidates = []
    if left_angle is not None and left_angle < ELBOW_START_ANGLE:
        candidates.append((left_angle, "left"))
    if right_angle is not None and right_angle < ELBOW_START_ANGLE:
        candidates.append((right_angle, "right"))
    if not candidates: 
        return None
    # Pilih lengan yang ditekuk paling tajam
    return min(candidates, key=lambda item: item[0])[1]

def put_text(frame, text: str, y: int, color=(255, 255, 255)) -> None:
    """Helper untuk menulis teks di atas frame video dengan cepat."""
    cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


# -----------------------------------------------------------------------------
# PROGRAM UTAMA (MAIN LOOP)
# -----------------------------------------------------------------------------
def main() -> None:
    # Set output dataset sesuai konfigurasi
    output_path = RAW_DATASET_PATH
    camera_index = 0
    is_mirrored = True

    # 1. Panggil form setup di terminal sebelum membuka OpenCV
    metadata_base = get_user_metadata()

    # 2. Setup sistem (Audio & File)
    init_audio()
    ensure_csv(output_path)
    sample_id = next_sample_id(output_path)

    # 3. Setup MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    draw_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)

    # 4. Buka Kamera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Tidak bisa membuka kamera (Index {camera_index}).")
        return

    # Inisialisasi variabel state/buffer
    segmenter = RepetitionSegmenter()
    buffer = RepBuffer()
    active_arm = None
    waiting_label = False
    pending_row = None
    counter = 0
    
    # Variabel Interaktif
    is_recording = False 
    status = "TEKAN SPASI UNTUK MULAI MEREKAM LATIHAN"
    prev_time = time.time()
    help_text = "Key: c=Correct, 1=BodySwing, 2=ElbowSwing, 3=NotFullUp, 4=TooFast, SPACE=Start"

    # Jalankan engine MediaPipe Pose
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: 
                break

            if is_mirrored: 
                frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            
            # Hitung FPS
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # Proses Pose
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            left_angle = right_angle = None

            # Jika tubuh terdeteksi dan tidak sedang menunggu label dari user
            if results.pose_landmarks and not waiting_label:
                landmarks = results.pose_landmarks.landmark
                try:
                    left_lm = get_arm_landmarks(landmarks, mp_pose, "left", is_mirrored)
                    right_lm = get_arm_landmarks(landmarks, mp_pose, "right", is_mirrored)

                    if arm_visibility_ok(left_lm): 
                        left_angle = compute_frame_angles(left_lm)["elbow_angle"]
                    if arm_visibility_ok(right_lm): 
                        right_angle = compute_frame_angles(right_lm)["elbow_angle"]

                    # -------------------------------------------------------------
                    # LOGIKA EKSTRAKSI (HANYA BERJALAN JIKA USER SUDAH TEKAN SPASI)
                    # -------------------------------------------------------------
                    if is_recording:
                        # Tentukan lengan mana yang bekerja saat pertama kali bergerak
                        if active_arm is None:
                            selected = choose_active_arm(left_angle, right_angle)
                            if selected is not None:
                                active_arm = selected
                                buffer.clear()
                                segmenter.reset(keep_down=True)
                                status = f"Merekam Lengan {active_arm.upper()}..."

                        # Evaluasi per frame dari lengan yang bekerja
                        if active_arm is not None:
                            arm_lm = get_arm_landmarks(landmarks, mp_pose, active_arm, is_mirrored)
                            if arm_visibility_ok(arm_lm, VISIBILITY_THRESHOLD):
                                frame_features = compute_frame_angles(arm_lm)
                                event = segmenter.update(frame_features["elbow_angle"], now)

                                # Masukkan sudut ke buffer selama fase pergerakan
                                if event in {"started", "moving_up", "top", "up", "moving_down", "completed", "completed_partial", "down"}:
                                    buffer.append(now, frame_features)

                                # Jika 1 repetisi penuh selesai
                                if event in {"completed", "completed_partial"}:
                                    features = extract_repetition_features(buffer)
                                    is_valid, reason = validate_repetition_quality(buffer, features)
                                    
                                    # Jika data rep tidak absurd, siapkan untuk diberi label
                                    if is_valid:
                                        pending_row = {
                                            "sample_id": sample_id, 
                                            "timestamp": datetime.utcnow().isoformat(timespec="seconds"), 
                                            **metadata_base, 
                                            "active_arm": active_arm, 
                                            **features
                                        }
                                        waiting_label = True
                                        counter += 1
                                        status = "Repetisi Selesai! Beri Label Sekarang (c/1/2/3/4)"
                                        threading.Thread(target=play_beep, daemon=True).start()
                                    else:
                                        status = f"Repetisi Ditolak: {reason}"
                                    
                                    # Bersihkan memori untuk repetisi berikutnya
                                    buffer.clear()
                                    segmenter.reset(keep_down=True)
                                    active_arm = None
                except Exception as exc:
                    status = f"Pose error: {exc}"

                # Gambar rangka/skeleton di layar
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, draw_spec, draw_spec)

            # -------------------------------------------------------------
            # LOGIKA INPUT KEYBOARD OLEH USER
            # -------------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            
            # Tekan Spasi untuk mulai merekam
            if key == ord(' ') and not is_recording and not waiting_label:
                is_recording = True
                status = "Mulai! Silakan lakukan repetisi."
                
            # Saat repetisi selesai dan menunggu tombol label ditekan
            elif waiting_label and pending_row is not None:
                if key in LABEL_KEYS:
                    label, error_type, notes = LABEL_KEYS[key]
                    pending_row.update({"label": label, "error_type": error_type, "notes": notes})
                    append_row(output_path, pending_row)
                    
                    print(f"TERSIPAN: ID={sample_id} | Label={label.upper()} | Error={error_type}")
                    
                    sample_id += 1
                    pending_row = None
                    waiting_label = False
                    status = "Tersimpan. Silakan lakukan repetisi berikutnya."
                
                # Tekan 's' untuk skip / membuang data (jika terdeteksi salah secara sistematis)
                elif key == ord("s"):
                    log_rejection("manual_skip", pending_row)
                    pending_row = None
                    waiting_label = False
                    status = "Dilewati (Skipped). Silakan lakukan repetisi berikutnya."

            # Tekan 'q' untuk keluar dari aplikasi
            if key == ord("q"): 
                break

            # -------------------------------------------------------------
            # UI OVERLAY (TAMPILAN TEKS DI ATAS KAMERA)
            # -------------------------------------------------------------
            cv2.rectangle(frame, (0, 0), (width, 130), (0, 0, 0), -1)
            
            # Beri warna highlight (kuning jika idle, hijau jika merekam, putih default)
            status_color = (0, 255, 255) if not is_recording else (0, 255, 0) if "Selesai" in status or "Tersimpan" in status else (255, 255, 255)
            
            # Baris 1: Info Sistem
            put_text(frame, f"Sesi: {metadata_base['subject_id']} | ID Berikutnya: {sample_id} | FPS: {fps:.1f}", 25)
            # Baris 2: Status Repetisi
            put_text(frame, f"State: {segmenter.state} | Arm: {active_arm}", 55)
            # Baris 3: Status Interaktif (Sangat Jelas)
            put_text(frame, f"Status: {status}", 85, status_color)
            # Baris 4: Instruksi Label (Disederhanakan)
            put_text(frame, help_text, 115, (255, 200, 0))

            # Tampilkan sudut siku secara live di bagian bawah
            if left_angle is not None:
                put_text(frame, f"Left Elbow: {left_angle:.1f}", height - 40)
            if right_angle is not None:
                put_text(frame, f"Right Elbow: {right_angle:.1f}", height - 15)

            cv2.imshow("Gerakan Collector - Berbasis Ahli", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()