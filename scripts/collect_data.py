from __future__ import annotations

import csv
import logging
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import mediapipe as mp
import numpy as np

try:
    import pygame
except Exception:
    pygame = None

from config import (
    DATASET_COLUMNS,
    EXERCISE_CONFIG,
    LOG_DIR,
    RAW_DATASET_PATH,
    REJECTED_SAMPLE_LOG,
    VISIBILITY_THRESHOLD,
)
from feature_utils import RepBuffer, RepetitionSegmenter, extract_repetition_features, validate_repetition_quality
from pose_utils import arm_visibility_ok, compute_frame_angles, get_arm_landmarks

LOG_DIR.mkdir(parents=True, exist_ok=True)
ERROR_LOG_PATH = LOG_DIR / "collection_error.log"

logging.basicConfig(
    filename=str(ERROR_LOG_PATH),
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

LABEL_KEYS = {
    ord("c"): ("correct",   "correct",           "good_form"),
    ord("1"): ("incorrect", "body_swing",         "badan_bergerak"),
    ord("2"): ("incorrect", "elbow_swing",        "siku_maju_mundur"),
    ord("3"): ("incorrect", "not_full_up",        "rom_tidak_penuh"),
    ord("4"): ("incorrect", "too_fast",           "tempo_terlalu_cepat"),
    ord("i"): ("incorrect", "unknown_incorrect",  "bad_form_unspecified"),
}


# ---------------------------------------------------------------------------
# Helper: get_user_metadata
# ---------------------------------------------------------------------------
def get_user_metadata() -> dict:
    print("="*45)
    print("  AI DATA COLLECTOR - PENGATURAN SESI")
    print("="*45)

    while True:
        try:
            print("\nPilih Latihan:")
            print("1. Biceps (Curl)")
            print("2. Triceps (Pushdown / Extension)")
            ex_choice = int(input("Masukkan angka (1/2): "))
            if ex_choice == 1:
                exercise_str = "biceps"; break
            elif ex_choice == 2:
                exercise_str = "triceps"; break
            print("Pilihan tidak valid! Masukkan angka 1 atau 2.")
        except ValueError:
            print("Input tidak valid! Masukkan angka.")

    while True:
        try:
            sub_id = int(input("\nPilih Subject ID (1-20) : "))
            if 1 <= sub_id <= 20:
                subject_str = f"S{sub_id:02d}"; break
            print("Harap masukkan angka 1 hingga 20.")
        except ValueError:
            print("Input tidak valid! Masukkan angka.")

    while True:
        try:
            ses_id = int(input("Pilih Session ID (1-10) : "))
            if 1 <= ses_id <= 10:
                session_str = f"session_{ses_id:02d}"; break
            print("Harap masukkan angka 1 hingga 10.")
        except ValueError:
            print("Input tidak valid! Masukkan angka.")

    print("\n" + "="*45)
    print("--- KONFIRMASI SESI ---")
    print(f"Latihan : {exercise_str.upper()}")
    print(f"Subject : {subject_str}")
    print(f"Session : {session_str}")
    print("Menyiapkan kamera...\n")
    print("="*45)

    return {
        "subject_id":        subject_str,
        "session_id":        session_str,
        "camera_position":   "side",
        "lighting_condition":"normal",
        "exercise_type":     exercise_str,
    }


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def init_audio() -> None:
    if pygame is None: return
    try: pygame.mixer.init()
    except Exception as e: logging.error(f"Gagal inisialisasi audio: {e}")

def play_beep() -> None:
    if pygame is None: return
    try:
        pygame.mixer.music.stop()
        sample_rate = 44100
        t    = np.linspace(0, 0.25, int(sample_rate * 0.25), False)
        wave = (np.sin(660 * 2 * np.pi * t) * 32767).astype(np.int16)
        pygame.sndarray.make_sound(np.column_stack((wave, wave))).play()
    except Exception as e: logging.error(f"Gagal memutar beep: {e}")


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def ensure_csv(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=DATASET_COLUMNS).writeheader()
    except Exception as e:
        logging.error(f"Gagal membuat CSV {path}: {e}")

def next_sample_id(path: Path) -> int:
    if not path.exists(): return 1
    try:
        import pandas as pd
        df = pd.read_csv(path)
        if "sample_id" not in df.columns or df.empty: return 1
        return int(df["sample_id"].max()) + 1
    except Exception as e:
        logging.error(f"Gagal membaca next_sample_id: {e}")
        return 1

def append_row(path: Path, row: Dict[str, object]) -> None:
    try:
        ensure_csv(path)
        with path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=DATASET_COLUMNS, extrasaction="ignore").writerow(row)
    except Exception as e:
        logging.error(f"Gagal menyimpan baris: {e}")

def log_rejection(reason: str, row_data: Optional[dict] = None) -> None:
    print(f"[REJECTED] {reason}")
    if row_data is None: return
    try:
        REJECTED_SAMPLE_LOG.parent.mkdir(parents=True, exist_ok=True)
        exists = REJECTED_SAMPLE_LOG.exists()
        with REJECTED_SAMPLE_LOG.open("a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(row_data.keys()) + ["reject_reason"],
                extrasaction="ignore"
            )
            if not exists: writer.writeheader()
            row_data["reject_reason"] = reason
            writer.writerow(row_data)
    except Exception as e:
        logging.error(f"Gagal mencatat log rejection: {e}")


# ---------------------------------------------------------------------------
# choose_active_arm
# ---------------------------------------------------------------------------
def choose_active_arm(
    left_angle: Optional[float],
    right_angle: Optional[float],
) -> Optional[str]:
    """
    Pilih lengan yang lebih tertekuk (sudut lebih kecil) sebagai lengan aktif.
    Ini lebih andal daripada memilih lengan paling 'berbeda' secara absolut,
    karena lengan yang sedang curl akan selalu memiliki sudut lebih kecil
    dari posisi netral.
    """
    try:
        candidates = []
        if left_angle  is not None: candidates.append((left_angle,  "left"))
        if right_angle is not None: candidates.append((right_angle, "right"))
        if not candidates: return None
        return min(candidates, key=lambda item: item[0])[1]
    except Exception as e:
        logging.error(f"Error di choose_active_arm: {e}")
        return None


# ---------------------------------------------------------------------------
# HUD helper
# ---------------------------------------------------------------------------
def put_text(frame, text: str, y: int, color=(255, 255, 255)) -> None:
    cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    try:
        metadata_base = get_user_metadata()
        exercise_type = metadata_base["exercise_type"]

        init_audio()
        ensure_csv(RAW_DATASET_PATH)
        sample_id = next_sample_id(RAW_DATASET_PATH)

        mp_pose    = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        draw_spec  = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Tidak bisa membuka kamera utama.")
            return

        segmenter     = RepetitionSegmenter(exercise_type=exercise_type)
        buffer        = RepBuffer()
        active_arm    = None
        waiting_label = False
        pending_row   = None
        is_recording  = False

        # -----------------------------------------------------------------------
        # BUG FIX: ARM LOCK TIMER
        # MASALAH SEBELUMNYA: Setiap frame, jika active_arm masih None, sistem
        # memanggil choose_active_arm yang hanya melihat sudut frame itu saja.
        # Jika landmark tidak stabil di detik pertama, active_arm tetap None
        # dan buffer tidak pernah diisi meskipun user sudah bergerak.
        # SOLUSI: Kumpulkan sudut selama ARM_LOCK_WINDOW detik pertama setelah
        # spasi ditekan, lalu pilih lengan berdasarkan rata-rata sudut minimum
        # (lengan yang lebih banyak bergerak/tertekuk).
        # -----------------------------------------------------------------------
        ARM_LOCK_WINDOW   = 1.5    # detik untuk observasi pemilihan lengan
        arm_lock_start    = 0.0
        arm_angle_history: Dict[str, list] = {"left": [], "right": []}

        status    = f"MODE: {exercise_type.upper()} | TEKAN SPASI UNTUK MULAI"
        prev_time = time.time()
        help_text = "Key: c=Correct, 1=BodySwing, 2=ElbowSwing, 3=NotFullUp, 4=TooFast | s=Skip | q=Quit"

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                try:
                    ok, frame = cap.read()
                    if not ok: break

                    frame          = cv2.flip(frame, 1)
                    height, width  = frame.shape[:2]
                    now            = time.time()
                    fps            = 1.0 / max(now - prev_time, 1e-6)
                    prev_time      = now

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results   = pose.process(frame_rgb)
                    left_angle = right_angle = None

                    if results.pose_landmarks and not waiting_label:
                        landmarks = results.pose_landmarks.landmark
                        try:
                            left_lm  = get_arm_landmarks(landmarks, mp_pose, "left",  True)
                            right_lm = get_arm_landmarks(landmarks, mp_pose, "right", True)

                            if arm_visibility_ok(left_lm):
                                left_angle = compute_frame_angles(left_lm)["elbow_angle"]
                            if arm_visibility_ok(right_lm):
                                right_angle = compute_frame_angles(right_lm)["elbow_angle"]

                            if is_recording:
                                # -----------------------------------------------
                                # FASE 1: Pilih lengan aktif dengan window observasi
                                # -----------------------------------------------
                                if active_arm is None:
                                    elapsed_lock = now - arm_lock_start

                                    # Kumpulkan riwayat sudut per lengan
                                    if left_angle  is not None:
                                        arm_angle_history["left"].append(left_angle)
                                    if right_angle is not None:
                                        arm_angle_history["right"].append(right_angle)

                                    if elapsed_lock < ARM_LOCK_WINDOW:
                                        status = (
                                            f"Menentukan lengan aktif... "
                                            f"({elapsed_lock:.1f}/{ARM_LOCK_WINDOW:.1f}s)"
                                        )
                                    else:
                                        # Pilih lengan yang memiliki sudut rata-rata lebih kecil
                                        # (lengan yang lebih banyak bergerak / tertekuk)
                                        selected = choose_active_arm(
                                            float(np.mean(arm_angle_history["left"]))
                                            if arm_angle_history["left"] else None,
                                            float(np.mean(arm_angle_history["right"]))
                                            if arm_angle_history["right"] else None,
                                        )
                                        if selected:
                                            active_arm = selected
                                            buffer.clear()
                                            segmenter.reset()
                                            arm_angle_history = {"left": [], "right": []}
                                            status = f"Lengan TERKUNCI: {active_arm.upper()} | Mulai Bergerak!"
                                            print(f"[ARM LOCK] Active arm: {active_arm.upper()}")
                                        else:
                                            # Tidak ada landmark valid; reset timer dan coba lagi
                                            arm_lock_start = now
                                            arm_angle_history = {"left": [], "right": []}
                                            status = "Landmark tidak terdeteksi, pastikan tubuh terlihat kamera."

                                # -----------------------------------------------
                                # FASE 2: Proses gerakan pada lengan aktif
                                # -----------------------------------------------
                                if active_arm is not None:
                                    arm_lm = get_arm_landmarks(landmarks, mp_pose, active_arm, True)
                                    if arm_visibility_ok(arm_lm, VISIBILITY_THRESHOLD):
                                        frame_features = compute_frame_angles(arm_lm)
                                        event = segmenter.update(frame_features["elbow_angle"], now)

                                        if segmenter.state != "idle":
                                            buffer.append(now, frame_features)

                                        if event == "completed":
                                            features = extract_repetition_features(buffer)
                                            is_valid, reason = validate_repetition_quality(
                                                buffer, features, exercise_type
                                            )

                                            if is_valid:
                                                pending_row = {
                                                    "sample_id": sample_id,
                                                    "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                                                    **metadata_base,
                                                    "active_arm": active_arm,
                                                    **features,
                                                }
                                                waiting_label = True
                                                status = "Rep Selesai! Beri Label Sekarang"
                                                threading.Thread(target=play_beep, daemon=True).start()
                                            else:
                                                status = f"Ditolak: {reason}"
                                                print(f"[REJECT] {reason}")

                                            buffer.clear()
                                            segmenter.reset()
                                            active_arm = None

                        except Exception as exc:
                            logging.error(f"Pose error: {exc}")

                        mp_drawing.draw_landmarks(
                            frame, results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS, draw_spec, draw_spec
                        )

                    # -----------------------------------------------------------
                    # KEYBOARD HANDLING
                    # BUG FIX: Gunakan waitKey(10) bukan waitKey(1).
                    # waitKey(1) terkadang melewatkan event key pada beberapa
                    # platform karena loop terlalu cepat dan OS event queue
                    # belum sempat terisi. 10ms cukup untuk menangkap key press
                    # tanpa terasa lag secara visual.
                    # -----------------------------------------------------------
                    key = cv2.waitKey(10) & 0xFF

                    if key == ord(' ') and not is_recording and not waiting_label:
                        is_recording      = True
                        arm_lock_start    = now
                        arm_angle_history = {"left": [], "right": []}
                        active_arm        = None
                        buffer.clear()
                        segmenter.reset()
                        status = "Rekaman dimulai. Menentukan lengan aktif..."

                    elif waiting_label and pending_row is not None:
                        if key in LABEL_KEYS:
                            label, error_type, notes = LABEL_KEYS[key]
                            pending_row.update({"label": label, "error_type": error_type, "notes": notes})
                            append_row(RAW_DATASET_PATH, pending_row)

                            print(
                                f"[{exercise_type.upper()}] SAVED: "
                                f"ID={sample_id} | Label={label.upper()} | Error={error_type}"
                            )

                            sample_id    += 1
                            pending_row   = None
                            waiting_label = False
                            is_recording  = False
                            active_arm    = None
                            status = "Tersimpan! TEKAN SPASI lalu bergerak lagi."

                        elif key == ord("s"):
                            log_rejection("manual_skip", pending_row)
                            pending_row   = None
                            waiting_label = False
                            is_recording  = False
                            active_arm    = None
                            status = "Dilewati. TEKAN SPASI lalu bergerak lagi."

                    if key == ord("q"):
                        break

                    # -----------------------------------------------------------
                    # HUD Overlay
                    # -----------------------------------------------------------
                    cv2.rectangle(frame, (0, 0), (width, 150), (0, 0, 0), -1)

                    if waiting_label:
                        status_color = (0, 255, 255)
                    elif is_recording and active_arm:
                        status_color = (0, 255, 0)
                    elif is_recording:
                        status_color = (255, 165, 0)   # oranye = sedang kunci lengan
                    else:
                        status_color = (200, 200, 200)

                    put_text(
                        frame,
                        f"MODE: {exercise_type.upper()} | ID Berikutnya: {sample_id} | FPS: {fps:.1f}",
                        25, (100, 255, 255)
                    )
                    put_text(frame, f"State: {segmenter.state:<12} | Arm: {active_arm}", 55)
                    put_text(frame, f"Status: {status}", 85, status_color)
                    put_text(frame, help_text, 120, (255, 200, 0))

                    if left_angle  is not None: put_text(frame, f"Left:  {left_angle:.1f}°",  height - 40)
                    if right_angle is not None: put_text(frame, f"Right: {right_angle:.1f}°", height - 15)

                    cv2.imshow("Gerakan Collector - Multi Exercise", frame)

                except Exception as inner_e:
                    logging.error(f"Loop error: {inner_e}\n{traceback.format_exc()}")

        cap.release()
        cv2.destroyAllWindows()

    except Exception as main_e:
        logging.critical(f"FATAL ERROR: {main_e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()