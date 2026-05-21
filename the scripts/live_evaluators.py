import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import time
import pygame
import threading

# inisialisasi audio
pygame.mixer.init()
def play_beep(is_correct):
    try:
        pygame.mixer.music.stop()
        # nada tinggi (benar) atau nada rendah (salah)
        frequency = 880 if is_correct else 220 
        sample_rate, duration = 44100, 0.3
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = np.sin(frequency * 2 * np.pi * t) * 32767
        stereo_wave = np.column_stack((wave, wave)).astype(np.int16)
        pygame.sndarray.make_sound(stereo_wave).play()
    except: pass

# 1. LOAD MODEL XGBOOST
MODEL_PATH = 'models/biceps_xgboost_model.pkl'
try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    print("INFO: Model AI berhasil dimuat! Menyalakan webcam...")
except FileNotFoundError:
    print(f"ERR: Model {MODEL_PATH} tidak ditemukan.")
    exit()

# inisialisasi mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
draw_spec_landmark = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
draw_spec_connection = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)

# fungsi kalkulasi sudut
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle

def angle_with_y_axis(p_top, p_bottom):
    dx, dy = p_bottom[0] - p_top[0], p_bottom[1] - p_top[1]
    return np.degrees(np.arctan2(abs(dx), abs(dy)))

# variabel sistem live
counter = 0
stage = "down"
active_arm = None
rep_start_time = 0

# variabel untuk menampilkan hasil di layar
last_prediction = "Waiting..."
prediction_color = (255, 255, 255) # putih default

# buffer penampung data real-time (The Core Four)
rep_data = {"elbow_angles": [], "shoulder_angles": [], "upper_arm_angles": [], "torso_angles": []}

# buka webcam
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # ekstrak koordinat (L & R)
            r_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            r_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            r_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            r_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

            l_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            l_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            l_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            l_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

            # 2. LOGIKA DETEKSI LENGAN & START REP
            if stage == "down":
                if l_elbow_angle < 130:
                    active_arm, stage = "left", "up"
                    last_prediction = "Recording Left..."
                    prediction_color = (0, 255, 255)
                elif r_elbow_angle < 130:
                    active_arm, stage = "right", "up"
                    last_prediction = "Recording Right..."
                    prediction_color = (0, 255, 255)

            # 3. MENGUMPULKAN FITUR SAAT BERGERAK NAIK
            if stage == "up":
                if active_arm == "left":
                    c_shoulder, c_elbow, c_wrist, c_hip = l_shoulder, l_elbow, l_wrist, l_hip
                    c_elbow_angle = l_elbow_angle
                else:
                    c_shoulder, c_elbow, c_wrist, c_hip = r_shoulder, r_elbow, r_wrist, r_hip
                    c_elbow_angle = r_elbow_angle

                # catat sudut ke buffer
                rep_data["elbow_angles"].append(c_elbow_angle)
                rep_data["shoulder_angles"].append(calculate_angle(c_hip, c_shoulder, c_elbow))
                rep_data["upper_arm_angles"].append(angle_with_y_axis(c_shoulder, c_elbow))
                rep_data["torso_angles"].append(angle_with_y_axis(c_shoulder, c_hip))

                # 4. REP SELESAI -> LAKUKAN INFERENSI (PREDIKSI) AI
                if c_elbow_angle > 140:
                    stage = "down"
                    
                    rom_elbow = max(rep_data["elbow_angles"]) - min(rep_data["elbow_angles"])
                    
                    # gatekeeper sederhana
                    if rom_elbow < 60:
                        last_prediction = "DROP: ROM tidak penuh!"
                        prediction_color = (0, 0, 255)
                    else:
                        counter += 1
                        
                        # hitung fitur core four
                        feat_rom = rom_elbow
                        feat_upper_std = np.std(rep_data["upper_arm_angles"])
                        feat_torso_sway = max(rep_data["torso_angles"]) - min(rep_data["torso_angles"])
                        feat_shoulder_range = max(rep_data["shoulder_angles"]) - min(rep_data["shoulder_angles"])

                        # susun sebagai DataFrame agar formatnya sama dengan saat training
                        X_live = pd.DataFrame([{
                            'rom_elbow': feat_rom,
                            'upper_arm_angle_std': feat_upper_std,
                            'torso_sway_range': feat_torso_sway,
                            'shoulder_angle_range': feat_shoulder_range
                        }])

                        # prediksi menggunakan model
                        # DEBUGGING: Cetak fitur ke terminal agar kita tahu apa yang dilihat AI
                        print(f"\n--- REP {counter} SELESAI ---")
                        print(f"ROM Siku     : {feat_rom:.2f}")
                        print(f"Upper Arm Std: {feat_upper_std:.2f} (Kunci ayunan siku)")
                        print(f"Shoulder Rng : {feat_shoulder_range:.2f} (Kunci ayunan bahu)")
                        
                        # prediksi menggunakan model
                        prediction = model.predict(X_live)[0] # 0=correct, 1=incorrect
                        print(f"HASIL AI     : {'CORRECT' if prediction == 0 else 'INCORRECT'}")

                        if prediction == 0:
                            last_prediction = "GOOD FORM! (Correct)"
                            prediction_color = (0, 255, 0) # hijau
                            threading.Thread(target=play_beep, args=(True,), daemon=True).start()
                        else:
                            last_prediction = "BAD FORM! (Incorrect)"
                            prediction_color = (0, 0, 255) # merah
                            threading.Thread(target=play_beep, args=(False,), daemon=True).start()

                    # bersihkan buffer untuk rep selanjutnya
                    for key in rep_data: rep_data[key].clear()

            # visualisasi skeleton
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, draw_spec_landmark, draw_spec_connection)

        # UI Overlay (Menampilkan hasil Evaluasi AI di layar)
        cv2.rectangle(frame, (0, 0), (width, 80), (0, 0, 0), -1) # background bar atas
        cv2.putText(frame, f"REPS: {counter} | ARM: {active_arm}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # teks prediksi akan berubah warna sesuai hasil (hijau/merah)
        cv2.putText(frame, f"AI EVAL: {last_prediction}", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, prediction_color, 2)
        cv2.putText(frame, "Press 'Q' to Exit", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Biceps Real-Time AI Evaluator", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()