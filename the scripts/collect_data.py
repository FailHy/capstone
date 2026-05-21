import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
import threading
import pygame

# inisialisasi pygame mixer untuk play audio
pygame.mixer.init()

# function untuk play sound beep saat 1 rep selesai
def play_beep():
    try:
        # stop music yang mungkin lagi play
        pygame.mixer.music.stop()
        
        # setup durasi dan frequency beep
        sample_rate = 44100
        duration = 0.3
        frequency = 440
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # generate wave sound
        wave = np.sin(frequency * 2 * np.pi * t) * 32767
        wave = wave.astype(np.int16)
        
        # bikin stereo sound dan play
        stereo_wave = np.column_stack((wave, wave))
        sound = pygame.sndarray.make_sound(stereo_wave)
        sound.play()
    except Exception as e:
        pass

# inisialisasi module mediapipe pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# setup styling untuk drawing skeleton
draw_spec_landmark = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
draw_spec_connection = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)

# setup folder dataset dan path file csv
DATASET_DIR = "dataset"
CSV_FILE = os.path.join(DATASET_DIR, "data_training_biceps.csv")
os.makedirs(DATASET_DIR, exist_ok=True)

# define header csv (hanya the core four biomechanical features)
CSV_HEADER = [
    "sample_id",
    "exercise",
    "active_arm",                
    "rom_elbow",                 
    "upper_arm_angle_std",       
    "torso_sway_range",          
    "shoulder_angle_range",      
    "label",
    "notes"
]

# create file csv dan tulis header kalau belum exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_HEADER)

EXERCISE_TYPE = "biceps"

# function untuk menghitung angle sendi berdasarkan 3 titik koordinat
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    # pastikan angle tidak lebih dari 180 derajat
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

# function menghitung angle vektor terhadap garis vertical y-axis (gravitasi)
def angle_with_y_axis(p_top, p_bottom):
    dx = p_bottom[0] - p_top[0]
    dy = p_bottom[1] - p_top[1]
    
    # convert ke degree angle
    angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
    return angle

# deklarasi variabel system
counter = 0
stage = "down"  
active_arm = None 
sample_id = 1
prev_time = 0
rep_start_time = 0

# siapkan array buffer untuk tampung data sudut per frame (koordinat piksel dihapus)
rep_data = {
    "elbow_angles": [], "shoulder_angles": [], "upper_arm_angles": [],
    "torso_angles": [], "visibilities": []
}

waiting_label = False
last_extracted_features = None
validation_msg = "Ready. Do 1 rep."

# open webcam
openCamera = cv2.VideoCapture(0)

# running mediapipe pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while openCamera.isOpened():
        success, frame = openCamera.read()
        if not success: break

        # mirror frame supaya natural view
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        
        # kalkulasi fps
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time

        # convert frame color untuk diproses mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # ekstrak tangan kanan (kita pakai landmark LEFT dari mediapipe efek dari cv2.flip)
            r_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            r_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            r_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            r_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            r_vis = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility + landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility) / 2.0
            r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

            # ekstrak tangan kiri (kita pakai landmark RIGHT dari mediapipe efek dari cv2.flip)
            l_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            l_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            l_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            l_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            l_vis = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility + landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility) / 2.0
            l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

            # cek active arm dan start rep
            if stage == "down" and not waiting_label:
                # jika siku kiri mulai naik duluan
                if l_elbow_angle < 130:
                    active_arm = "left"
                    stage = "up"
                    rep_start_time = time.time()
                    validation_msg = "Recording LEFT Arm..."
                # jika siku kanan mulai naik duluan
                elif r_elbow_angle < 130:
                    active_arm = "right"
                    stage = "up"
                    rep_start_time = time.time()
                    validation_msg = "Recording RIGHT Arm..."

            # proses rekam data hanya untuk lengan yang active
            if stage == "up" and not waiting_label:
                # set koordinat target berdasarkan active arm
                if active_arm == "left":
                    cur_shoulder, cur_elbow, cur_wrist, cur_hip, cur_vis = l_shoulder, l_elbow, l_wrist, l_hip, l_vis
                    cur_elbow_angle = l_elbow_angle
                else:
                    cur_shoulder, cur_elbow, cur_wrist, cur_hip, cur_vis = r_shoulder, r_elbow, r_wrist, r_hip, r_vis
                    cur_elbow_angle = r_elbow_angle

                # hitung angle features untuk machine learning
                shoulder_angle = calculate_angle(cur_hip, cur_shoulder, cur_elbow)
                upper_arm_angle = angle_with_y_axis(cur_shoulder, cur_elbow)
                torso_angle = angle_with_y_axis(cur_shoulder, cur_hip)

                # append hasil hitungan ke dictionary buffer murni sudut saja
                rep_data["elbow_angles"].append(cur_elbow_angle)
                rep_data["shoulder_angles"].append(shoulder_angle)
                rep_data["upper_arm_angles"].append(upper_arm_angle)
                rep_data["torso_angles"].append(torso_angle)
                rep_data["visibilities"].append(cur_vis)

                # trigger untuk end rep (gerakan balik ke posisi lurus)
                if cur_elbow_angle > 140:
                    stage = "down"
                    
                    # durasi waktu tidak lagi disimpan ke csv, murni jadi gatekeeper internal
                    rep_duration_internal = time.time() - rep_start_time
                    
                    # ekstrak data buat validasi dan features
                    rom_elbow = max(rep_data["elbow_angles"]) - min(rep_data["elbow_angles"])
                    avg_vis = sum(rep_data["visibilities"]) / len(rep_data["visibilities"])
                    
                    # gatekeeper rules buat filter data jelek
                    is_valid = True
                    if rom_elbow < 60:
                        validation_msg = f"DROP: ROM < 60 ({active_arm.upper()})"
                        is_valid = False
                    elif rep_duration_internal < 0.6:
                        validation_msg = "DROP: Terlalu Cepat (< 0.6s)"
                        is_valid = False
                    elif avg_vis < 0.6:
                        validation_msg = "DROP: Visibilitas rendah"
                        is_valid = False

                    # kalau data tembus gatekeeper, lanjut siapkan fitur core four
                    if is_valid:
                        counter += 1
                        # running beep pake thread biar frame nggak nge-freeze
                        threading.Thread(target=play_beep, daemon=True).start()
                        validation_msg = f"Valid ({active_arm.upper()})! C=Correct, I=Incorrect, S=Skip"
                        
                        # kumpulkan fitur ke dalam array untuk ditulis ke csv nanti
                        last_extracted_features = [
                            sample_id,
                            EXERCISE_TYPE,
                            active_arm, 
                            round(rom_elbow, 2),
                            round(np.std(rep_data["upper_arm_angles"]), 2),
                            round(max(rep_data["torso_angles"]) - min(rep_data["torso_angles"]), 2),
                            round(max(rep_data["shoulder_angles"]) - min(rep_data["shoulder_angles"]), 2)
                        ]
                        # hold system buat minta input label user
                        waiting_label = True
                    
                    # bersihkan array buffer buat rep selanjutnya
                    for key in rep_data:
                        rep_data[key].clear()

            # render text angle di area siku
            l_pixel = tuple(np.multiply(l_elbow, [width, height]).astype(int))
            r_pixel = tuple(np.multiply(r_elbow, [width, height]).astype(int))
            cv2.putText(frame, f"L:{int(l_elbow_angle)}", l_pixel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, f"R:{int(r_elbow_angle)}", r_pixel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # draw skeleton connections
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, draw_spec_landmark, draw_spec_connection)

        # render ui informasi text di pojok kiri atas
        cv2.putText(frame, f"Reps: {counter} | Stage: {stage} | Arm: {active_arm}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Status: {validation_msg}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if "Valid" in validation_msg else (0, 0, 255), 2)

        # baca input keyboard buat interact system
        key = cv2.waitKey(1) & 0xFF
        
        # logic kalau lagi mode input label manual
        if waiting_label and last_extracted_features is not None:
            label, notes = None, None
            if key == ord("c"):
                label, notes = "correct", "good form"
            elif key == ord("i"):
                label, notes = "incorrect", "bad form"
            elif key == ord("s"):
                print("[SKIPPED] Data dibuang.")
                waiting_label = False
                validation_msg = "Skipped. Do next rep."

            # save ke csv kalau user udah tekan tombol label
            if label:
                row_data = last_extracted_features + [label, notes]
                with open(CSV_FILE, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(row_data)
                
                print(f"\n[SAVED] ID: {sample_id} | Lengan: {active_arm} | Label: {label}")
                
                # increment ID dan update UI text
                sample_id += 1
                waiting_label = False
                validation_msg = "Saved! Do next rep."

        # quit app
        if key == ord("q"):
            break

        # tampilin preview frame
        cv2.imshow("Biceps CV Evaluator", frame)

# release resource kalau script udah kelar
openCamera.release()
cv2.destroyAllWindows()