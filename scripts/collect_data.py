# import library dulu
import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import os

# init mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# drawing point skeleton
draw_spec_landmark = mp_drawing.DrawingSpec (
    color = (0, 255, 0), #inimah ijo jir
    thickness = 2,
    circle_radius = 3
)
# drawing connector per line skeleton
draw_spec_connection = mp_drawing.DrawingSpec (
    color = (255, 255, 0), #ini gtau warna apa coba jaa
    thickness = 2
)

# setup file ke dataset CSV
DATASET_DIR = "dataset"
CSV_FILE = os.path.join(DATASET_DIR, "data_training.csv")

# folder dataset jika belum ada
os.makedirs(DATASET_DIR, exist_ok= True)

# header CSV dataset
CSV_HEADER = [
    "exercise",
    "elbow_angle",
    "counter",
    "stage",
    "label"
]

# jika file belum ada, buat file dan header
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_HEADER)
        
# fungc buat hitng sudut
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(
        c[1] - b[1],
        c[0] - b[0]
    ) - np.arctan2(
        a[1] - b[1],
        a[0] - b[0]
    )

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle

# simpan data ke CSV
def save_data(exercise, elbow_angle, counter, stage, label):
    with open(CSV_FILE, mode ="a", newline = "" ) as file:
        writer = csv.writer(file)
        writer.writerow([
            exercise,
            round(elbow_angle, 2),
            counter,
            stage,
            label
        ])
        
    print(f"Data saved: {exercise}, {round(elbow_angle, 2)}, {counter}, {stage}, {label}")

# konfigurasi latihan
# pilihan latihan secara manual
# pilih biceps 
EXERCISE_TYPE = "biceps"

# variabel counter 
counter = 0
stage = None
previous_time = 0

# open kamera
openCamera = cv2.VideoCapture(0)

# proses dengan mediapipe
with mp_pose.Pose (
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as pose:

    while openCamera.isOpened():
        success, frame = openCamera.read() 
        
        if not success:
            print("gagal membuka kamera")
            break
        
        # hitung fps
        current_time = time.time()
        fps = 1/(current_time-previous_time)
        previous_time = current_time
        
        # ukuran frame
        height, width, _ = frame.shape
        
        # ubah warna frame 
        frame_rgb =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        # proses deteksi
        results = pose.process(frame_rgb)        
        frame_rgb.flags.writeable = True
        
        # variabel default
        right_elbow_angle = 0
        
        # jika pose terdeteksi 
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # koordinat sikut kanan
            right_shoulder = [
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            ]

            # Ambil titik siku kanan
            right_elbow = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
            ]

            # Ambil titik pergelangan kanan
            right_wrist = [
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
            ]

            # Hitung sudut siku kanan
            right_elbow_angle = calculate_angle(
                right_shoulder,
                right_elbow,
                right_wrist
            )

            # convert koordinat siku, bahu, dan tangan
            right_elbow_pixel = tuple(
                np.multiply(right_elbow, [width, height]).astype(int)
            )

            # logik perhitngan
            if right_elbow_angle > 150:
                stage = "down"

            if right_elbow_angle < 50 and stage == "down":
                stage = "up"
                counter += 1

            # tampilkan sudut di dekat siku
            cv2.putText(
                frame,
                f"{int(right_elbow_angle)}",
                right_elbow_pixel,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # gambar skeleton
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                draw_spec_landmark,
                draw_spec_connection
            )
            
            # informasi di tampilan
            cv2.putText(
            frame,
            f"Exercise: {EXERCISE_TYPE}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Reps: {counter}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Stage: {stage}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (10, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Press C=correct | I=incorrect | Q=quit",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        # keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            save_data(EXERCISE_TYPE, right_elbow_angle, counter, stage, "correct")
        elif key == ord('i'):
            save_data(EXERCISE_TYPE, right_elbow_angle, counter, stage, "incorrect")
        elif key == ord('q'):
            break
        
        cv2.imshow("collect training data", frame)
    
# release kamera
openCamera.release()
cv2.destroyAllWindows()