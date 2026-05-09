# importing cv2 from opencv to accessing camera
# importing mediapipe as framework to detect pose
# importing time to calculate fps
import cv2
import mediapipe as mp
import time

# import numpy for calculating vectors and trigonometry
import numpy as np


# init mediapipe pose and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# set drawing spec for landmark and connection line
# color format is BGR: (0, 255, 0) = green, (255, 255, 255) = white
draw_spec_landmark = mp_drawing.DrawingSpec(
    color=(0, 255, 0),
    thickness=2,
    circle_radius=3
)

draw_spec_connection = mp_drawing.DrawingSpec(
    color=(255, 255, 255),
    thickness=2
)


# open camera with VideoCapture, 0 means default camera
openCam = cv2.VideoCapture(0)


# variable to store previous time for fps calculation
prev_time = 0


# function calculate angle
# a = shoulder, b = elbow, c = wrist
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # arctan2 is used to calculate vector direction
    radians = np.arctan2(
        c[1] - b[1],
        c[0] - b[0]
    ) - np.arctan2(
        a[1] - b[1],
        a[0] - b[0]
    )

    angle = np.abs(radians * 180.0 / np.pi)

    # normalize angle to 0 - 180 degrees
    if angle > 180:
        angle = 360 - angle

    return angle

# init first counter repetition
counter = 0
stage = None

# create Pose object with context manager
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while openCam.isOpened():
        success, frame = openCam.read()

        if not success:
            break

        # calculate current fps
        curr_time = time.time()

        if prev_time == 0:
            fps = 0
        else:
            fps = 1 / (curr_time - prev_time)

        prev_time = curr_time

        # get frame size
        height, width, _ = frame.shape

        # convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # set image as not writeable to improve performance
        image_rgb.flags.writeable = False

        # process image to detect pose landmarks
        result = pose.process(image_rgb)

        # set image back to writeable before drawing
        image_rgb.flags.writeable = True

        # if pose detected
        if result.pose_landmarks:
            # get all body landmarks
            landmarks = result.pose_landmarks.landmark

            # right shoulder point
            right_shoulder = [
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            ]

            # right elbow point
            right_elbow = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
            ]

            # right wrist point
            right_wrist = [
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
            ]

            # calculate right elbow angle
            right_elbow_angle = calculate_angle(
                right_shoulder,
                right_elbow,
                right_wrist
            )
            
            # counter logic
            if right_elbow_angle > 150:
                stage = "down"
            
            if right_elbow_angle < 50 and stage == "down":
                stage = "up"
                counter+=1

            # convert normalized coordinate to pixel
            right_elbow_pixel = tuple(
                np.multiply(right_elbow, [width, height]).astype(int)
            )

            # show angle value near elbow
            cv2.putText(
                frame,
                f"{int(right_elbow_angle)} deg",
                right_elbow_pixel,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # draw pose landmarks and connections
            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                draw_spec_landmark,
                draw_spec_connection
            )

        # display fps on top left corner
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # display counter on left corner
        cv2.putText(
            frame,
            f"Reps: {counter}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )
        
        cv2.imshow("Pose detection", frame)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# release camera and close all windows
openCam.release()
cv2.destroyAllWindows()