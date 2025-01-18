import numpy as np
import cv2
import mediapipe as mp
import time
import math

mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Toggle for rendering window
RENDER_WINDOW = False

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()

    if not success:
        break

    start = time.time()

    # Preprocess the image
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)  # Flipped for selfie view
    image.flags.writeable = False
    face_results = face_mesh.process(image)
    holistic_results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_2d = []
    face_3d = []
    mouth_open = False
    face_stretched = False

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Draw face mesh points on the image if rendering is enabled
            if RENDER_WINDOW:
                for idx, lm in enumerate(landmarks):
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

            # Mouth opening detection
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            upper_lip_y = upper_lip.y * img_h
            lower_lip_y = lower_lip.y * img_h
            mouth_open_distance = lower_lip_y - upper_lip_y

            # Face stretch detection
            left_face = landmarks[234]
            right_face = landmarks[454]
            top_face = landmarks[10]
            bottom_face = landmarks[152]

            face_width = (right_face.x - left_face.x) * img_w
            face_height = (bottom_face.y - top_face.y) * img_h
            aspect_ratio = face_width / face_height

            # Determine thresholds for notable changes
            if mouth_open_distance > img_h * 0.05:  # Mouth opens more than 5% of face height
                mouth_open = True

            baseline_aspect_ratio = 1.5  # Adjust based on typical face shape

            # Head offset calculation
            nose = landmarks[1]
            nose_2d = (nose.x * img_w, nose.y * img_h)
            center_x, center_y = img_w // 2, img_h // 2
            offset_x = nose_2d[0] - center_x
            offset_y = nose_2d[1] - center_y

            # Print the results
            print(f"! {face_landmarks.landmark[1].x} {face_landmarks.landmark[1].y} 0.0", flush=True)
            print(f"+ {offset_x} {offset_y}", flush=True)

    if holistic_results.pose_landmarks:
        # Draw pose landmarks (including arms) on the image if rendering is enabled
        if RENDER_WINDOW:
            for idx, lm in enumerate(holistic_results.pose_landmarks.landmark):
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                cv2.circle(image, (x, y), 2, (255, 0, 0), -1)

        # Print arm landmarks for each hand
        left_shoulder = holistic_results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
        left_elbow = holistic_results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW]
        left_wrist = holistic_results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST]
        right_shoulder = holistic_results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = holistic_results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW]
        right_wrist = holistic_results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]

        left_arm_points = [
            [left_shoulder.x * img_w, left_shoulder.y * img_h],
            [left_elbow.x * img_w, left_elbow.y * img_h],
            [left_wrist.x * img_w, left_wrist.y * img_h]
        ]

        right_arm_points = [
            [right_shoulder.x * img_w, right_shoulder.y * img_h],
            [right_elbow.x * img_w, right_elbow.y * img_h],
            [right_wrist.x * img_w, right_wrist.y * img_h]
        ]

        print(f"hand 0: {left_arm_points}", flush=True)
        print(f"hand 1: {right_arm_points}", flush=True)

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime

    # Display the image with face and pose landmarks if rendering is enabled
    if RENDER_WINDOW:
        cv2.imshow('Face and Pose Mesh', image)

        # Exit on ESC key
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
if RENDER_WINDOW:
    cv2.destroyAllWindows()
