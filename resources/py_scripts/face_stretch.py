import numpy as np
import cv2
import mediapipe as mp
import time
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()

    if not success:
        break

    start = time.time()

    # Preprocess the image
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)  # Flipped for selfie view
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_2d = []
    face_3d = []
    mouth_open = False
    face_stretched = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

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
            # if abs(aspect_ratio - baseline_aspect_ratio) > 0.2:  # Significant stretch
            #     face_stretched = True

            # Head offset calculation
            nose = landmarks[1]
            nose_2d = (nose.x * img_w, nose.y * img_h)
            center_x, center_y = img_w // 2, img_h // 2
            offset_x = nose_2d[0] - center_x
            offset_y = nose_2d[1] - center_y

            # Print the results
            print(f"! {face_landmarks.landmark[1].x} {face_landmarks.landmark[1].y} 0.0", flush=True)
            print(f"+ {offset_x} {offset_y}", flush=True)
            # print(f"Mouth Open: {mouth_open}", flush=True)
            # print(f"= {aspect_ratio}", flush=True)

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime
    print(f"FPS: {fps:.2f}", flush=True)

    # Exit on ESC key
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
