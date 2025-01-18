import numpy as np
import cv2
import mediapipe as mp
import time
import math
import threading
import sys

print("Starting!", flush=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# Shared variable to stop the main loop
stop_flag = False

def listen_for_quit():
    global stop_flag
    while not stop_flag:
        user_input = input()
        if user_input.lower() == 'q':
            stop_flag = True

# Start the input listening thread
input_thread = threading.Thread(target=listen_for_quit, daemon=True)
input_thread.start()

while cap.isOpened() and not stop_flag:
    success, image = cap.read()
    if not success:
        break

    start = time.time()

    # Preprocess the image
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True

    img_h, img_w, _ = image.shape
    face_2d = []
    face_3d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            # Convert to numpy arrays
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # Camera matrix
            focal_length = 1 * img_w
            cam_matrix = np.array([
                [focal_length, 0, img_w / 2],
                [0, focal_length, img_h / 2],
                [0, 0, 1]
            ])

            # Distortion coefficients
            distortion_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rotation_vec, translation_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, distortion_matrix
            )

            # Get rotational matrix and angles
            rmat, _ = cv2.Rodrigues(rotation_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            x_angle = angles[0] * math.tau
            y_angle = angles[1] * math.tau
            z_angle = angles[2] * math.tau

            
            # y left right
            # z bottom top

            # Print angles in radians
            print(f"! {x_angle} {y_angle} {z_angle}", flush=True)

    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime

    if cv2.waitKey(5) & 0xFF == 27:  # Esc key to break (optional)
        break

cap.release()
print("Finished!", flush=True)
