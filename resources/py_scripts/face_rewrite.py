import numpy as np
import cv2
import mediapipe as mp
import time
import math

# Toggle variable to control displaying the camera output
show_camera_output = True

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()

    start = time.time()

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)  # flipped for selfie view

    image.flags.writeable = False

    holistic_results = holistic.process(image)

    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape

    if holistic_results.face_landmarks:
        face_2d = []
        face_3d = []
        for idx, lm in enumerate(holistic_results.face_landmarks.landmark):
            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                if idx == 1:
                    nose_2d = (lm.x * img_w, lm.y * img_h)
                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                x, y = int(lm.x * img_w), int(lm.y * img_h)

                face_2d.append([x, y])
                face_3d.append(([x, y, lm.z]))

        # Get 2D Coord
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = 1 * img_w

        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                               [0, focal_length, img_w / 2],
                               [0, 0, 1]])
        distortion_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

        # Getting rotational values
        rmat, jac = cv2.Rodrigues(rotation_vec)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        x = angles[0] * math.tau
        y = angles[1] * math.tau
        z = angles[2] * math.tau * 0.0

        print(f"! {x} {y} {z}", flush=True)

        nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rotation_vec, translation_vec, cam_matrix, distortion_matrix)

        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

        print(f"+ {nose_2d[0]} {nose_2d[1]}", flush=True)

        # Face stretch
        left_face = holistic_results.face_landmarks.landmark[234]
        right_face = holistic_results.face_landmarks.landmark[454]
        top_face = holistic_results.face_landmarks.landmark[10]
        bottom_face = holistic_results.face_landmarks.landmark[152]

        face_width = (right_face.x - left_face.x) * img_w
        face_height = (bottom_face.y - top_face.y) * img_h
        aspect_ratio = face_width / face_height
        print(f"= {aspect_ratio}", flush=True)

    if holistic_results.pose_landmarks:
        # Print arm landmarks for each hand
        left_shoulder = holistic_results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
        left_elbow = holistic_results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW]
        left_wrist = holistic_results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST]
        right_shoulder = holistic_results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = holistic_results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW]
        right_wrist = holistic_results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]

        left_hip = holistic_results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP]
        right_hip = holistic_results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP]

        mid_shoulder = [(left_shoulder.x + right_shoulder.x) / 2 * img_w, (left_shoulder.y + right_shoulder.y) / 2 * img_h]
        mid_hip = [(left_hip.x + right_hip.x) / 2 * img_w, (left_hip.y + right_hip.y) / 2 * img_h]

        chest_center = [(mid_shoulder[0] + mid_hip[0]) / 2, (mid_shoulder[1] + mid_hip[1]) / 2]

        print(f"shoulder 0: {[left_shoulder.x * img_w, left_shoulder.y * img_h]}")
        print(f"shoulder 1: {[right_shoulder.x * img_w, right_shoulder.y * img_h]}")

        print(f"hip 0: {[left_hip.x * img_w, left_hip.y * img_h]}")
        print(f"hip 1: {[right_hip.x * img_w, right_hip.y * img_h]}")

        if show_camera_output:
            cv2.circle(image, (int(mid_shoulder[0]), int(mid_shoulder[1])), 5, (0, 255, 0), -1)
            cv2.circle(image, (int(mid_hip[0]), int(mid_hip[1])), 5, (255, 0, 0), -1)
            cv2.circle(image, (int(chest_center[0]), int(chest_center[1])), 5, (0, 0, 255), -1)

            left_shoulder_point = (int(left_shoulder.x * img_w), int(left_shoulder.y * img_h))
            right_shoulder_point = (int(right_shoulder.x * img_w), int(right_shoulder.y * img_h))
            left_hip_point = (int(left_hip.x * img_w), int(left_hip.y * img_h))
            right_hip_point = (int(right_hip.x * img_w), int(right_hip.y * img_h))

            cv2.circle(image, left_shoulder_point, 5, (255, 255, 0), -1)
            cv2.circle(image, right_shoulder_point, 5, (255, 255, 0), -1)
            cv2.circle(image, left_hip_point, 5, (255, 255, 0), -1)
            cv2.circle(image, right_hip_point, 5, (255, 255, 0), -1)

            cv2.line(image, left_shoulder_point, right_shoulder_point, (0, 255, 255), 2)
            cv2.line(image, left_hip_point, right_hip_point, (0, 255, 255), 2)
            cv2.line(image, left_shoulder_point, left_hip_point, (0, 255, 255), 2)
            cv2.line(image, right_shoulder_point, right_hip_point, (0, 255, 255), 2)

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

    if holistic_results.right_hand_landmarks:
        # Get image dimensions for scaling
        img_h, img_w, _ = image.shape

        # Map MediaPipe landmark indices to fingers
        finger_connections = {
            "thumb": [1, 2, 3, 4],
            "index": [5, 6, 7, 8],
            "middle": [9, 10, 11, 12],
            "ring": [13, 14, 15, 16],
            "pinky": [17, 18, 19, 20]
        }

        # Draw lines for each finger
        for finger, indices in finger_connections.items():
            finger_points = []
            for i in range(len(indices)):
                thing = holistic_results.right_hand_landmarks.landmark[indices[i]]
                finger_points.append([thing.x * img_w, thing.y * img_h])
            print(f"{finger} 1: {finger_points}", flush=True)
            
            for i in range(len(indices) - 1):
                start_idx = indices[i]
                end_idx = indices[i + 1]
                start_landmark = holistic_results.right_hand_landmarks.landmark[start_idx]
                end_landmark = holistic_results.right_hand_landmarks.landmark[end_idx]

                start_point = (int(start_landmark.x * img_w), int(start_landmark.y * img_h))
                end_point = (int(end_landmark.x * img_w), int(end_landmark.y * img_h))

                cv2.line(image, start_point, end_point, (0, 255, 0), 2)  # Green lines for left hand

        # Optionally, draw circles for landmarks
        for lm in holistic_results.right_hand_landmarks.landmark:
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Red circles for landmarks
    
    if holistic_results.left_hand_landmarks:
        # Get image dimensions for scaling
        img_h, img_w, _ = image.shape

        # Map MediaPipe landmark indices to fingers
        finger_connections = {
            "thumb": [1, 2, 3, 4],
            "index": [5, 6, 7, 8],
            "middle": [9, 10, 11, 12],
            "ring": [13, 14, 15, 16],
            "pinky": [17, 18, 19, 20]
        }

        # Draw lines for each finger
        for finger, indices in finger_connections.items():
            finger_points = []
            for i in range(len(indices)):
                thing = holistic_results.left_hand_landmarks.landmark[indices[i]]
                finger_points.append([thing.x * img_w, thing.y * img_h])
            print(f"{finger} 0: {finger_points}", flush=True)
            
            for i in range(len(indices) - 1):
                start_idx = indices[i]
                end_idx = indices[i + 1]
                start_landmark = holistic_results.left_hand_landmarks.landmark[start_idx]
                end_landmark = holistic_results.left_hand_landmarks.landmark[end_idx]

                start_point = (int(start_landmark.x * img_w), int(start_landmark.y * img_h))
                end_point = (int(end_landmark.x * img_w), int(end_landmark.y * img_h))

                cv2.line(image, start_point, end_point, (0, 255, 0), 2)  # Green lines for left hand

        # Optionally, draw circles for landmarks
        for lm in holistic_results.left_hand_landmarks.landmark:
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Red circles for landmarks

    if show_camera_output:
        # Draw face debug points
        if holistic_results.face_landmarks:
            for lm in holistic_results.face_landmarks.landmark:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # Draw arm lines
        if holistic_results.pose_landmarks:
            cv2.line(image, tuple(map(int, left_arm_points[0])), tuple(map(int, left_arm_points[1])), (255, 0, 0), 2)
            cv2.line(image, tuple(map(int, left_arm_points[1])), tuple(map(int, left_arm_points[2])), (255, 0, 0), 2)
            cv2.line(image, tuple(map(int, right_arm_points[0])), tuple(map(int, right_arm_points[1])), (0, 0, 255), 2)
            cv2.line(image, tuple(map(int, right_arm_points[1])), tuple(map(int, right_arm_points[2])), (0, 0, 255), 2)

        # Display the image
        cv2.imshow('Camera Output', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
