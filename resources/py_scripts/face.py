import numpy as np
import cv2
import mediapipe as mp
import math

# Toggle variable to control displaying the camera output
show_camera_output = True

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# mp_

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)  # flipped for selfie view
    image.flags.writeable = False

    # reading data n stuff
    results = face_mesh.process(image)
    holistic_results = holistic.process(image)

    image.flags.writeable = True
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_2d = []
    face_3d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    face_2d.append([x, y])
                    face_3d.append(([x, y, lm.z]))

            # Get 2d Coord
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                  [0, focal_length, img_w / 2],
                                  [0, 0, 1]])
            distortion_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

            # getting rotational of face
            rmat, jac = cv2.Rodrigues(rotation_vec)

            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * math.tau
            y = angles[1] * math.tau
            z = angles[2] * math.tau * 0.0

            # aaaaa
            # _, rotation_matrix = cv2.Rodrigues(rotation_vec)
            # z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])  # Rotation around the Z-axis

            print(f"! {x} {y} {z}", flush=True)

            # head angle rotation
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rotation_vec, translation_vec, cam_matrix, distortion_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            # # Calculate offset from image center
            # center_x, center_y = img_w // 2, img_h // 2
            # offset_x = nose_2d[0] - center_x
            # offset_y = nose_2d[1] - center_y
            # print(f"+ {offset_x} {offset_y}", flush=True)
            print(f"+ {nose_2d[0]} {nose_2d[1]}", flush=True)

            # face stretch
            left_face = landmarks[234]
            right_face = landmarks[454]
            top_face = landmarks[10]
            bottom_face = landmarks[152]

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

        # print(f"Shoulders: Left ({left_shoulder.x * img_w}, {left_shoulder.y * img_h}), Right ({right_shoulder.x * img_w}, {right_shoulder.y * img_h})", flush=True)
        # print(f"Hips: Left ({left_hip.x * img_w}, {left_hip.y * img_h}), Right ({right_hip.x * img_w}, {right_hip.y * img_h})", flush=True)
        # print(f"Chest Center: {chest_center}", flush=True)
        
        print(f"shoulder 0: {[left_shoulder.x * img_w, left_shoulder.y * img_h]}")
        print(f"shoulder 1: {[right_shoulder.x * img_w, right_shoulder.y * img_h]}")
        
        print(f"hip 0: {[left_hip.x * img_w, left_hip.y * img_h]}")
        print(f"hip 1: {[right_hip.x * img_w, right_hip.y * img_h]}")

        if show_camera_output:
            cv2.circle(image, (int(mid_shoulder[0]), int(mid_shoulder[1])), 5, (0, 255, 0), -1)
            cv2.circle(image, (int(mid_hip[0]), int(mid_hip[1])), 5, (255, 0, 0), -1)
            cv2.circle(image, (int(chest_center[0]), int(chest_center[1])), 5, (0, 0, 255), -1)

            # Draw bounding box lines
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

    if show_camera_output:
        # Draw face debug points
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
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
