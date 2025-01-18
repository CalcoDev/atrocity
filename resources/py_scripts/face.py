import numpy as np
import cv2
import mediapipe as mp
import time
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()

    start = time.time()

    image = cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB) #flipped for selfie view

    image.flags.writeable = False

    results = face_mesh.process(image)
    
    holistic_results = holistic.process(image)

    image.flags.writeable = True

    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

    img_h , img_w, img_c = image.shape
    face_2d = []
    face_3d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx ==1 or idx == 61 or idx == 291 or idx==199:
                    if idx ==1:
                        nose_2d = (lm.x * img_w,lm.y * img_h)
                        nose_3d = (lm.x * img_w,lm.y * img_h,lm.z * 3000)
                    x,y = int(lm.x * img_w),int(lm.y * img_h)

                    face_2d.append([x,y])
                    face_3d.append(([x,y,lm.z]))


            #Get 2d Coord
            face_2d = np.array(face_2d,dtype=np.float64)

            face_3d = np.array(face_3d,dtype=np.float64)

            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length,0,img_h/2],
                                  [0,focal_length,img_w/2],
                                  [0,0,1]])
            distortion_matrix = np.zeros((4,1),dtype=np.float64)

            success,rotation_vec,translation_vec = cv2.solvePnP(face_3d,face_2d,cam_matrix,distortion_matrix)


            #getting rotational of face
            rmat,jac = cv2.Rodrigues(rotation_vec)

            angles,mtxR,mtxQ,Qx,Qy,Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * math.tau
            y = angles[1] * math.tau
            z = angles[2] * math.tau * 0.0
            print(f"! {x} {y} {z}", flush=True)

            # head angle rotation
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rotation_vec, translation_vec, cam_matrix, distortion_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            # Calculate offset from image center
            center_x, center_y = img_w // 2, img_h // 2
            offset_x = nose_2d[0] - center_x
            offset_y = nose_2d[1] - center_y
            print(f"+ {offset_x} {offset_y}", flush=True)

            # face stretcj
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
        totalTime = end-start

        fps = 1/totalTime
    if cv2.waitKey(5) & 0xFF ==27:
        break
cap.release()
