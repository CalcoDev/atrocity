import cv2
import numpy as np
import mediapipe as mp
import sys

def calculate_rotation_vector(landmarks, image_shape):
    """
    Calculate rotation vector based on landmarks and image shape.
    """
    # Define 3D model points of a human face
    model_points = np.array([
        (0.0, 0.0, 0.0),            # Nose tip
        (0.0, -330.0, -65.0),       # Chin
        (-225.0, 170.0, -135.0),    # Left eye left corner
        (225.0, 170.0, -135.0),     # Right eye right corner
        (-150.0, -150.0, -125.0),   # Left mouth corner
        (150.0, -150.0, -125.0)     # Right mouth corner
    ], dtype=np.float32)

    # 2D image points from detected landmarks
    image_points = np.array([
        (landmarks[1][0], landmarks[1][1]),    # Nose tip
        (landmarks[152][0], landmarks[152][1]), # Chin
        (landmarks[263][0], landmarks[263][1]), # Left eye left corner
        (landmarks[33][0], landmarks[33][1]),   # Right eye right corner
        (landmarks[287][0], landmarks[287][1]), # Left mouth corner
        (landmarks[57][0], landmarks[57][1])    # Right mouth corner
    ], dtype=np.float32)

    # Camera matrix
    focal_length = image_shape[1]
    center = (image_shape[1] / 2, image_shape[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    # Assuming no lens distortion
    dist_coeffs = np.zeros((4, 1))

    # Solve PnP to find rotation and translation vectors
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    return rotation_vector

def main():
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not found!", flush=True)
        return

    # Create a thread to listen for stdin input
    import threading
    stop_flag = [False]

    def listen_for_input():
        for line in sys.stdin:
            if line.strip().upper() == 'Q':
                stop_flag[0] = True
                break

    input_thread = threading.Thread(target=listen_for_input, daemon=True)
    input_thread.start()

    while True:
        if stop_flag[0]:
            print("Exiting...", flush=True)
            break

        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture video!")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract landmarks
                height, width, _ = frame.shape
                landmarks = [
                    (int(lm.x * width), int(lm.y * height))
                    for lm in face_landmarks.landmark
                ]

                # Calculate rotation vector
                rotation_vector = calculate_rotation_vector(landmarks, frame.shape)

                # Print rotation vector
                print("!", rotation_vector.flatten(), flush=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

if __name__ == "__main__":
    print("Starting stuff!", flush=True)
    main()
    print("Finished and closed stuff!", flush=True)