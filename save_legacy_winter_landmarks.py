import cv2
import numpy as np
from src.preprocessing.mediapipe_detector import detect_landmarks


video_path = r"outputs\webcam_recordings\winter_test_20260908_110507.mp4"
output_path = r"outputs\webcam_recordings\winter_legacy_mp_0.10.21_landmarks.npy"


cap = cv2.VideoCapture(video_path)

landmarks = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = detect_landmarks(frame)

    # Pose: 33 x 4
    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility]
             for lm in results.pose_landmarks.landmark],
            dtype=np.float32
        )
    else:
        pose = np.zeros((33, 4), dtype=np.float32)

    # Left hand: 21 x 3
    if results.left_hand_landmarks:
        left_hand = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32
        )
    else:
        left_hand = np.zeros((21, 3), dtype=np.float32)

    # Right hand: 21 x 3
    if results.right_hand_landmarks:
        right_hand = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32
        )
    else:
        right_hand = np.zeros((21, 3), dtype=np.float32)

    frame_landmarks = np.concatenate([
        pose.flatten(),
        left_hand.flatten(),
        right_hand.flatten()
    ])

    landmarks.append(frame_landmarks)


cap.release()

landmarks = np.array(landmarks, dtype=np.float32)

np.save(output_path, landmarks)

print()
print("=== Legacy MediaPipe Landmark Extraction ===")
print("MediaPipe     : 0.10.21")
print("Frames        :", len(landmarks))
print("Shape         :", landmarks.shape)
print("Saved to      :", output_path)