import cv2

from src.preprocessing.detector import MediaPipeDetector

VIDEO = r"dataset/raw/INCLUDE/Seasons/61. Summer/MVI_4565.MOV"

cap = cv2.VideoCapture(VIDEO)

for _ in range(30):
    ret, frame = cap.read()

cap.release()

detector = MediaPipeDetector()

pose, hands = detector.detect(frame)

print("Pose :", len(pose.pose_landmarks))

print("Hands :", len(hands.hand_landmarks))

if len(pose.pose_landmarks):
    print("Pose points :", len(pose.pose_landmarks[0]))

for i, hand in enumerate(hands.hand_landmarks):
    print(f"Hand {i+1} :", len(hand))

detector.close()