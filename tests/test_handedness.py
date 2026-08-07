import cv2

from src.preprocessing.detector import MediaPipeDetector

VIDEO = r"dataset/raw/INCLUDE/Seasons/61. Summer/MVI_4565.MOV"

cap = cv2.VideoCapture(VIDEO)

for _ in range(30):
    ret, frame = cap.read()

cap.release()

detector = MediaPipeDetector()

_, hand_result = detector.detect(frame)

print("Hands detected:", len(hand_result.hand_landmarks))

print()

print(hand_result.handedness)

detector.close()