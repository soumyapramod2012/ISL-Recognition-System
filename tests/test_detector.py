from src.preprocessing.detector import MediaPipeDetector

detector = MediaPipeDetector()

print("Face Detector :", detector.face is not None)
print("Pose Detector :", detector.pose is not None)
print("Hand Detector :", detector.hand is not None)

detector.close()

print("Detector closed successfully.")