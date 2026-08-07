import cv2

from src.preprocessing.landmark_extractor import LandmarkExtractor

VIDEO = r"dataset/raw/INCLUDE/Seasons/61. Summer/MVI_4565.MOV"

cap = cv2.VideoCapture(VIDEO)

for _ in range(30):
    ret, frame = cap.read()

cap.release()

extractor = LandmarkExtractor()

vector = extractor.extract(frame)

print("Vector Shape :", vector.shape)

print("Total Values :", len(vector))

extractor.close()