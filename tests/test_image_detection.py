import cv2
import mediapipe as mp

from src.preprocessing.detector import MediaPipeDetector

IMAGE_PATH = r"dataset/raw/INCLUDE/Seasons/61. Summer/MVI_4565.MOV"

cap = cv2.VideoCapture(IMAGE_PATH)

# Skip first 30 frames
for _ in range(30):
    success, frame = cap.read()

if not success:
    print("Unable to read frame.")
    exit()

cap.release()

if not success:
    print("Unable to read first frame.")
    exit()

cv2.imshow("First Frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=frame_rgb
)'''

saved_frame = cv2.imread("frame30.jpg")

# Increase contrast
saved_frame = cv2.convertScaleAbs(
    saved_frame,
    alpha=1.8,
    beta=20
)

cv2.imshow("Enhanced", saved_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

saved_frame_rgb = cv2.cvtColor(saved_frame, cv2.COLOR_BGR2RGB)

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=saved_frame_rgb
)

detector = MediaPipeDetector()

print("Image created successfully.")
print(mp_image)

'''print("Frame shape:", frame.shape)
print("Frame dtype:", frame.dtype)'''

cv2.imwrite("frame30.jpg", frame)
print("Frame saved as frame30.jpg")

'''face_result = detector.face.detect(mp_image)

print("Face result:", face_result)
print("Face landmarks:", face_result.face_landmarks)
print("Face blendshapes:", face_result.face_blendshapes)
print("Facial transformation matrices:", face_result.facial_transformation_matrixes)'''

'''pose_result = detector.pose.detect(mp_image)

print("Number of poses:", len(pose_result.pose_landmarks))

if len(pose_result.pose_landmarks) > 0:
    print("Pose landmarks:", len(pose_result.pose_landmarks[0]))'''

hand_result = detector.hand.detect(mp_image)

print("Number of hands:", len(hand_result.hand_landmarks))

if len(hand_result.hand_landmarks) > 0:
    for i, hand in enumerate(hand_result.hand_landmarks):
        print(f"Hand {i+1} landmarks:", len(hand))

detector.close()