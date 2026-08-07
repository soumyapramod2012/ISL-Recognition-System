import cv2
import mediapipe as mp

IMAGE_PATH = r"test_data/face.jpg"

frame = cv2.imread(IMAGE_PATH)

if frame is None:
    print("Unable to read image.")
    exit()

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=frame_rgb
)

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/face_landmarker.task"
    ),
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)

detector = FaceLandmarker.create_from_options(options)

result = detector.detect(image)

print(result)
print("Faces:", len(result.face_landmarks))

detector.close()