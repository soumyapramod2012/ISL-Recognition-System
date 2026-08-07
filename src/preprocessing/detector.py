import cv2
import mediapipe as mp

from src.preprocessing.config import (
    FACE_MODEL,
    POSE_MODEL,
    HAND_MODEL,
)


BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode


class MediaPipeDetector:

    def __init__(self):

        face_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_MODEL)),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
        )

        self.face = mp.tasks.vision.FaceLandmarker.create_from_options(
            face_options
        )

        self.pose = mp.tasks.vision.PoseLandmarker.create_from_model_path(
            str(POSE_MODEL)
        )

        self.hand = mp.tasks.vision.HandLandmarker.create_from_model_path(
            str(HAND_MODEL)
        )

    def detect(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        pose_result = self.pose.detect(image)
        hand_result = self.hand.detect(image)

        return pose_result, hand_result

    def close(self):
        self.face.close()
        self.pose.close()
        self.hand.close()