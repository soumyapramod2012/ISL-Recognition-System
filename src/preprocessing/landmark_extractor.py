import cv2
import numpy as np
import mediapipe as mp

from src.preprocessing.detector import MediaPipeDetector


class LandmarkExtractor:

    def __init__(self):
        self.detector = MediaPipeDetector()

    def extract(self, frame):

        pose_result, hand_result = self.detector.detect(frame)

        landmarks = []

        # ---------------- Pose ----------------

        if len(pose_result.pose_landmarks):

            for point in pose_result.pose_landmarks[0]:

                landmarks.extend([
                    point.x,
                    point.y,
                    point.z,
                    point.visibility,
                ])

        else:

            landmarks.extend([0.0] * (33 * 4))

        # ---------------- Hands ----------------

        left_hand = [0.0] * (21 * 3)
        right_hand = [0.0] * (21 * 3)

        for hand_landmarks, handedness in zip(
            hand_result.hand_landmarks,
            hand_result.handedness,
        ):

            hand_vector = []

            for point in hand_landmarks:

                hand_vector.extend([
                    point.x,
                    point.y,
                    point.z,
                ])

            label = handedness[0].category_name

            if label == "Left":
                left_hand = hand_vector

            elif label == "Right":
                right_hand = hand_vector

        landmarks.extend(left_hand)
        landmarks.extend(right_hand)

        return np.array(landmarks, dtype=np.float32)

    def close(self):
        self.detector.close()