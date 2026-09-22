import cv2
import numpy as np
import mediapipe as mp


class LegacyLandmarkExtractor:
    """
    MediaPipe 0.10.21 Holistic-based extractor.

    Output:
        33 pose landmarks × 4 values = 132
        21 left-hand landmarks × 3 values = 63
        21 right-hand landmarks × 3 values = 63
        Total = 258 features
    """

    def __init__(self):
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, frame):
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        result = self.holistic.process(rgb)

        landmarks = []

        # ---------------- Pose ----------------
        if result.pose_landmarks:
            for point in result.pose_landmarks.landmark:
                landmarks.extend([
                    point.x,
                    point.y,
                    point.z,
                    point.visibility,
                ])
        else:
            landmarks.extend([0.0] * (33 * 4))

        # ---------------- Left hand ----------------
        if result.left_hand_landmarks:
            for point in result.left_hand_landmarks.landmark:
                landmarks.extend([
                    point.x,
                    point.y,
                    point.z,
                ])
        else:
            landmarks.extend([0.0] * (21 * 3))

        # ---------------- Right hand ----------------
        if result.right_hand_landmarks:
            for point in result.right_hand_landmarks.landmark:
                landmarks.extend([
                    point.x,
                    point.y,
                    point.z,
                ])
        else:
            landmarks.extend([0.0] * (21 * 3))

        output = np.asarray(
            landmarks,
            dtype=np.float32,
        )

        if output.shape != (258,):
            raise ValueError(
                f"Unexpected landmark shape: {output.shape}"
            )

        return output

    def close(self):
        self.holistic.close()