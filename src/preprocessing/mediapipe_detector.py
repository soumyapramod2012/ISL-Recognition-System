"""
MediaPipe Holistic Detector
"""

import cv2
import mediapipe as mp


mp_holistic = mp.solutions.holistic


holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
)


def detect_landmarks(frame):
    """
    Detects landmarks from one frame.
    """

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = holistic.process(rgb_frame)

    return results