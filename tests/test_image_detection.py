import os

import cv2
import pytest

from src.preprocessing.detector import MediaPipeDetector


VIDEO_PATH = (
    r"dataset/raw/INCLUDE/Seasons/"
    r"61. Summer/MVI_4565.MOV"
)


def test_image_detection_pipeline():

    if not os.path.exists(VIDEO_PATH):

        pytest.skip(
            f"Test video not found: {VIDEO_PATH}"
        )

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    assert cap.isOpened(), (
        f"Unable to open video: {VIDEO_PATH}"
    )

    frame = None

    try:

        # Skip first 30 frames

        for _ in range(30):

            success, frame = cap.read()

            if not success:
                break

        assert success, (
            "Unable to read frame 30 from video"
        )

        assert frame is not None

        assert frame.size > 0

        # Convert frame to RGB

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        # Create MediaPipe image

        import mediapipe as mp

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        assert mp_image is not None

        # Create detector

        detector = MediaPipeDetector()

        try:

            hand_result = detector.hand.detect(
                mp_image
            )

            assert hand_result is not None

        finally:

            detector.close()

    finally:

        cap.release()