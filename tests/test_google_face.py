import os

import cv2
import mediapipe as mp
import pytest


IMAGE_PATH = r"test_data/face.jpg"
MODEL_PATH = r"models/face_landmarker.task"


def test_google_face_landmarker():

    if not os.path.exists(IMAGE_PATH):
        pytest.skip(
            f"Test image not found: {IMAGE_PATH}"
        )

    if not os.path.exists(MODEL_PATH):
        pytest.skip(
            f"Face landmarker model not found: {MODEL_PATH}"
        )

    frame = cv2.imread(IMAGE_PATH)

    assert frame is not None, (
        f"Unable to read image: {IMAGE_PATH}"
    )

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb,
    )

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = (
        mp.tasks.vision.FaceLandmarkerOptions
    )

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )

    detector = FaceLandmarker.create_from_options(
        options
    )

    try:

        result = detector.detect(image)

        assert result is not None

        assert hasattr(
            result,
            "face_landmarks",
        )

    finally:

        detector.close()