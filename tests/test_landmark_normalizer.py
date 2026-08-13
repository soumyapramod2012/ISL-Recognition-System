import numpy as np

from src.training.landmark_normalizer import (
    LandmarkNormalizer,
)


def test_normalizer_output_shape():

    landmarks = np.zeros(
        (10, 258),
        dtype=np.float32,
    )

    # Create shoulders.

    landmarks[:, 11 * 4:11 * 4 + 3] = [
        0.4,
        0.5,
        0.0,
    ]

    landmarks[:, 12 * 4:12 * 4 + 3] = [
        0.6,
        0.5,
        0.0,
    ]

    normalizer = LandmarkNormalizer()

    result = normalizer.normalize(
        landmarks
    )

    assert result.shape == landmarks.shape


def test_shoulders_are_centered_and_scaled():

    landmarks = np.zeros(
        (1, 258),
        dtype=np.float32,
    )

    # Left shoulder

    landmarks[
        0,
        11 * 4:11 * 4 + 3
    ] = [
        0.4,
        0.5,
        0.0,
    ]

    # Right shoulder

    landmarks[
        0,
        12 * 4:12 * 4 + 3
    ] = [
        0.6,
        0.5,
        0.0,
    ]

    normalizer = LandmarkNormalizer()

    result = normalizer.normalize(
        landmarks
    )

    left = result[
        0,
        11 * 4:11 * 4 + 3
    ]

    right = result[
        0,
        12 * 4:12 * 4 + 3
    ]

    np.testing.assert_allclose(
        left,
        [-0.5, 0.0, 0.0],
        atol=1e-6,
    )

    np.testing.assert_allclose(
        right,
        [0.5, 0.0, 0.0],
        atol=1e-6,
    )


def test_missing_landmarks_remain_zero():

    landmarks = np.zeros(
        (1, 258),
        dtype=np.float32,
    )

    landmarks[
        0,
        11 * 4:11 * 4 + 3
    ] = [
        0.4,
        0.5,
        0.0,
    ]

    landmarks[
        0,
        12 * 4:12 * 4 + 3
    ] = [
        0.6,
        0.5,
        0.0,
    ]

    normalizer = LandmarkNormalizer()

    result = normalizer.normalize(
        landmarks
    )

    left_hand = result[
        0,
        132:195
    ]

    right_hand = result[
        0,
        195:258
    ]

    assert np.all(
        left_hand == 0
    )

    assert np.all(
        right_hand == 0
    )