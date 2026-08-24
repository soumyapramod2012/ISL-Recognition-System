import numpy as np

from src.training.hand_landmark_interpolator import (
    HandLandmarkInterpolator,
)


FEATURES = 258

POSE_FEATURES = 33 * 4
HAND_FEATURES = 21 * 3

LEFT_HAND_START = POSE_FEATURES
RIGHT_HAND_START = (
    POSE_FEATURES + HAND_FEATURES
)


def create_landmarks(
    frames=10,
):
    """
    Create a simple landmark sequence
    with valid values for all landmarks.
    """

    data = np.ones(
        (frames, FEATURES),
        dtype=np.float32,
    )

    return data


def make_hand_missing(
    data,
    start,
    end,
    hand_start,
):
    """
    Make an entire hand missing for
    frames [start, end].
    """

    data[
        start:end + 1,
        hand_start:
        hand_start + HAND_FEATURES,
    ] = 0.0


def test_single_frame_gap_is_interpolated():

    data = create_landmarks(
        frames=5
    )

    # Give neighboring frames different values.

    data[0, LEFT_HAND_START:
         LEFT_HAND_START + HAND_FEATURES] = 0.0

    data[0, LEFT_HAND_START:
         LEFT_HAND_START + HAND_FEATURES] = 1.0

    data[2, LEFT_HAND_START:
         LEFT_HAND_START + HAND_FEATURES] = 3.0

    make_hand_missing(
        data,
        start=1,
        end=1,
        hand_start=LEFT_HAND_START,
    )

    interpolator = HandLandmarkInterpolator(
        max_gap=5
    )

    result = interpolator.interpolate(
        data
    )

    expected = 2.0

    interpolated = result[
        1,
        LEFT_HAND_START:
        LEFT_HAND_START + HAND_FEATURES,
    ]

    assert np.allclose(
        interpolated,
        expected,
    )


def test_five_frame_gap_is_interpolated():

    data = create_landmarks(
        frames=7
    )

    data[
        0,
        LEFT_HAND_START:
        LEFT_HAND_START + HAND_FEATURES,
    ] = 1.0

    data[
        6,
        LEFT_HAND_START:
        LEFT_HAND_START + HAND_FEATURES,
    ] = 7.0

    make_hand_missing(
        data,
        start=1,
        end=5,
        hand_start=LEFT_HAND_START,
    )

    interpolator = HandLandmarkInterpolator(
        max_gap=5
    )

    result = interpolator.interpolate(
        data
    )

    for frame_index in range(1, 6):

        expected = float(
            frame_index + 1
        )

        actual = result[
            frame_index,
            LEFT_HAND_START:
            LEFT_HAND_START + HAND_FEATURES,
        ]

        assert np.allclose(
            actual,
            expected,
        )


def test_six_frame_gap_is_not_interpolated():

    data = create_landmarks(
        frames=8
    )

    data[
        0,
        LEFT_HAND_START:
        LEFT_HAND_START + HAND_FEATURES,
    ] = 1.0

    data[
        7,
        LEFT_HAND_START:
        LEFT_HAND_START + HAND_FEATURES,
    ] = 8.0

    make_hand_missing(
        data,
        start=1,
        end=6,
        hand_start=LEFT_HAND_START,
    )

    interpolator = HandLandmarkInterpolator(
        max_gap=5
    )

    result = interpolator.interpolate(
        data
    )

    missing = result[
        1:7,
        LEFT_HAND_START:
        LEFT_HAND_START + HAND_FEATURES,
    ]

    assert np.allclose(
        missing,
        0.0,
    )


def test_gap_at_beginning_is_not_interpolated():

    data = create_landmarks(
        frames=5
    )

    data[
        3,
        LEFT_HAND_START:
        LEFT_HAND_START + HAND_FEATURES,
    ] = 5.0

    make_hand_missing(
        data,
        start=0,
        end=2,
        hand_start=LEFT_HAND_START,
    )

    interpolator = HandLandmarkInterpolator(
        max_gap=5
    )

    result = interpolator.interpolate(
        data
    )

    missing = result[
        0:3,
        LEFT_HAND_START:
        LEFT_HAND_START + HAND_FEATURES,
    ]

    assert np.allclose(
        missing,
        0.0,
    )


def test_gap_at_end_is_not_interpolated():

    data = create_landmarks(
        frames=5
    )

    data[
        1,
        LEFT_HAND_START:
        LEFT_HAND_START + HAND_FEATURES,
    ] = 5.0

    make_hand_missing(
        data,
        start=2,
        end=4,
        hand_start=LEFT_HAND_START,
    )

    interpolator = HandLandmarkInterpolator(
        max_gap=5
    )

    result = interpolator.interpolate(
        data
    )

    missing = result[
        2:5,
        LEFT_HAND_START:
        LEFT_HAND_START + HAND_FEATURES,
    ]

    assert np.allclose(
        missing,
        0.0,
    )


def test_right_hand_is_also_interpolated():

    data = create_landmarks(
        frames=5
    )

    data[
        0,
        RIGHT_HAND_START:
        RIGHT_HAND_START + HAND_FEATURES,
    ] = 2.0

    data[
        2,
        RIGHT_HAND_START:
        RIGHT_HAND_START + HAND_FEATURES,
    ] = 6.0

    make_hand_missing(
        data,
        start=1,
        end=1,
        hand_start=RIGHT_HAND_START,
    )

    interpolator = HandLandmarkInterpolator(
        max_gap=5
    )

    result = interpolator.interpolate(
        data
    )

    interpolated = result[
        1,
        RIGHT_HAND_START:
        RIGHT_HAND_START + HAND_FEATURES,
    ]

    assert np.allclose(
        interpolated,
        4.0,
    )


def test_valid_landmarks_are_not_changed():

    data = create_landmarks(
        frames=10
    )

    original = data.copy()

    interpolator = HandLandmarkInterpolator(
        max_gap=5
    )

    result = interpolator.interpolate(
        data
    )

    assert np.array_equal(
        result,
        original,
    )


def test_pose_landmarks_are_not_changed():

    data = create_landmarks(
        frames=10
    )

    original_pose = data[
        :,
        :POSE_FEATURES,
    ].copy()

    # Create a hand gap.

    make_hand_missing(
        data,
        start=2,
        end=4,
        hand_start=LEFT_HAND_START,
    )

    interpolator = HandLandmarkInterpolator(
        max_gap=5
    )

    result = interpolator.interpolate(
        data
    )

    assert np.array_equal(
        result[
            :,
            :POSE_FEATURES,
        ],
        original_pose,
    )


def test_original_input_is_not_modified():

    data = create_landmarks(
        frames=5
    )

    make_hand_missing(
        data,
        start=1,
        end=1,
        hand_start=LEFT_HAND_START,
    )

    original = data.copy()

    interpolator = HandLandmarkInterpolator(
        max_gap=5
    )

    interpolator.interpolate(
        data
    )

    assert np.array_equal(
        data,
        original,
    )