import numpy as np

from src.training.sequence_generator import SequenceGenerator


def test_short_sequence_zero_padding():

    landmarks = np.random.rand(
        38,
        258,
    ).astype(np.float32)

    generator = SequenceGenerator()

    result = generator.generate(
        landmarks
    )

    assert result.shape == (
        60,
        258,
    )

    # Original baseline behavior:
    # short sequences are zero-padded.

    assert np.allclose(
        result[:38],
        landmarks,
    )

    assert np.all(
        result[38:] == 0
    )


def test_exact_sequence_length():

    landmarks = np.random.rand(
        60,
        258,
    ).astype(np.float32)

    generator = SequenceGenerator()

    result = generator.generate(
        landmarks
    )

    assert result.shape == (
        60,
        258,
    )

    assert np.allclose(
        result,
        landmarks,
    )


def test_long_sequence_resampling():

    landmarks = np.random.rand(
        138,
        258,
    ).astype(np.float32)

    generator = SequenceGenerator()

    result = generator.generate(
        landmarks
    )

    assert result.shape == (
        60,
        258,
    )

    # The first and last frames should be preserved
    # by the np.linspace-based sampling.

    assert np.allclose(
        result[0],
        landmarks[0],
    )

    assert np.allclose(
        result[-1],
        landmarks[-1],
    )