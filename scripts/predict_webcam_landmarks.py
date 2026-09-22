import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.hand_landmark_interpolator import (
    HandLandmarkInterpolator,
)
from src.training.landmark_normalizer import (
    LandmarkNormalizer,
)
from src.training.sequence_generator import (
    SequenceGenerator,
)


MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "isl_lstm_class_weighted_legacy_compatible.h5"
)

LANDMARK_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "landmarks_legacy"
)

INPUT_FILES = [
    (
        "FAN",
        PROJECT_ROOT
        / "outputs"
        / "webcam_recordings"
        / "fan_live_landmarks.npy",
    ),
    (
        "SUMMER",
        PROJECT_ROOT
        / "outputs"
        / "webcam_recordings"
        / "summer_live_landmarks.npy",
    ),
    (
        "CLOCK",
        PROJECT_ROOT
        / "outputs"
        / "webcam_recordings"
        / "clock_live_landmarks.npy",
    ),
    (
        "WINTER",
        PROJECT_ROOT
        / "outputs"
        / "webcam_recordings"
        / "winter_legacy_mp_0.10.21_landmarks.npy",
    ),
]


def load_labels():

    return sorted(
        p.name
        for p in LANDMARK_ROOT.iterdir()
        if p.is_dir()
        and p.name != "Extra"
    )


def main():

    labels = load_labels()

    print("=" * 80)
    print("WEBCAM LANDMARK OFFLINE DIAGNOSTIC")
    print("=" * 80)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    interpolator = (
        HandLandmarkInterpolator(
            max_gap=5
        )
    )

    normalizer = LandmarkNormalizer()
    generator = SequenceGenerator()

    for true_name, path in INPUT_FILES:

        print()
        print("-" * 80)
        print(f"TRUE SAMPLE: {true_name}")
        print(f"FILE       : {path}")

        if not path.exists():

            print("FILE NOT FOUND")
            continue

        landmarks = np.load(path)

        print(
            f"Original shape: {landmarks.shape}"
        )

        if landmarks.ndim != 2:
            print("INVALID: not 2D")
            continue

        if landmarks.shape[1] != 258:
            print(
                f"INVALID: expected 258 features, "
                f"got {landmarks.shape[1]}"
            )
            continue

        # Exact training preprocessing
        interpolated = (
            interpolator.interpolate(
                landmarks
            )
        )

        normalized = (
            normalizer.normalize(
                interpolated
            )
        )

        sequence = (
            generator.generate(
                normalized
            )
        )

        probabilities = model.predict(
            np.expand_dims(
                sequence,
                axis=0,
            ),
            verbose=0,
        )[0]

        top_indices = np.argsort(
            probabilities
        )[::-1][:5]

        print()
        print("TOP 5 PREDICTIONS:")

        for rank, index in enumerate(
            top_indices,
            start=1,
        ):

            print(
                f"{rank}. "
                f"{labels[index]:20s} "
                f"{probabilities[index]:.4f}"
            )


if __name__ == "__main__":
    main()