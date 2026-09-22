"""
Analyze temporal coverage of all generalized Clock landmark samples.

Purpose:
    1. Show frame counts for every Clock sample.
    2. Reproduce the exact training SequenceGenerator behavior.
    3. For videos longer than 60 frames, compare consecutive 60-frame
       windows with the training-style 60-frame sequence.
    4. For videos shorter than 60 frames, report that the training
       pipeline zero-pads them to 60 frames.

This is diagnostic only.
It does not modify the model, dataset, or realtime.py.
"""

from pathlib import Path
import json

import numpy as np
import tensorflow as tf

from src.training.hand_landmark_interpolator import HandLandmarkInterpolator
from src.training.landmark_normalizer import LandmarkNormalizer
from src.training.sequence_generator import SequenceGenerator


MODEL_PATH = Path(
    "saved_models/isl_lstm_generalized_114_legacy.keras"
)

LABEL_PATH = Path(
    "outputs/generalized_114_label_mapping.json"
)

DATASET_DIR = Path(
    "dataset/processed/generalized_landmarks_legacy"
)

SEQUENCE_LENGTH = 60
SLIDING_STEP = 10


def load_class_names():

    with open(
        LABEL_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        mapping = json.load(f)

    return [
        label
        for label, index
        in sorted(
            mapping.items(),
            key=lambda item: item[1]
        )
    ]


def preprocess(data):

    data = np.asarray(
        data,
        dtype=np.float32
    )

    data = HandLandmarkInterpolator().interpolate(data)
    data = LandmarkNormalizer().normalize(data)
    data = SequenceGenerator().generate(data)

    return data.astype(np.float32)


def predict(model, class_names, data):

    sequence = preprocess(data)

    probs = model.predict(
        sequence[np.newaxis, ...],
        verbose=0
    )[0]

    order = np.argsort(probs)[::-1]

    return [
        (
            class_names[int(i)],
            float(probs[int(i)])
        )
        for i in order[:5]
    ]


def find_clock_files():

    return sorted(
        p
        for p in (
            DATASET_DIR / "51. Clock"
        ).glob("*.npy")
    )


def main():

    print("=" * 80)
    print("CLOCK - TEMPORAL COVERAGE ANALYSIS")
    print("=" * 80)

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    class_names = load_class_names()

    files = find_clock_files()

    print(
        f"\nClock landmark samples: {len(files)}"
    )

    print("\n" + "-" * 80)
    print(
        f"{'File':35} "
        f"{'Frames':>7} "
        f"{'Training-style':30} "
        f"{'Conf.':>8}"
    )
    print("-" * 80)

    longer_files = []

    for path in files:

        data = np.load(path)

        results = predict(
            model,
            class_names,
            data
        )

        label, confidence = results[0]

        print(
            f"{path.name:35} "
            f"{len(data):7d} "
            f"{label:30} "
            f"{confidence * 100:7.2f}%"
        )

        if len(data) > SEQUENCE_LENGTH:
            longer_files.append(path)

    # --------------------------------------------------------
    # Detailed analysis for videos > 60 frames
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("CONSECUTIVE-WINDOW ANALYSIS FOR VIDEOS LONGER THAN 60 FRAMES")
    print("=" * 80)

    if not longer_files:

        print(
            "\nNo Clock videos contain more than 60 frames."
        )

        print(
            "Therefore the current Clock dataset cannot provide "
            "a true consecutive-60-vs-training-style comparison."
        )

    for path in longer_files:

        data = np.load(path)

        print("\n" + "-" * 80)
        print(
            f"{path.name} | {len(data)} frames"
        )

        starts = list(
            range(
                0,
                len(data) - SEQUENCE_LENGTH + 1,
                SLIDING_STEP
            )
        )

        final_start = (
            len(data) - SEQUENCE_LENGTH
        )

        if final_start not in starts:
            starts.append(final_start)

        starts = sorted(set(starts))

        for start in starts:

            window = data[
                start:start + SEQUENCE_LENGTH
            ]

            results = predict(
                model,
                class_names,
                window
            )

            label, confidence = results[0]

            print(
                f"  Frames "
                f"{start:4d}-{start + SEQUENCE_LENGTH - 1:4d}"
                f" -> {label:<30}"
                f" {confidence * 100:7.2f}%"
            )

    # --------------------------------------------------------
    # Explain short-video behavior
    # --------------------------------------------------------

    short_count = sum(
        len(np.load(path)) < SEQUENCE_LENGTH
        for path in files
    )

    exact_count = sum(
        len(np.load(path)) == SEQUENCE_LENGTH
        for path in files
    )

    long_count = sum(
        len(np.load(path)) > SEQUENCE_LENGTH
        for path in files
    )

    print("\n" + "=" * 80)
    print("FRAME-LENGTH SUMMARY")
    print("=" * 80)

    print(
        f"  < 60 frames : {short_count}"
    )

    print(
        f"  = 60 frames : {exact_count}"
    )

    print(
        f"  > 60 frames : {long_count}"
    )

    print("\nTraining behavior for short videos:")
    print(
        "  SequenceGenerator pads the sequence with zeros "
        "until it reaches exactly 60 frames."
    )

    print(
        "\nThis means MVI_4959 (58 frames) and "
        "MVI_4960 (57 frames) are NOT suitable for "
        "testing a true 60-consecutive-frame temporal window."
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
