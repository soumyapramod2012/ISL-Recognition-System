from pathlib import Path
import numpy as np

from src.training.hand_landmark_interpolator import HandLandmarkInterpolator
from src.training.landmark_normalizer import LandmarkNormalizer
from src.training.sequence_generator import SequenceGenerator


ROOT = Path(__file__).resolve().parents[2]

WEBCAM_PATH = (
    ROOT
    / "outputs"
    / "webcam_test"
    / "clock_webcam_20260920_222015.npy"
)

DATASET_DIR = (
    ROOT
    / "dataset"
    / "processed"
    / "generalized_landmarks_legacy"
)

TARGET_CLASSES = [
    "51. Clock",
    "12. Truck",
]


# ------------------------------------------------------------
# Landmark structure
# ------------------------------------------------------------

POSE_START = 0
POSE_END = 132

LEFT_START = 132
LEFT_END = 195

RIGHT_START = 195
RIGHT_END = 258


def raw_hand_detection_stats(raw):
    """
    Determine whether each hand is present in the raw MediaPipe
    landmark sequence.

    Missing hands are represented by all-zero coordinates.
    """

    left = raw[:, LEFT_START:LEFT_END]
    right = raw[:, RIGHT_START:RIGHT_END]

    left_present = np.any(
        np.abs(left) > 1e-8,
        axis=1
    )

    right_present = np.any(
        np.abs(right) > 1e-8,
        axis=1
    )

    return left_present, right_present


def preprocess(raw):

    interpolator = HandLandmarkInterpolator(max_gap=5)
    normalizer = LandmarkNormalizer()
    generator = SequenceGenerator()

    x = interpolator.interpolate(raw)
    x = normalizer.normalize(x)
    x = generator.generate(x)

    return x


def print_detection_stats(name, raw):

    left, right = raw_hand_detection_stats(raw)

    total = len(raw)

    left_count = int(np.sum(left))
    right_count = int(np.sum(right))

    both_count = int(np.sum(left & right))
    neither_count = int(np.sum(~left & ~right))

    print("\n" + name)
    print("-" * 65)

    print(f"Frames:                    {total}")

    print(
        f"Left hand detected:        "
        f"{left_count:3d} / {total:3d} "
        f"({left_count / total * 100:6.2f}%)"
    )

    print(
        f"Right hand detected:       "
        f"{right_count:3d} / {total:3d} "
        f"({right_count / total * 100:6.2f}%)"
    )

    print(
        f"Both hands detected:       "
        f"{both_count:3d} / {total:3d} "
        f"({both_count / total * 100:6.2f}%)"
    )

    print(
        f"Neither hand detected:     "
        f"{neither_count:3d} / {total:3d} "
        f"({neither_count / total * 100:6.2f}%)"
    )

    print(
        "\nMissing left-hand frames:",
        np.where(~left)[0].tolist()
    )

    print(
        "Missing right-hand frames:",
        np.where(~right)[0].tolist()
    )

    return left, right


def component_statistics(raw):

    left = raw[:, LEFT_START:LEFT_END]
    right = raw[:, RIGHT_START:RIGHT_END]

    left_present, right_present = raw_hand_detection_stats(raw)

    stats = {}

    if np.any(left_present):
        stats["left_mean"] = float(
            np.mean(
                np.abs(left[left_present])
            )
        )
    else:
        stats["left_mean"] = 0.0

    if np.any(right_present):
        stats["right_mean"] = float(
            np.mean(
                np.abs(right[right_present])
            )
        )
    else:
        stats["right_mean"] = 0.0

    stats["left_std"] = float(np.std(left))
    stats["right_std"] = float(np.std(right))

    return stats


def main():

    print("=" * 75)
    print("WEBCAM HAND DETECTION ANALYSIS")
    print("=" * 75)

    # ------------------------------------------------------------
    # Webcam
    # ------------------------------------------------------------

    print("\nLoading webcam recording:")
    print(WEBCAM_PATH)

    webcam_raw = np.load(
        WEBCAM_PATH
    )

    print(
        "\nWebcam raw shape:",
        webcam_raw.shape
    )

    webcam_left, webcam_right = print_detection_stats(
        "WEBCAM CLOCK",
        webcam_raw
    )

    webcam_stats = component_statistics(
        webcam_raw
    )

    print("\nWebcam component statistics:")
    print(
        f"Left hand mean magnitude:  "
        f"{webcam_stats['left_mean']:.6f}"
    )
    print(
        f"Right hand mean magnitude: "
        f"{webcam_stats['right_mean']:.6f}"
    )
    print(
        f"Left hand std:             "
        f"{webcam_stats['left_std']:.6f}"
    )
    print(
        f"Right hand std:            "
        f"{webcam_stats['right_std']:.6f}"
    )

    # ------------------------------------------------------------
    # Training samples
    # ------------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("TRAINING DATA HAND DETECTION")
    print("=" * 75)

    for class_name in TARGET_CLASSES:

        class_dir = DATASET_DIR / class_name

        files = sorted(
            class_dir.glob("*.npy")
        )

        print("\n")
        print("=" * 75)
        print(class_name)
        print("=" * 75)

        print(
            "Number of samples:",
            len(files)
        )

        sample_left_rates = []
        sample_right_rates = []

        for file in files:

            try:

                raw = np.load(file)

                left, right = raw_hand_detection_stats(
                    raw
                )

                sample_left_rates.append(
                    np.mean(left) * 100
                )

                sample_right_rates.append(
                    np.mean(right) * 100
                )

            except Exception as e:

                print(
                    "Skipping",
                    file.name,
                    ":",
                    e
                )

        if not sample_left_rates:
            continue

        print(
            "\nLeft hand detection:"
        )

        print(
            f"  Minimum: "
            f"{np.min(sample_left_rates):.2f}%"
        )

        print(
            f"  Mean:    "
            f"{np.mean(sample_left_rates):.2f}%"
        )

        print(
            f"  Maximum: "
            f"{np.max(sample_left_rates):.2f}%"
        )

        print(
            "\nRight hand detection:"
        )

        print(
            f"  Minimum: "
            f"{np.min(sample_right_rates):.2f}%"
        )

        print(
            f"  Mean:    "
            f"{np.mean(sample_right_rates):.2f}%"
        )

        print(
            f"  Maximum: "
            f"{np.max(sample_right_rates):.2f}%"
        )

    # ------------------------------------------------------------
    # Frame-by-frame webcam report
    # ------------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("WEBCAM FRAME-BY-FRAME HAND DETECTION")
    print("=" * 75)

    print(
        f"\n{'Frame':>8}"
        f"{'Left':>12}"
        f"{'Right':>12}"
    )

    print("-" * 35)

    for i in range(len(webcam_raw)):

        print(
            f"{i:8d}"
            f"{'YES' if webcam_left[i] else 'NO':>12}"
            f"{'YES' if webcam_right[i] else 'NO':>12}"
        )


if __name__ == "__main__":
    main()