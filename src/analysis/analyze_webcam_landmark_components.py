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


def preprocess(raw):
    """Apply the exact project preprocessing pipeline."""

    interpolator = HandLandmarkInterpolator(max_gap=5)
    normalizer = LandmarkNormalizer()
    generator = SequenceGenerator()

    x = interpolator.interpolate(raw)
    x = normalizer.normalize(x)
    x = generator.generate(x)

    return x


def component_distance(a, b, start, end):
    """
    Calculate mean Euclidean distance for a feature component.
    """

    a_part = a[:, start:end]
    b_part = b[:, start:end]

    return float(
        np.mean(
            np.linalg.norm(
                a_part - b_part,
                axis=1
            )
        )
    )


def calculate_distances(webcam, sample):

    return {
        "Pose": component_distance(
            webcam,
            sample,
            0,
            132
        ),

        "Left Hand": component_distance(
            webcam,
            sample,
            132,
            195
        ),

        "Right Hand": component_distance(
            webcam,
            sample,
            195,
            258
        ),

        "Overall": component_distance(
            webcam,
            sample,
            0,
            258
        ),
    }


def main():

    print("=" * 75)
    print("WEBCAM LANDMARK COMPONENT ANALYSIS")
    print("=" * 75)

    print("\nWebcam:")
    print(WEBCAM_PATH)

    webcam_raw = np.load(WEBCAM_PATH)

    print(
        "Raw shape:",
        webcam_raw.shape
    )

    webcam = preprocess(webcam_raw)

    print(
        "Preprocessed shape:",
        webcam.shape
    )

    all_results = {}

    # ------------------------------------------------------------
    # Process each target class
    # ------------------------------------------------------------

    for class_name in TARGET_CLASSES:

        class_dir = DATASET_DIR / class_name

        files = sorted(
            class_dir.glob("*.npy")
        )

        print(
            f"\nProcessing {class_name}: "
            f"{len(files)} samples"
        )

        class_results = []

        for file in files:

            try:

                raw = np.load(file)

                sample = preprocess(raw)

                distances = calculate_distances(
                    webcam,
                    sample
                )

                class_results.append(
                    (
                        file.name,
                        distances
                    )
                )

            except Exception as e:

                print(
                    f"Skipping {file.name}: {e}"
                )

        all_results[class_name] = class_results

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("COMPONENT DISTANCE SUMMARY")
    print("=" * 75)

    print(
        f"\n{'Class':20s}"
        f"{'Pose':>12s}"
        f"{'Left Hand':>14s}"
        f"{'Right Hand':>15s}"
        f"{'Overall':>12s}"
    )

    print("-" * 75)

    for class_name in TARGET_CLASSES:

        results = all_results[class_name]

        pose = np.array(
            [r[1]["Pose"] for r in results]
        )

        left = np.array(
            [r[1]["Left Hand"] for r in results]
        )

        right = np.array(
            [r[1]["Right Hand"] for r in results]
        )

        overall = np.array(
            [r[1]["Overall"] for r in results]
        )

        print(
            f"{class_name:20s}"
            f"{np.mean(pose):12.4f}"
            f"{np.mean(left):14.4f}"
            f"{np.mean(right):15.4f}"
            f"{np.mean(overall):12.4f}"
        )

    # ------------------------------------------------------------
    # Closest examples by each component
    # ------------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("CLOSEST TRAINING EXAMPLES")
    print("=" * 75)

    components = [
        "Pose",
        "Left Hand",
        "Right Hand",
        "Overall",
    ]

    for class_name in TARGET_CLASSES:

        print("\n")
        print(class_name)
        print("-" * 75)

        results = all_results[class_name]

        for component in components:

            ranked = sorted(
                results,
                key=lambda x: x[1][component]
            )

            print(
                f"\nClosest by {component}:"
            )

            for filename, distances in ranked[:5]:

                print(
                    f"  {distances[component]:10.4f}"
                    f"   {filename}"
                )

    # ------------------------------------------------------------
    # Best-of-class comparison
    # ------------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("BEST MATCH PER COMPONENT")
    print("=" * 75)

    for component in components:

        print(
            f"\n{component}"
        )

        print("-" * 50)

        for class_name in TARGET_CLASSES:

            results = all_results[class_name]

            ranked = sorted(
                results,
                key=lambda x: x[1][component]
            )

            best_filename, best_distances = ranked[0]

            print(
                f"{class_name:20s}"
                f"{best_distances[component]:10.4f}"
                f"   {best_filename}"
            )


if __name__ == "__main__":
    main()