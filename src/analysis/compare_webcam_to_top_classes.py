from pathlib import Path
import json
import numpy as np
import tensorflow as tf

from src.training.hand_landmark_interpolator import HandLandmarkInterpolator
from src.training.landmark_normalizer import LandmarkNormalizer
from src.training.sequence_generator import SequenceGenerator


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "saved_models" / "isl_lstm_generalized_114_legacy.keras"
MAPPING_PATH = ROOT / "outputs" / "generalized_114_label_mapping.json"
WEBCAM_PATH = ROOT / "outputs" / "webcam_test" / "clock_webcam_20260920_222015.npy"

DATASET_DIR = ROOT / "dataset" / "processed" / "generalized_landmarks_legacy"

TARGET_CLASSES = [
    "51. Clock",
    "12. Truck",
    "11. Car",
    "38. Page",
    "10. Plane",
    "27. Dream",
]


def load_label_mapping():
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # Handle either {index: label} or {label: index}
    if all(str(k).isdigit() for k in mapping.keys()):
        return {int(k): v for k, v in mapping.items()}

    return {int(v): k for k, v in mapping.items()}


def preprocess(raw):
    interpolator = HandLandmarkInterpolator(max_gap=5)
    normalizer = LandmarkNormalizer()
    generator = SequenceGenerator()

    x = interpolator.interpolate(raw)
    x = normalizer.normalize(x)
    x = generator.generate(x)

    return x


def normalize_for_distance(x):
    """
    Normalize the sequence for distance comparison.

    This removes overall magnitude differences while preserving
    the temporal landmark pattern.
    """
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)

    std[std < 1e-6] = 1.0

    return (x - mean) / std


def distance(a, b):
    """
    Mean Euclidean distance over all 60 frames.
    """
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def main():

    print("=" * 70)
    print("WEBCAM vs TRAINING LANDMARK SIMILARITY TEST")
    print("=" * 70)

    print("\nLoading webcam recording:")
    print(WEBCAM_PATH)

    webcam_raw = np.load(WEBCAM_PATH)

    print("Raw webcam shape:", webcam_raw.shape)

    webcam_seq = preprocess(webcam_raw)

    print("Preprocessed webcam shape:", webcam_seq.shape)

    webcam_cmp = normalize_for_distance(webcam_seq)

    label_mapping = load_label_mapping()

    print("\nClasses to compare:")
    for c in TARGET_CLASSES:
        print(" ", c)

    print("\nSearching training samples...\n")

    results = {}

    for class_name in TARGET_CLASSES:

        class_dir = DATASET_DIR / class_name

        if not class_dir.exists():
            print(f"WARNING: Missing class directory: {class_dir}")
            continue

        files = sorted(class_dir.glob("*.npy"))

        class_distances = []

        for file in files:

            try:
                raw = np.load(file)

                seq = preprocess(raw)

                seq_cmp = normalize_for_distance(seq)

                d = distance(webcam_cmp, seq_cmp)

                class_distances.append(
                    (d, file.name)
                )

            except Exception as e:
                print(
                    f"Skipping {file.name}: {e}"
                )

        class_distances.sort(key=lambda x: x[0])

        results[class_name] = class_distances

    print("\n")
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    for class_name in TARGET_CLASSES:

        distances = results.get(class_name, [])

        if not distances:
            continue

        values = [x[0] for x in distances]

        print("\n" + class_name)

        print("-" * 50)

        print(f"Samples:       {len(values)}")
        print(f"Minimum:       {np.min(values):.4f}")
        print(f"Mean:          {np.mean(values):.4f}")
        print(f"Median:        {np.median(values):.4f}")

        print("\nClosest 5 samples:")

        for d, name in distances[:5]:
            print(f"  {d:10.4f}   {name}")

    # ------------------------------------------------------------
    # Compare class statistics
    # ------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CLASS DISTANCE SUMMARY")
    print("=" * 70)

    summary = []

    for class_name, distances in results.items():

        if not distances:
            continue

        values = np.array([x[0] for x in distances])

        summary.append(
            (
                np.min(values),
                np.mean(values),
                np.median(values),
                class_name,
            )
        )

    summary.sort()

    print(
        f"\n{'Class':25s}"
        f"{'Min':>12s}"
        f"{'Mean':>12s}"
        f"{'Median':>12s}"
    )

    print("-" * 65)

    for min_d, mean_d, median_d, class_name in summary:

        print(
            f"{class_name:25s}"
            f"{min_d:12.4f}"
            f"{mean_d:12.4f}"
            f"{median_d:12.4f}"
        )

    # ------------------------------------------------------------
    # Model prediction
    # ------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MODEL PREDICTION")
    print("=" * 70)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    prediction = model.predict(
        webcam_seq[np.newaxis, ...],
        verbose=0
    )[0]

    top_indices = np.argsort(prediction)[::-1][:10]

    for rank, idx in enumerate(top_indices, start=1):

        label = label_mapping.get(
            int(idx),
            f"Class {idx}"
        )

        print(
            f"{rank:2d}. "
            f"{label:30s} "
            f"{prediction[idx] * 100:6.2f}%"
        )


if __name__ == "__main__":
    main()