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


def preprocess(raw):

    interpolator = HandLandmarkInterpolator(max_gap=5)
    normalizer = LandmarkNormalizer()
    generator = SequenceGenerator()

    x = interpolator.interpolate(raw)
    x = normalizer.normalize(x)
    x = generator.generate(x)

    return x


def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("\nModel layers:")
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.output_shape)

    return model


def main():

    print("=" * 70)
    print("MODEL FEATURE-SPACE ANALYSIS")
    print("=" * 70)

    model = load_model()

    # ------------------------------------------------------------
    # Find the Dense(64) layer
    # ------------------------------------------------------------

    feature_layer = None

    for layer in model.layers:

        output_shape = getattr(
            layer,
            "output_shape",
            None
        )

        if output_shape is not None:

            if len(output_shape) == 2 and output_shape[-1] == 64:
                feature_layer = layer

    if feature_layer is None:
        raise RuntimeError(
            "Could not find the 64-dimensional feature layer."
        )

    print(
        "\nFeature layer:",
        feature_layer.name
    )

    feature_model = tf.keras.Model(
        inputs=model.input,
        outputs=feature_layer.output
    )

    # ------------------------------------------------------------
    # Webcam
    # ------------------------------------------------------------

    webcam_raw = np.load(WEBCAM_PATH)

    webcam_seq = preprocess(webcam_raw)

    webcam_feature = feature_model.predict(
        webcam_seq[np.newaxis, ...],
        verbose=0
    )[0]

    print(
        "Webcam feature shape:",
        webcam_feature.shape
    )

    # ------------------------------------------------------------
    # Compare against training samples
    # ------------------------------------------------------------

    results = {}

    for class_name in TARGET_CLASSES:

        class_dir = DATASET_DIR / class_name

        if not class_dir.exists():
            print(
                "Missing:",
                class_dir
            )
            continue

        distances = []

        for file in sorted(class_dir.glob("*.npy")):

            try:

                raw = np.load(file)

                seq = preprocess(raw)

                feature = feature_model.predict(
                    seq[np.newaxis, ...],
                    verbose=0
                )[0]

                d = float(
                    np.linalg.norm(
                        webcam_feature - feature
                    )
                )

                distances.append(
                    (d, file.name)
                )

            except Exception as e:

                print(
                    "Skipping",
                    file.name,
                    ":",
                    e
                )

        distances.sort(
            key=lambda x: x[0]
        )

        results[class_name] = distances

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MODEL FEATURE-SPACE DISTANCES")
    print("=" * 70)

    print(
        f"\n{'Class':25s}"
        f"{'Min':>12s}"
        f"{'Mean':>12s}"
        f"{'Median':>12s}"
    )

    print("-" * 65)

    summary = []

    for class_name, distances in results.items():

        values = np.array(
            [x[0] for x in distances]
        )

        summary.append(
            (
                np.min(values),
                np.mean(values),
                np.median(values),
                class_name
            )
        )

    summary.sort()

    for min_d, mean_d, median_d, class_name in summary:

        print(
            f"{class_name:25s}"
            f"{min_d:12.4f}"
            f"{mean_d:12.4f}"
            f"{median_d:12.4f}"
        )

    # ------------------------------------------------------------
    # Closest examples
    # ------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CLOSEST MODEL-SPACE EXAMPLES")
    print("=" * 70)

    for class_name, distances in results.items():

        print("\n" + class_name)

        for d, filename in distances[:5]:

            print(
                f"  {d:10.4f}   {filename}"
            )

    # ------------------------------------------------------------
    # WebCam feature vector
    # ------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("WEBCAM FEATURE VECTOR")
    print("=" * 70)

    print(webcam_feature)


if __name__ == "__main__":
    main()