from pathlib import Path
import numpy as np
import tensorflow as tf
import json

from src.training.hand_landmark_interpolator import HandLandmarkInterpolator
from src.training.landmark_normalizer import LandmarkNormalizer
from src.training.sequence_generator import SequenceGenerator


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "saved_models" / "isl_lstm_generalized_114_legacy.keras"
MAPPING_PATH = ROOT / "outputs" / "generalized_114_label_mapping.json"

WEBCAM_PATH = (
    ROOT
    / "outputs"
    / "webcam_test"
    / "clock_webcam_20260920_222015.npy"
)


def preprocess(raw):

    interpolator = HandLandmarkInterpolator(max_gap=5)
    normalizer = LandmarkNormalizer()
    generator = SequenceGenerator()

    x = interpolator.interpolate(raw)
    x = normalizer.normalize(x)
    x = generator.generate(x)

    return x


def load_mapping():

    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    if all(str(k).isdigit() for k in mapping.keys()):
        return {int(k): v for k, v in mapping.items()}

    return {int(v): k for k, v in mapping.items()}


def predict_segment(model, mapping, name, raw_segment):

    sequence = preprocess(raw_segment)

    prediction = model.predict(
        sequence[np.newaxis, ...],
        verbose=0
    )[0]

    top_indices = np.argsort(prediction)[::-1][:10]

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Raw frames: {len(raw_segment)}"
    )

    print(
        f"Model input: {sequence.shape}"
    )

    print("\nTop predictions:")

    for rank, idx in enumerate(top_indices, start=1):

        label = mapping.get(
            int(idx),
            f"Class {idx}"
        )

        print(
            f"{rank:2d}. "
            f"{label:30s} "
            f"{prediction[idx] * 100:6.2f}%"
        )

    print("\nSpecific classes:")

    for target in [
        "51. Clock",
        "12. Truck",
        "11. Car",
        "10. Plane",
        "38. Page",
        "27. Dream",
    ]:

        try:
            idx = next(
                k for k, v in mapping.items()
                if v == target
            )

            print(
                f"{target:25s} "
                f"{prediction[idx] * 100:6.2f}%"
            )

        except StopIteration:
            print(
                f"{target:25s} NOT FOUND"
            )


def main():

    print("=" * 70)
    print("WEBCAM GESTURE SEGMENT TEST")
    print("=" * 70)

    raw = np.load(WEBCAM_PATH)

    print(
        "\nOriginal recording:",
        raw.shape
    )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    mapping = load_mapping()

    # ------------------------------------------------------------
    # Test 1: Full recording
    # ------------------------------------------------------------

    predict_segment(
        model,
        mapping,
        "TEST 1 — FULL RECORDING (0–65)",
        raw[0:66]
    )

    # ------------------------------------------------------------
    # Test 2: Gesture only
    # Frames 18–46 inclusive
    # ------------------------------------------------------------

    predict_segment(
        model,
        mapping,
        "TEST 2 — GESTURE ONLY (18–46)",
        raw[18:47]
    )

    # ------------------------------------------------------------
    # Test 3: Expanded gesture
    # Frames 15–49 inclusive
    # ------------------------------------------------------------

    predict_segment(
        model,
        mapping,
        "TEST 3 — EXPANDED GESTURE (15–49)",
        raw[15:50]
    )

    # ------------------------------------------------------------
    # Test 4: Slightly wider gesture
    # Frames 14–50 inclusive
    # ------------------------------------------------------------

    predict_segment(
        model,
        mapping,
        "TEST 4 — WIDER GESTURE (14–50)",
        raw[14:51]
    )


if __name__ == "__main__":
    main()