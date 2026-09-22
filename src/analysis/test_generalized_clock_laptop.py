from pathlib import Path
import json

import numpy as np
import tensorflow as tf

from src.training.hand_landmark_interpolator import HandLandmarkInterpolator
from src.training.landmark_normalizer import LandmarkNormalizer
from src.training.sequence_generator import SequenceGenerator


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path(
    "saved_models/isl_lstm_generalized_114_legacy.keras"
)

LABEL_PATH = Path(
    "outputs/generalized_114_label_mapping.json"
)

DATASET_DIR = Path(
    "dataset/processed/generalized_landmarks_legacy"
)


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(path):
    data = np.load(path)

    if data.ndim != 2 or data.shape[1] != 258:
        raise ValueError(
            f"Invalid shape {data.shape}: {path}"
        )

    data = HandLandmarkInterpolator().interpolate(data)
    data = LandmarkNormalizer().normalize(data)
    data = SequenceGenerator().generate(data)

    return data.astype(np.float32)


# ============================================================
# FIND ALL MATCHING FILES
# ============================================================

def find_all(filename):
    """
    Find every generalized landmark file corresponding
    to the original filename.

    Examples:

        existing24__MVI_4959.npy
        include__MVI_4959.npy
    """

    matches = []

    for path in DATASET_DIR.rglob("*.npy"):

        if (
            path.name == filename
            or path.name.endswith("__" + filename)
        ):
            matches.append(path)

    return sorted(matches)


# ============================================================
# PREDICTION
# ============================================================

def predict_file(model, path):

    sequence = preprocess(path)

    probs = model.predict(
        sequence[np.newaxis, ...],
        verbose=0
    )[0]

    order = np.argsort(probs)[::-1]

    return [
        (
            encoder.classes[int(i)],
            float(probs[int(i)])
        )
        for i in order[:5]
    ]


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(expected, path, results):

    print("\n" + "-" * 70)

    print("File     :", path.name)
    print("Source   :", path.parent.name)
    print("Expected :", expected)

    print(
        "Top-1    :",
        results[0][0]
    )

    print(
        "Conf.    :",
        f"{results[0][1] * 100:.2f}%"
    )

    print("Top-5    :")

    for rank, (label, conf) in enumerate(results, 1):

        print(
            f"  {rank}. "
            f"{label:<30} "
            f"{conf * 100:6.2f}%"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "GENERALIZED 114-CLASS MODEL - "
        "DIRECT CLOCK/LAPTOP TEST"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Load label mapping
    # --------------------------------------------------------

    with open(
        LABEL_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        mapping = json.load(f)

    # Saved mapping is:
    #
    # label -> index
    #
    # Reconstruct:
    #
    # index -> label

    class_names = [
        label
        for label, index
        in sorted(
            mapping.items(),
            key=lambda item: item[1]
        )
    ]

    global encoder

    encoder = type(
        "Encoder",
        (),
        {"classes": class_names}
    )()

    print("Model loaded.")
    print("Classes:", len(encoder.classes))

    # ========================================================
    # CLOCK TEST
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "CLOCK - PREVIOUSLY PROBLEMATIC VIDEOS"
    )
    print("=" * 70)

    clock_files = [
        "MVI_4959.npy",
        "MVI_4960.npy"
    ]

    for filename in clock_files:

        matches = find_all(filename)

        if not matches:

            print(
                f"\nNOT FOUND: {filename}"
            )

            continue

        for path in matches:

            results = predict_file(
                model,
                path
            )

            print_result(
                "51. Clock",
                path,
                results
            )

    # ========================================================
    # LAPTOP TEST
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "LAPTOP - ALL AVAILABLE LANDMARK SAMPLES"
    )
    print("=" * 70)

    laptop_dir = DATASET_DIR / "56. Laptop"

    files = sorted(
        laptop_dir.glob("*.npy")
    )

    correct = 0

    for path in files:

        results = predict_file(
            model,
            path
        )

        label, conf = results[0]

        ok = (
            label == "56. Laptop"
        )

        if ok:
            correct += 1

        print(
            f"{path.name:<35} "
            f"-> {label:<25} "
            f"{conf * 100:6.2f}% "
            f"{'OK' if ok else 'WRONG'}"
        )

    print("\nLaptop summary:")

    print(
        "  Samples :",
        len(files)
    )

    print(
        "  Correct :",
        correct
    )

    if files:

        print(
            "  Accuracy:",
            f"{100 * correct / len(files):.2f}%"
        )

    else:

        print(
            "  Accuracy: No samples found"
        )

    print(
        "\nDIRECT TEST COMPLETE"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()