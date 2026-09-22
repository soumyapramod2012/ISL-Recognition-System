from pathlib import Path

import numpy as np
import tensorflow as tf

from src.training.hand_landmark_interpolator import HandLandmarkInterpolator
from src.training.landmark_normalizer import LandmarkNormalizer
from src.training.sequence_generator import SequenceGenerator


MODEL_PATH = Path("saved_models/isl_lstm.keras")

WINTER_DIR = Path(
    "dataset/processed/landmarks/63. Winter"
)

LABELS = [
    "61. Summer",
    "62. Spring",
    "63. Winter",
    "64. Fall",
    "65. Season",
    "Ex. Monsoon",
]


def preprocess(landmarks):

    interpolator = HandLandmarkInterpolator(
        max_gap=5
    )

    landmarks = interpolator.interpolate(
        landmarks
    )

    normalizer = LandmarkNormalizer()

    landmarks = normalizer.normalize(
        landmarks
    )

    generator = SequenceGenerator()

    sequence = generator.generate(
        landmarks
    )

    return sequence.astype(np.float32)


def main():

    print("=" * 75)
    print("WINTER CLASS ANALYSIS")
    print("=" * 75)

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    files = sorted(
        WINTER_DIR.glob("*.npy")
    )

    print(
        f"Winter samples found : {len(files)}"
    )

    results = []

    print("\n" + "=" * 75)
    print("INDIVIDUAL WINTER SAMPLES")
    print("=" * 75)

    for path in files:

        landmarks = np.load(path)

        sequence = preprocess(landmarks)

        prediction = model.predict(
            np.expand_dims(sequence, axis=0),
            verbose=0,
        )[0]

        predicted_index = int(
            np.argmax(prediction)
        )

        predicted_label = LABELS[
            predicted_index
        ]

        winter_probability = float(
            prediction[2]
        )

        confidence = float(
            prediction[predicted_index]
        )

        results.append(
            {
                "file": path.name,
                "frames": len(landmarks),
                "predicted": predicted_label,
                "confidence": confidence,
                "winter_probability": winter_probability,
            }
        )

        print(
            f"{path.name:<15} | "
            f"Frames: {len(landmarks):>3} | "
            f"Predicted: {predicted_label:<18} | "
            f"Confidence: {confidence:.4f} | "
            f"Winter prob: {winter_probability:.4f}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)

    winter_correct = sum(
        r["predicted"] == "63. Winter"
        for r in results
    )

    print(
        f"\nCorrectly predicted Winter : "
        f"{winter_correct}/{len(results)}"
    )

    average_winter_probability = np.mean(
        [
            r["winter_probability"]
            for r in results
        ]
    )

    print(
        f"Average Winter probability : "
        f"{average_winter_probability:.4f}"
    )

    print("\nPrediction distribution:")

    distribution = {}

    for result in results:

        label = result["predicted"]

        distribution[label] = (
            distribution.get(label, 0) + 1
        )

    for label, count in distribution.items():

        print(
            f"    {label:<18} : {count}"
        )

    # --------------------------------------------------------
    # Weakest Winter samples
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("LOWEST WINTER PROBABILITIES")
    print("=" * 75)

    weakest = sorted(
        results,
        key=lambda r: r["winter_probability"]
    )

    for result in weakest:

        print(
            f"{result['file']:<15} | "
            f"Frames: {result['frames']:>3} | "
            f"Winter probability: "
            f"{result['winter_probability']:.4f} | "
            f"Predicted: "
            f"{result['predicted']}"
        )

    print("\n" + "=" * 75)
    print("ANALYSIS COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()