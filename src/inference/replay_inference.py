import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.analysis.landmark_strategy_analysis import (
    apply_recovery_strategy,
)

from src.training.landmark_normalizer import LandmarkNormalizer
from src.training.sequence_generator import SequenceGenerator


MODEL_PATH = Path("saved_models/isl_lstm.keras")

LABEL_MAPPING_PATH = Path(
    "outputs/label_mapping.json"
)

SAMPLE_PATH = Path(
    "dataset/processed/landmarks/"
    "63. Winter/MVI_5432.npy"
)

EXPECTED_LABEL = "63. Winter"

SEQUENCE_LENGTH = 60


def load_labels():

    with open(
        LABEL_MAPPING_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        mapping = json.load(file)

    if isinstance(mapping, list):
        return mapping

    if isinstance(mapping, dict):

        if all(
            str(key).isdigit()
            for key in mapping.keys()
        ):

            return [
                mapping[str(index)]
                for index in range(
                    len(mapping)
                )
            ]

        if all(
            isinstance(value, int)
            for value in mapping.values()
        ):

            labels = [
                None
            ] * len(mapping)

            for label, index in mapping.items():
                labels[index] = label

            return labels

    raise ValueError(
        "Unsupported label mapping format."
    )


def main():

    print()
    print("=" * 70)
    print("REAL-TIME PIPELINE REPLAY TEST")
    print("=" * 70)

    print()
    print("Loading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded.")

    print()
    print("Loading labels...")

    labels = load_labels()

    print(
        f"Classes: {len(labels)}"
    )

    print()
    print("Loading landmark sequence:")

    print(SAMPLE_PATH)

    landmarks = np.load(
        SAMPLE_PATH
    ).astype(
        np.float32
    )

    print()
    print(
        f"Input shape : {landmarks.shape}"
    )

    print(
        f"Input dtype : {landmarks.dtype}"
    )

    # --------------------------------------------------------
    # Validate landmark representation
    # --------------------------------------------------------

    if landmarks.ndim != 2:
        raise ValueError(
            "Expected 2-dimensional landmark array."
        )

    if landmarks.shape[1] != 258:
        raise ValueError(
            "Expected 258 features per frame."
        )

    if landmarks.shape[0] < SEQUENCE_LENGTH:
        raise ValueError(
            "Sample contains fewer than "
            f"{SEQUENCE_LENGTH} frames."
        )

    # --------------------------------------------------------
    # Test all landmark recovery strategies
    # --------------------------------------------------------

    strategies = [
        "none",
        "linear",
        "v1.7",
        "v1.7.1",
    ]

    print()
    print("=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)

    for strategy in strategies:

        print()
        print("-" * 70)
        print(f"Strategy : {strategy}")
        print("-" * 70)

        # Apply recovery strategy first
        recovered = apply_recovery_strategy(
            landmarks,
            strategy=strategy,
        )

        # Use first 60 frames
        sequence = recovered[
            :SEQUENCE_LENGTH
        ]

        print(
            f"Frames used : {sequence.shape[0]}"
        )

        # Normalize
        normalizer = LandmarkNormalizer()

        normalized = normalizer.normalize(
            sequence.copy()
        )

        # Generate model sequence
        generator = SequenceGenerator()

        model_sequence = generator.generate(
            normalized
        )

        X = np.expand_dims(
            model_sequence,
            axis=0,
        )

        print(
            f"Model input : {X.shape}"
        )

        # Prediction
        probabilities = model.predict(
            X,
            verbose=0,
        )[0]

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_label = labels[
            predicted_index
        ]

        confidence = float(
            probabilities[predicted_index]
        )

        # --------------------------------------------------------
        # Result
        # --------------------------------------------------------
        
        result = (
            "CORRECT"
            if predicted_label == EXPECTED_LABEL
            else "INCORRECT"
        )

        print(
            f"{strategy:<10} → "
            f"{predicted_label:<15} "
            f"{confidence:.4f} "
            f"{result}"
        )

        print(
            f"Predicted  : {predicted_label}"
        )

        print(
            f"Confidence : {confidence:.4f}"
        )

        print(
            f"Result     : {result}"
        )

    


if __name__ == "__main__":
    main()