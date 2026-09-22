import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.training.landmark_normalizer import LandmarkNormalizer
from src.training.sequence_generator import SequenceGenerator
from src.analysis.landmark_strategy_analysis import (
    apply_recovery_strategy,
)


MODEL_PATH = Path("saved_models/isl_lstm.keras")
LABEL_MAPPING_PATH = Path("outputs/label_mapping.json")


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
                for index in range(len(mapping))
            ]

        if all(
            isinstance(value, int)
            for value in mapping.values()
        ):
            labels = [None] * len(mapping)

            for label, index in mapping.items():
                labels[index] = label

            return labels

    raise ValueError(
        "Unsupported label mapping format."
    )


def predict_file(model, labels, path):

    landmarks = np.load(path).astype(
        np.float32
    )

    recovered = apply_recovery_strategy(
        landmarks,
        strategy="v1.7.1",
    )

    normalizer = LandmarkNormalizer()

    normalized = normalizer.normalize(
        recovered.copy()
    )

    generator = SequenceGenerator()

    sequence = generator.generate(
        normalized
    )

    X = np.expand_dims(
        sequence,
        axis=0,
    )

    probabilities = model.predict(
        X,
        verbose=0,
    )[0]

    print()
    print("=" * 60)
    print(path.name)
    print("=" * 60)

    for index in np.argsort(
        probabilities
    )[::-1]:

        print(
            f"{labels[index]:20s} "
            f"{probabilities[index]:.4f}"
        )


def main():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    labels = load_labels()

    files = [
        Path(
            "dataset/processed/landmarks/"
            "63. Winter/MVI_5432.npy"
        ),
        Path(
            "dataset/processed/landmarks/"
            "63. Winter/MVI_5431.npy"
        ),
        Path(
            "dataset/processed/landmarks/"
            "62. Spring/MVI_5425.npy"
        ),
    ]

    for path in files:
        predict_file(
            model,
            labels,
            path,
        )


if __name__ == "__main__":
    main()