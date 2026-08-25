import json
from pathlib import Path
import sys

import cv2
import numpy as np
import tensorflow as tf

from src.preprocessing.landmark_extractor import LandmarkExtractor
from src.training.landmark_normalizer import LandmarkNormalizer
from src.training.sequence_generator import SequenceGenerator
from src.analysis.landmark_strategy_analysis import (
    apply_recovery_strategy,
)


MODEL_PATH = Path(
    "saved_models/isl_lstm.keras"
)

LABEL_MAPPING_PATH = Path(
    "outputs/label_mapping.json"
)

if len(sys.argv) != 3:
    raise SystemExit(
        "Usage: python -m src.inference.test_video "
        "<video_path> <expected_label>"
    )

VIDEO_PATH = Path(sys.argv[1])
EXPECTED_LABEL = sys.argv[2]

SEQUENCE_LENGTH = 60
RECOVERY_STRATEGY = "v1.7.1"


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


def main():

    print()
    print("=" * 70)
    print("RECORDED VIDEO INFERENCE TEST")
    print("=" * 70)

    print()
    print("Loading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded.")

    labels = load_labels()

    print(
        f"Classes: {len(labels)}"
    )

    print()
    print("Video:")
    print(VIDEO_PATH)

    video = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not video.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    extractor = LandmarkExtractor()

    frames = []

    try:

        while True:

            success, frame = video.read()

            if not success:
                break

            landmarks = extractor.extract(
                frame
            )

            if landmarks.shape != (258,):
                raise ValueError(
                    "Unexpected landmark shape: "
                    f"{landmarks.shape}"
                )

            frames.append(landmarks)

    finally:

        video.release()
        extractor.close()

    landmarks = np.asarray(
        frames,
        dtype=np.float32,
    )

    print()
    print(
        f"Extracted frames : "
        f"{landmarks.shape[0]}"
    )

    print(
        f"Landmark shape   : "
        f"{landmarks.shape}"
    )

    if landmarks.shape[0] < SEQUENCE_LENGTH:
        raise ValueError(
            "Video contains fewer than "
            f"{SEQUENCE_LENGTH} usable frames."
        )

    # --------------------------------------------------------
    # Use first 60 frames
    # --------------------------------------------------------

    raw_sequence = landmarks[
        :SEQUENCE_LENGTH
    ]

    # --------------------------------------------------------
    # Apply v1.7.1 recovery
    # --------------------------------------------------------

    recovered_sequence = (
        apply_recovery_strategy(
            raw_sequence,
            strategy=RECOVERY_STRATEGY,
        )
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    normalizer = LandmarkNormalizer()

    normalized_sequence = (
        normalizer.normalize(
            recovered_sequence.copy()
        )
    )

    # --------------------------------------------------------
    # Generate model sequence
    # --------------------------------------------------------

    generator = SequenceGenerator()

    sequence = generator.generate(
        normalized_sequence
    )

    X = np.expand_dims(
        sequence,
        axis=0,
    )

    print(
        f"Model input      : {X.shape}"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

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

    result = (
        "CORRECT"
        if predicted_label == EXPECTED_LABEL
        else "INCORRECT"
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VIDEO INFERENCE RESULT")
    print("=" * 70)

    print()
    print(
        f"Expected   : {EXPECTED_LABEL}"
    )

    print(
        f"Predicted  : {predicted_label}"
    )

    print(
        f"Confidence : {confidence:.4f}"
    )

    print(
        f"Recovery   : {RECOVERY_STRATEGY}"
    )

    print(
        f"Result     : {result}"
    )

    print()


if __name__ == "__main__":
    main()