import json
import sys

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from collections import deque

from src.preprocessing.landmark_extractor import LandmarkExtractor
from src.training.landmark_normalizer import LandmarkNormalizer
from src.training.sequence_generator import SequenceGenerator

from src.analysis.landmark_strategy_analysis import (
    apply_recovery_strategy,
)

from src.inference.temporal_stabilizer import (
    TemporalStabilizer,
)


MODEL_PATH = Path(
    "saved_models/isl_lstm.keras"
)

LABEL_MAPPING_PATH = Path(
    "outputs/label_mapping.json"
)

SEQUENCE_LENGTH = 60
PREDICTION_INTERVAL = 3
RECOVERY_STRATEGY = "v1.7.1"


def load_labels():
    """
    Load the label mapping used by the trained model.
    """

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


def hand_detected(landmarks):
    """
    Check whether at least one hand is present.

    The landmark extractor represents missing hand
    landmarks using zeros in the final 126 features.
    """

    hand_landmarks = landmarks[132:258]

    return not np.allclose(
        hand_landmarks,
        0.0,
    )


def predict_sequence(
    model,
    labels,
    raw_sequence,
):
    """
    Apply the same preprocessing pipeline used
    by the realtime application and return the
    predicted label and confidence.
    """

    recovered_sequence = (
        apply_recovery_strategy(
            raw_sequence,
            strategy=RECOVERY_STRATEGY,
        )
    )

    normalizer = LandmarkNormalizer()

    normalized_sequence = (
        normalizer.normalize(
            recovered_sequence.copy()
        )
    )

    generator = SequenceGenerator()

    sequence = generator.generate(
        normalized_sequence
    )

    X = np.expand_dims(
        sequence,
        axis=0,
    )

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
        probabilities[
            predicted_index
        ]
    )

    return (
        predicted_label,
        confidence,
    )


def analyze_predictions(
    predictions,
    title,
):
    """
    Analyze temporal stability of a prediction sequence.

    'Uncertain' is reported separately and does not count
    as a class change.
    """

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    if not predictions:

        print()
        print(
            "No valid predictions were generated."
        )

        return None

    labels_only = [
        item["label"]
        for item in predictions
    ]

    # --------------------------------------------------------
    # Count uncertain outputs
    # --------------------------------------------------------

    uncertain_count = sum(
        label == "Uncertain"
        for label in labels_only
    )

    # --------------------------------------------------------
    # Count class changes
    #
    # Ignore "Uncertain" because it is not a recognized class.
    # --------------------------------------------------------

    recognized_labels = [
        label
        for label in labels_only
        if label != "Uncertain"
    ]

    changes = 0

    for previous, current in zip(
        recognized_labels,
        recognized_labels[1:],
    ):

        if current != previous:
            changes += 1

    # --------------------------------------------------------
    # Stability
    # --------------------------------------------------------

    recognized_predictions = len(
        recognized_labels
    )

    if recognized_predictions <= 1:

        stability = 100.0

    else:

        stable_transitions = (
            recognized_predictions
            - 1
            - changes
        )

        stability = (
            stable_transitions
            / (recognized_predictions - 1)
        ) * 100.0

    # --------------------------------------------------------
    # Confidence
    #
    # Do not include synthetic 0.0 confidence values
    # generated for "Uncertain".
    # --------------------------------------------------------

    recognized_confidences = [
        item["confidence"]
        for item in predictions
        if item["label"] != "Uncertain"
    ]

    if recognized_confidences:

        average_confidence = float(
            np.mean(
                recognized_confidences
            )
        )

    else:

        average_confidence = 0.0

    # --------------------------------------------------------
    # Most common recognized label
    # --------------------------------------------------------

    if recognized_labels:

        most_common_label = max(
            set(recognized_labels),
            key=recognized_labels.count,
        )

    else:

        most_common_label = "None"

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()

    print(
        f"Total predictions   : "
        f"{len(predictions)}"
    )

    print(
        f"Recognized outputs  : "
        f"{recognized_predictions}"
    )

    print(
        f"Uncertain outputs   : "
        f"{uncertain_count}"
    )

    print(
        f"Prediction changes  : "
        f"{changes}"
    )

    print(
        f"Stability           : "
        f"{stability:.2f}%"
    )

    print(
        f"Average confidence  : "
        f"{average_confidence:.4f}"
    )

    print(
        f"Most common label   : "
        f"{most_common_label}"
    )

    return {
        "total_predictions": len(predictions),
        "recognized_outputs": recognized_predictions,
        "uncertain_outputs": uncertain_count,
        "prediction_changes": changes,
        "stability": stability,
        "average_confidence": average_confidence,
        "most_common_label": most_common_label,
    }


def main():

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m src.inference.stability_test "
            "<video_path>"
        )

    video_path = Path(
        sys.argv[1]
    )

    print()
    print("=" * 70)
    print("RAW TEMPORAL STABILITY TEST")
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
    print(video_path)

    video = cv2.VideoCapture(
        str(video_path)
    )

    if not video.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
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

            frames.append(
                landmarks
            )

    finally:

        video.release()
        extractor.close()

    landmarks = np.asarray(
        frames,
        dtype=np.float32,
    )

    print()
    print(
        f"Extracted frames : {landmarks.shape[0]}"
    )

    print(
        f"Landmark shape   : {landmarks.shape}"
    )

    if landmarks.shape[0] < SEQUENCE_LENGTH:
        raise ValueError(
            "Video contains fewer than "
            f"{SEQUENCE_LENGTH} frames."
        )

    # --------------------------------------------------------
    # Sliding-window prediction
    # --------------------------------------------------------

    frame_buffer = deque(
        maxlen=SEQUENCE_LENGTH
    )

    predictions = []

    prediction_number = 0

    for frame_index, frame_landmarks in enumerate(
        landmarks
    ):

        frame_buffer.append(
            frame_landmarks
        )

        if len(frame_buffer) < SEQUENCE_LENGTH:
            continue

        # Match the realtime prediction interval.
        if (
            (frame_index + 1 - SEQUENCE_LENGTH)
            % PREDICTION_INTERVAL
            != 0
        ):
            continue

        # Match realtime missing-hand behavior.
        if not hand_detected(
            frame_landmarks
        ):
            continue

        raw_sequence = np.asarray(
            frame_buffer,
            dtype=np.float32,
        )

        predicted_label, confidence = (
            predict_sequence(
                model,
                labels,
                raw_sequence,
            )
        )

        prediction_number += 1

        predictions.append(
            {
                "frame": frame_index + 1,
                "label": predicted_label,
                "confidence": confidence,
            }
        )

        print(
            f"{prediction_number:03d} | "
            f"Frame {frame_index + 1:03d} | "
            f"{predicted_label:<15} | "
            f"{confidence:.4f}"
        )


    # --------------------------------------------------------
    # RAW stability analysis
    # --------------------------------------------------------

    raw_results = analyze_predictions(
        predictions,
        "RAW TEMPORAL STABILITY",
    )

    # --------------------------------------------------------
    # v1.9 temporal stabilization
    # --------------------------------------------------------

    stabilizer = TemporalStabilizer(
        history_size=5,
        initial_min_votes=2,
        transition_min_votes=3,
    )

    stabilized_predictions = []

    for item in predictions:

        stable_label, stable_confidence = (
            stabilizer.update(
                item["label"],
                item["confidence"],
                hand_detected=True,
            )
        )

        stabilized_predictions.append(
            {
                "frame": item["frame"],
                "label": stable_label,
                "confidence": stable_confidence,
            }
        )

    # --------------------------------------------------------
    # Display stabilized prediction sequence
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V1.9 STABILIZED PREDICTIONS")
    print("=" * 70)

    print()

    for index, item in enumerate(
        stabilized_predictions,
        start=1,
    ):

        print(
            f"{index:03d} | "
            f"Frame {item['frame']:03d} | "
            f"{item['label']:<15} | "
            f"{item['confidence']:.4f}"
        )

    # --------------------------------------------------------
    # v1.9 stability analysis
    # --------------------------------------------------------

    stabilized_results = analyze_predictions(
        stabilized_predictions,
        "V1.9 STABILIZED TEMPORAL STABILITY",
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    if (
        raw_results is not None
        and stabilized_results is not None
    ):

        print()
        print("=" * 70)
        print("RAW vs V1.9 COMPARISON")
        print("=" * 70)

        print()

        print(
            f"Raw prediction changes       : "
            f"{raw_results['prediction_changes']}"
        )

        print(
            f"V1.9 prediction changes      : "
            f"{stabilized_results['prediction_changes']}"
        )

        print()

        print(
            f"Raw stability                : "
            f"{raw_results['stability']:.2f}%"
        )

        print(
            f"V1.9 stability               : "
            f"{stabilized_results['stability']:.2f}%"
        )

        print()

        print(
            f"Raw average confidence       : "
            f"{raw_results['average_confidence']:.4f}"
        )

        print(
            f"V1.9 average confidence      : "
            f"{stabilized_results['average_confidence']:.4f}"
        )

        print()
    

    labels_only = [
        item["label"]
        for item in predictions
    ]

    changes = 0

    for previous, current in zip(
        labels_only,
        labels_only[1:],
    ):

        if current != previous:
            changes += 1

    total_predictions = len(
        predictions
    )

    stable_transitions = (
        total_predictions - 1 - changes
    )

    if total_predictions == 1:
        stability = 100.0

    else:
        stability = (
            stable_transitions
            / (total_predictions - 1)
        ) * 100.0

    average_confidence = float(
        np.mean(
            [
                item["confidence"]
                for item in predictions
            ]
        )
    )

    most_common_label = max(
        set(labels_only),
        key=labels_only.count,
    )



if __name__ == "__main__":
    main()