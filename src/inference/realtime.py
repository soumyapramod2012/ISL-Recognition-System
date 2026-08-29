import cv2
import time
import json
import numpy as np
import tensorflow as tf

from collections import deque
from pathlib import Path

from src.preprocessing.landmark_extractor import LandmarkExtractor
from src.training.landmark_normalizer import LandmarkNormalizer
from src.training.sequence_generator import SequenceGenerator

from src.analysis.landmark_strategy_analysis import (
    apply_recovery_strategy,
)


MODEL_PATH = Path("saved_models/isl_lstm.keras")
LABEL_MAPPING_PATH = Path("outputs/label_mapping.json")

SEQUENCE_LENGTH = 60
CONFIDENCE_THRESHOLD = 0.50
RECOVERY_STRATEGY = "v1.7.1"
PREDICTION_INTERVAL = 3


def load_labels():
    """
    Load the label mapping used by the trained model.
    """

    with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as file:
        mapping = json.load(file)

    # Handle common mapping formats.
    if isinstance(mapping, list):
        return mapping

    if isinstance(mapping, dict):

        # Example:
        # {"0": "61. Summer", "1": "62. Spring", ...}

        if all(str(key).isdigit() for key in mapping.keys()):
            return [
                mapping[str(index)]
                for index in range(len(mapping))
            ]

        # Example:
        # {"61. Summer": 0, "62. Spring": 1, ...}

        if all(isinstance(value, int) for value in mapping.values()):
            labels = [None] * len(mapping)

            for label, index in mapping.items():
                labels[index] = label

            return labels

    raise ValueError(
        "Unsupported label mapping format."
    )


def normalize_sequence(sequence):
    """
    Apply the same landmark normalization used
    by the trained model.
    """

    normalizer = LandmarkNormalizer()

    normalized = normalizer.normalize(
        sequence.copy()
    )

    return normalized


def hand_detected(landmarks):
    """
    Check whether at least one hand was detected.

    The landmark extractor represents missing hand landmarks
    using zeros in the final 126 features.
    """

    hand_landmarks = landmarks[132:258]

    return not np.allclose(
        hand_landmarks,
        0.0,
    )


def main():

    print("=" * 70)
    print("REAL-TIME INDIAN SIGN LANGUAGE RECOGNITION")
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
        f"Classes loaded: {len(labels)}"
    )

    print()
    print("Initializing MediaPipe...")

    extractor = LandmarkExtractor()

    print("MediaPipe initialized.")

    sequence_generator = SequenceGenerator()

    # Store the most recent 60 frames.
    frame_buffer = deque(
        maxlen=SEQUENCE_LENGTH
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam."
        )

    print()
    print("Webcam started.")
    print("Perform an ISL sign in front of the camera.")
    print("Press Q to quit.")

    prediction = "Waiting..."
    confidence = 0.0

    frame_count = 0
    prediction_count = 0
    predictions_this_second = 0
    prediction_frame_counter = 0

    fps = 0.0
    prediction_rate = 0.0
    inference_time_ms = 0.0

    fps_start_time = time.perf_counter()

    try:

        while True:

            success, frame = cap.read()

            frame_count += 1

            if not success:
                print(
                    "Failed to read webcam frame."
                )
                break

            # --------------------------------------------------
            # Extract landmarks
            # --------------------------------------------------

            landmarks = extractor.extract(
                frame
            )

            # Validate expected representation.
            if landmarks.shape != (258,):
                raise ValueError(
                    "Unexpected landmark shape: "
                    f"{landmarks.shape}. "
                    "Expected (258,)."
                )

            has_hand = hand_detected(
                landmarks
            )

            frame_buffer.append(
                landmarks
            )

            if not has_hand:
                prediction = "Uncertain"
                confidence = 0.0

            prediction_frame_counter += 1

            # --------------------------------------------------
            # Predict when enough frames are available
            # --------------------------------------------------

            if(
                len(frame_buffer) == SEQUENCE_LENGTH
                and prediction_frame_counter >= PREDICTION_INTERVAL
                and has_hand
            ):

                raw_sequence = np.asarray(
                    frame_buffer,
                    dtype=np.float32,
                )

                # --------------------------------------------------
                # Landmark recovery
                # --------------------------------------------------

                recovered_sequence = apply_recovery_strategy(
                    raw_sequence,
                    strategy=RECOVERY_STRATEGY,
                )

                # --------------------------------------------------
                # Normalize recovered landmarks
                # --------------------------------------------------

                normalized_sequence = (
                    normalize_sequence(
                        recovered_sequence
                    )
                )

                # --------------------------------------------------
                # Generate the model sequence.
                # --------------------------------------------------

                sequence = (
                    sequence_generator.generate(
                        normalized_sequence
                    )
                )

                # Add batch dimension.
                X = np.expand_dims(
                    sequence,
                    axis=0,
                )

                inference_start = time.perf_counter()

                # Model prediction.
                probabilities = model.predict(
                    X,
                    verbose=0,
                )[0]

                inference_time_ms = (
                    time.perf_counter() - inference_start
                ) * 1000

                prediction_count += 1
                predictions_this_second += 1
                prediction_frame_counter = 0

                predicted_index = int(
                    np.argmax(probabilities)
                )

                confidence = float(
                    probabilities[
                        predicted_index
                    ]
                )

                if (
                    confidence
                    >= CONFIDENCE_THRESHOLD
                ):
                    prediction = labels[
                        predicted_index
                    ]
                else:
                    prediction = (
                        "Uncertain"
                    )


            # --------------------------------------------------
            # Performance metrics
            # --------------------------------------------------

            elapsed = time.perf_counter() - fps_start_time

            if elapsed >= 1.0:
                fps = frame_count / elapsed
                prediction_rate = predictions_this_second / elapsed

                frame_count = 0
                predictions_this_second = 0
                fps_start_time = time.perf_counter()

            # --------------------------------------------------
            # Display
            # --------------------------------------------------

            cv2.rectangle(
                frame,
                (10, 10),
                (630, 215),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                frame,
                f"Prediction: {prediction}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence:.2%}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Recovery: {RECOVERY_STRATEGY}",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Inference: {inference_time_ms:.1f} ms",
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Predictions/sec: {prediction_rate:.1f}",
                (20, 205),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Real-Time ISL Recognition",
                frame,
            )

            # Q = quit
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:

        cap.release()

        cv2.destroyAllWindows()

        extractor.close()

        print()
        print("Application stopped.")


if __name__ == "__main__":
    main()