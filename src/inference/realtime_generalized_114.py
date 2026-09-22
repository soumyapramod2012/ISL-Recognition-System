"""
Experimental real-time inference for the generalized 114-class ISL model.

IMPORTANT:
- This file is separate from src/inference/realtime.py.
- Do NOT modify the existing realtime.py.
- Uses the same legacy MediaPipe + 258-feature preprocessing used for training.
"""

from collections import deque
from pathlib import Path
import json

import cv2
import numpy as np
import tensorflow as tf

from src.preprocessing.legacy_landmark_extractor import LegacyLandmarkExtractor
from src.training.hand_landmark_interpolator import HandLandmarkInterpolator
from src.training.landmark_normalizer import LandmarkNormalizer
from src.training.sequence_generator import SequenceGenerator


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    "saved_models/isl_lstm_generalized_114_legacy.keras"
)

LABEL_PATH = Path(
    "outputs/generalized_114_label_mapping.json"
)

SEQUENCE_LENGTH = 60
FEATURE_SIZE = 258

CAMERA_INDEX = 0

# Display threshold only.
# It does NOT change the trained model.
CONFIDENCE_THRESHOLD = 0.60

# Predict every N frames.
# 1 = prediction on every frame.
PREDICT_EVERY_N_FRAMES = 1


# ============================================================
# LOAD LABELS
# ============================================================

def load_class_names():
    with open(
        LABEL_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        mapping = json.load(f)

    # Saved mapping:
    # label -> index
    #
    # Reconstruct:
    # index -> label
    class_names = [
        label
        for label, index
        in sorted(
            mapping.items(),
            key=lambda item: item[1]
        )
    ]

    if len(class_names) != 114:
        raise ValueError(
            f"Expected 114 classes, found {len(class_names)}"
        )

    return class_names


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_sequence(frames):
    """
    Apply exactly the same preprocessing used during training:

        258 raw landmarks
            -> hand interpolation
            -> landmark normalization
            -> 60-frame sequence
    """

    data = np.asarray(
        frames,
        dtype=np.float32
    )

    if data.ndim != 2:
        raise ValueError(
            f"Expected 2D landmarks, got {data.shape}"
        )

    if data.shape[1] != FEATURE_SIZE:
        raise ValueError(
            f"Expected {FEATURE_SIZE} features, "
            f"got {data.shape[1]}"
        )

    data = HandLandmarkInterpolator().interpolate(data)

    data = LandmarkNormalizer().normalize(data)

    data = SequenceGenerator().generate(data)

    if data.shape != (
        SEQUENCE_LENGTH,
        FEATURE_SIZE
    ):
        raise ValueError(
            f"Unexpected sequence shape: {data.shape}"
        )

    return data.astype(np.float32)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GENERALIZED ISL - EXPERIMENTAL REAL-TIME TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded.")

    # --------------------------------------------------------
    # Load labels
    # --------------------------------------------------------

    class_names = load_class_names()

    print(
        f"Classes loaded: {len(class_names)}"
    )

    # --------------------------------------------------------
    # MediaPipe legacy extractor
    # --------------------------------------------------------

    print("\nInitializing MediaPipe Holistic...")

    extractor = LegacyLandmarkExtractor()

    # --------------------------------------------------------
    # Webcam
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not cap.isOpened():
        extractor.close()

        raise RuntimeError(
            f"Could not open webcam "
            f"(camera index {CAMERA_INDEX})"
        )

    print("Webcam opened.")

    # --------------------------------------------------------
    # Frame buffer
    # --------------------------------------------------------

    frame_buffer = deque(
        maxlen=SEQUENCE_LENGTH
    )

    frame_count = 0

    current_label = "Waiting..."
    current_confidence = 0.0

    print("\n" + "=" * 70)
    print("CONTROLS")
    print("=" * 70)
    print("Perform a sign in front of the camera.")
    print("Press Q to quit.")
    print("=" * 70)

    try:

        while True:

            success, frame = cap.read()

            if not success:
                print(
                    "ERROR: Could not read webcam frame."
                )
                break

            frame_count += 1

            # ------------------------------------------------
            # Extract 258 landmarks
            # ------------------------------------------------

            try:
                landmarks = extractor.extract(frame)
            except Exception as exc:
                print(
                    f"Landmark extraction error: {exc}"
                )
                continue

            frame_buffer.append(landmarks)

            # ------------------------------------------------
            # Predict only after 60 frames
            # ------------------------------------------------

            if (
                len(frame_buffer) == SEQUENCE_LENGTH
                and
                frame_count % PREDICT_EVERY_N_FRAMES == 0
            ):

                try:

                    sequence = preprocess_sequence(
                        np.asarray(
                            frame_buffer,
                            dtype=np.float32
                        )
                    )

                    probabilities = model.predict(
                        sequence[np.newaxis, ...],
                        verbose=0
                    )[0]

                    predicted_index = int(
                        np.argmax(probabilities)
                    )

                    confidence = float(
                        probabilities[predicted_index]
                    )

                    predicted_label = class_names[
                        predicted_index
                    ]

                    current_confidence = confidence

                    if confidence >= CONFIDENCE_THRESHOLD:

                        current_label = predicted_label

                    else:

                        current_label = (
                            "Low confidence"
                        )

                except Exception as exc:

                    current_label = "Prediction error"
                    current_confidence = 0.0

                    print(
                        f"Prediction error: {exc}"
                    )

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            display = frame.copy()

            # Header
            cv2.rectangle(
                display,
                (0, 0),
                (display.shape[1], 110),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                display,
                "GENERALIZED ISL - 114 CLASS MODEL",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                display,
                f"Prediction: {current_label}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                display,
                f"Confidence: "
                f"{current_confidence * 100:.1f}%",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Buffer status
            buffer_text = (
                f"Frames: "
                f"{len(frame_buffer)}/"
                f"{SEQUENCE_LENGTH}"
            )

            cv2.putText(
                display,
                buffer_text,
                (20, display.shape[0] - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                display,
                "Press Q to quit",
                (20, display.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            cv2.imshow(
                "Generalized ISL - Experimental",
                display
            )

            # ------------------------------------------------
            # Quit
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:

        cap.release()

        cv2.destroyAllWindows()

        extractor.close()

        print("\nWebcam test stopped.")


if __name__ == "__main__":
    main()
