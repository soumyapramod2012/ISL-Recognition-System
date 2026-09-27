"""
Experimental multi-window uploaded-video inference.

IMPORTANT:
- Does NOT modify realtime.py.
- Does NOT modify the trained model.
- Uses the same 258-feature legacy MediaPipe representation.
- Uses interpolation + normalization before prediction.
"""

from pathlib import Path
import json
import sys

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "isl_lstm_generalized_114_legacy.keras"
)

LABEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "generalized_114_label_mapping.json"
)

SEQUENCE_LENGTH = 60
FEATURE_SIZE = 258

# Experimental temporal-span sampling
SOURCE_SPAN = 75
SPAN_STEP = 15

# Number of frames between consecutive windows.
#WINDOW_STEP = 15


# ============================================================
# LABELS
# ============================================================

def load_class_names():
    with open(
        LABEL_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        mapping = json.load(f)

    class_names = [
        label
        for label, index in sorted(
            mapping.items(),
            key=lambda item: item[1],
        )
    ]

    if len(class_names) != 114:
        raise ValueError(
            f"Expected 114 classes, found {len(class_names)}"
        )

    return class_names


# ============================================================
# LANDMARK EXTRACTION
# ============================================================

def extract_video_landmarks(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    extractor = LegacyLandmarkExtractor()

    frames = []

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            landmarks = extractor.extract(frame)

            if landmarks.shape != (FEATURE_SIZE,):
                raise ValueError(
                    f"Unexpected landmark shape: "
                    f"{landmarks.shape}"
                )

            frames.append(landmarks)

    finally:
        cap.release()
        extractor.close()

    if not frames:
        raise RuntimeError(
            "No frames were extracted."
        )

    return np.asarray(
        frames,
        dtype=np.float32,
    )


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_landmarks(landmarks):
    if landmarks.ndim != 2:
        raise ValueError(
            f"Expected 2D landmarks, got {landmarks.shape}"
        )

    if landmarks.shape[1] != FEATURE_SIZE:
        raise ValueError(
            f"Expected {FEATURE_SIZE} features, "
            f"got {landmarks.shape[1]}"
        )

    interpolator = HandLandmarkInterpolator(
        max_gap=5
    )

    normalizer = LandmarkNormalizer()

    landmarks = interpolator.interpolate(
        landmarks
    )

    landmarks = normalizer.normalize(
        landmarks
    )

    return landmarks.astype(np.float32)


# ============================================================
# WINDOW GENERATION
# ============================================================

def generate_windows(landmarks):
    """
    Experimental temporal-span sampling.

    Each source span contains 75 frames.
    Each 75-frame span is converted to the model's
    required 60-frame sequence using SequenceGenerator.
    """

    total_frames = len(landmarks)

    if total_frames < SEQUENCE_LENGTH:
        raise ValueError(
            f"Video has only {total_frames} frames. "
            f"At least {SEQUENCE_LENGTH} are required."
        )

    generator = SequenceGenerator()

    windows = []

    # If the video is shorter than the experimental
    # source span, use the existing sequence generator.
    if total_frames < SOURCE_SPAN:
        sequence = generator.generate(landmarks)

        if sequence.shape == (
            SEQUENCE_LENGTH,
            FEATURE_SIZE,
        ):
            windows.append(sequence)

        return np.asarray(
            windows,
            dtype=np.float32,
        )

    # Generate overlapping 75-frame source spans.
    starts = list(
        range(
            0,
            total_frames - SOURCE_SPAN + 1,
            SPAN_STEP,
        )
    )

    # Always include the final part of the video.
    final_start = total_frames - SOURCE_SPAN

    if starts[-1] != final_start:
        starts.append(final_start)

    for start in starts:

        end = start + SOURCE_SPAN

        span = landmarks[start:end]

        # Convert the 75-frame source span into
        # the model's required 60-frame sequence.
        sequence = generator.generate(span)

        if sequence.shape == (
            SEQUENCE_LENGTH,
            FEATURE_SIZE,
        ):
            windows.append(sequence)

    return np.asarray(
        windows,
        dtype=np.float32,
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_windows(
    model,
    windows,
    class_names,
):
    probabilities = model.predict(
        windows,
        verbose=0,
    )

    predictions = []

    for index, probs in enumerate(
        probabilities,
        start=1,
    ):
        top_indices = np.argsort(
            probs
        )[::-1][:5]

        top5 = [
            (
                class_names[i],
                float(probs[i]),
            )
            for i in top_indices
        ]

        predictions.append(top5)

    return predictions


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_predictions(
    predictions,
):
    class_data = {}

    for top5 in predictions:

        label, confidence = top5[0]

        if label not in class_data:
            class_data[label] = {
                "windows": 0,
                "confidence": [],
            }

        class_data[label]["windows"] += 1
        class_data[label]["confidence"].append(
            confidence
        )

    results = []

    for label, data in class_data.items():

        average_confidence = float(
            np.mean(
                data["confidence"]
            )
        )

        results.append(
            (
                label,
                data["windows"],
                average_confidence,
            )
        )

    results.sort(
        key=lambda x: (
            x[1],
            x[2],
        ),
        reverse=True,
    )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage:\n"
            "python -m src.inference.predict_uploaded_video_windows "
            "<video_path>"
        )

    video_path = Path(
        sys.argv[1]
    )

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    print()
    print("=" * 80)
    print(
        "INDIAN SIGN LANGUAGE"
    )
    print(
        "GENERALIZED 114-CLASS "
        "MULTI-WINDOW VIDEO PREDICTION"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading trained model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Load labels
    # --------------------------------------------------------

    class_names = load_class_names()

    print(
        f"Classes loaded: {len(class_names)}"
    )

    # --------------------------------------------------------
    # Extract landmarks
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("VIDEO PROCESSING")
    print("=" * 80)

    print(
        f"Video: {video_path}"
    )

    landmarks = extract_video_landmarks(
        video_path
    )

    print(
        f"Extracted landmark frames: "
        f"{len(landmarks)}"
    )

    print(
        f"Raw landmark shape: "
        f"{landmarks.shape}"
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("PREPROCESSING")
    print("=" * 80)

    processed = preprocess_landmarks(
        landmarks
    )

    print(
        f"Preprocessed shape: "
        f"{processed.shape}"
    )

    # --------------------------------------------------------
    # Generate windows
    # --------------------------------------------------------

    windows = generate_windows(
        processed
    )

    print(
        f"Window length: "
        f"{SEQUENCE_LENGTH}"
    )

    print(
        f"Source span: "
        f"{SOURCE_SPAN}"
    )

    print(
        f"Span step: "
        f"{SPAN_STEP}"
    )
    
    print(
        f"Prediction windows: "
        f"{len(windows)}"
    )

    print(
        f"Prediction input shape: "
        f"{windows.shape}"
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("WINDOW PREDICTIONS")
    print("=" * 80)

    predictions = predict_windows(
        model,
        windows,
        class_names,
    )

    for number, top5 in enumerate(
        predictions,
        start=1,
    ):

        label, confidence = top5[0]

        print(
            f"Window {number:3d}: "
            f"{label:30s} "
            f"{confidence * 100:6.2f}%"
        )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    results = aggregate_predictions(
        predictions
    )

    print()
    print("=" * 80)
    print("FINAL AGGREGATED RESULT")
    print("=" * 80)

    print(
        f"{'Rank':<6}"
        f"{'Class':<30}"
        f"{'Windows':<10}"
        f"{'Avg Confidence':<18}"
    )

    print("-" * 80)

    for rank, (
        label,
        count,
        confidence,
    ) in enumerate(
        results[:5],
        start=1,
    ):

        print(
            f"{rank:<6}"
            f"{label:<30}"
            f"{count:<10}"
            f"{confidence * 100:6.2f}%"
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    final_label = results[0][0]

    total_windows = len(predictions)

    agreeing_windows = results[0][1]

    final_confidence = results[0][2]

    print()
    print("=" * 80)
    print(
        f"FINAL PREDICTION : "
        f"{final_label}"
    )

    print(
        f"WINDOW AGREEMENT : "
        f"{agreeing_windows}/{total_windows}"
    )

    print(
        f"AVERAGE CONFIDENCE: "
        f"{final_confidence * 100:.2f}%"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
