"""
Uploaded-video prediction for the generalized 114-class ISL model.

This script uses the SAME:
    - MediaPipe legacy landmark extractor
    - 258-feature representation
    - HandLandmarkInterpolator
    - LandmarkNormalizer
    - SequenceGenerator
    - 60-frame sequence length
    - trained 114-class model

It does NOT modify realtime.py or realtime_generalized_114.py.

Run from project root:

    venv_mp_legacy\Scripts\activate

    python -m src.inference.predict_uploaded_video "path\to\video.mp4"
"""

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from src.preprocessing.legacy_landmark_extractor import (
    LegacyLandmarkExtractor,
)

from src.training.hand_landmark_interpolator import (
    HandLandmarkInterpolator,
)

from src.training.landmark_normalizer import (
    LandmarkNormalizer,
)

from src.training.sequence_generator import (
    SequenceGenerator,
)


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

# Number of frames to move between prediction windows.
#
# 1  = maximum overlap
# 15 = moderate overlap
#
# We will start with 15 to keep processing reasonable.
WINDOW_STEP = 15


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names():
    with open(
        LABEL_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        mapping = json.load(f)

    # Mapping is:
    #
    #     label -> index
    #
    # Convert to:
    #
    #     index -> label

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
# PREPROCESSING
# ============================================================

def preprocess_landmarks(frames):
    """
    Apply the SAME preprocessing used by the existing
    generalized realtime inference program.

    258 raw landmarks
        ->
    hand interpolation
        ->
    landmark normalization
        ->
    sequence generation
    """

    data = np.asarray(
        frames,
        dtype=np.float32,
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

    interpolator = HandLandmarkInterpolator()
    normalizer = LandmarkNormalizer()
    generator = SequenceGenerator()

    data = interpolator.interpolate(data)

    data = normalizer.normalize(data)

    data = generator.generate(data)

    return data.astype(np.float32)


# ============================================================
# GENERATE WINDOWS
# ============================================================

def generate_windows(data):
    """
    Generate overlapping 60-frame windows.

    Returns:
        list of arrays with shape (60, 258)
    """

    windows = []

    total_frames = data.shape[0]

    if total_frames < SEQUENCE_LENGTH:
        return windows

    for start in range(
        0,
        total_frames - SEQUENCE_LENGTH + 1,
        WINDOW_STEP,
    ):

        end = start + SEQUENCE_LENGTH

        window = data[start:end]

        if window.shape == (
            SEQUENCE_LENGTH,
            FEATURE_SIZE,
        ):
            windows.append(window)

    return windows


# ============================================================
# PREDICT
# ============================================================

def predict_windows(
    model,
    windows,
    class_names,
):
    """
    Predict all video windows in one batch.
    """

    x = np.asarray(
        windows,
        dtype=np.float32,
    )

    print()
    print(
        f"Prediction input shape: {x.shape}"
    )

    probabilities = model.predict(
        x,
        verbose=0,
    )

    predictions = []

    for window_index, probs in enumerate(
        probabilities,
        start=1,
    ):

        top_indices = np.argsort(
            probs
        )[::-1][:5]

        top5 = []

        for index in top_indices:

            top5.append(
                (
                    class_names[int(index)],
                    float(probs[index]),
                )
            )

        predictions.append(top5)

    return predictions


# ============================================================
# DISPLAY WINDOW RESULTS
# ============================================================

def display_window_results(predictions):

    print()
    print("=" * 80)
    print("WINDOW PREDICTIONS")
    print("=" * 80)

    for window_number, top5 in enumerate(
        predictions,
        start=1,
    ):

        label, confidence = top5[0]

        print(
            f"Window {window_number:3d}: "
            f"{label:<30} "
            f"{confidence * 100:6.2f}%"
        )


# ============================================================
# AGGREGATE RESULTS
# ============================================================

def aggregate_predictions(predictions):

    # Count how many windows selected each class
    # as their TOP-1 prediction.

    top1_labels = [
        top5[0][0]
        for top5 in predictions
    ]

    counts = Counter(top1_labels)

    # Calculate average top-1 confidence
    # for each predicted class.

    confidence_values = {}

    for top5 in predictions:

        label = top5[0][0]
        confidence = top5[0][1]

        confidence_values.setdefault(
            label,
            [],
        )

        confidence_values[label].append(
            confidence
        )

    results = []

    for label, count in counts.items():

        average_confidence = np.mean(
            confidence_values[label]
        )

        results.append(
            (
                label,
                count,
                float(average_confidence),
            )
        )

    # First sort by number of windows.
    #
    # Then by average confidence.

    results.sort(
        key=lambda item: (
            item[1],
            item[2],
        ),
        reverse=True,
    )

    return results


# ============================================================
# VIDEO PROCESSING
# ============================================================

def extract_video_landmarks(video_path):

    print()
    print("=" * 80)
    print("VIDEO PROCESSING")
    print("=" * 80)

    print(
        f"Video: {video_path}"
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    print(
        f"FPS          : {fps:.2f}"
    )

    print(
        f"Total frames : {total_frames}"
    )

    extractor = LegacyLandmarkExtractor()

    landmarks = []

    frame_number = 0

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            vector = extractor.extract(
                frame
            )

            landmarks.append(vector)

            if (
                frame_number % 100 == 0
                or frame_number == total_frames
            ):

                print(
                    f"Frames processed: "
                    f"{frame_number}/{total_frames}"
                )

    finally:

        cap.release()
        extractor.close()

    print()
    print(
        f"Extracted landmark frames: "
        f"{len(landmarks)}"
    )

    if not landmarks:
        raise RuntimeError(
            "No landmarks were extracted."
        )

    landmarks = np.asarray(
        landmarks,
        dtype=np.float32,
    )

    print(
        f"Raw landmark shape: "
        f"{landmarks.shape}"
    )

    return landmarks


# ============================================================
# MAIN
# ============================================================

def predict_video(video_path):

    video_path = Path(
        video_path
    )

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found:\n{video_path}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not LABEL_PATH.exists():
        raise FileNotFoundError(
            f"Label mapping not found:\n{LABEL_PATH}"
        )

    print()
    print("=" * 80)
    print("INDIAN SIGN LANGUAGE")
    print("GENERALIZED 114-CLASS VIDEO PREDICTION")
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

    landmarks = extract_video_landmarks(
        video_path
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
        f"Prediction windows: "
        f"{len(windows)}"
    )

    if not windows:

        print()
        print(
            f"ERROR: Video does not contain "
            f"enough frames for a "
            f"{SEQUENCE_LENGTH}-frame sequence."
        )

        return

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = predict_windows(
        model,
        windows,
        class_names,
    )

    # --------------------------------------------------------
    # Display individual windows
    # --------------------------------------------------------

    display_window_results(
        predictions
    )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    aggregated = aggregate_predictions(
        predictions
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL AGGREGATED RESULT")
    print("=" * 80)

    print(
        f"{'Rank':<6}"
        f"{'Class':<32}"
        f"{'Windows':<10}"
        f"{'Avg Confidence':<18}"
    )

    print("-" * 80)

    for rank, (
        label,
        count,
        confidence,
    ) in enumerate(
        aggregated[:5],
        start=1,
    ):

        print(
            f"{rank:<6}"
            f"{label:<32}"
            f"{count:<10}"
            f"{confidence * 100:>8.2f}%"
        )

    print()
    print("=" * 80)

    final_label = aggregated[0][0]
    final_count = aggregated[0][1]
    final_confidence = aggregated[0][2]

    print(
        f"FINAL PREDICTION : {final_label}"
    )

    print(
        f"WINDOW AGREEMENT : "
        f"{final_count}/{len(predictions)}"
    )

    print(
        f"AVERAGE CONFIDENCE: "
        f"{final_confidence * 100:.2f}%"
    )

    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print()
        print("Usage:")
        print(
            'python -m src.inference.predict_uploaded_video '
            '"path\\to\\video.mp4"'
        )
        print()

        sys.exit(1)

    predict_video(
        sys.argv[1]
    )