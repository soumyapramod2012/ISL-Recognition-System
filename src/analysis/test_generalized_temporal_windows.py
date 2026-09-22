"""
Offline temporal-window experiment for the generalized 114-class ISL model.

Purpose:
    Determine whether the real-time Clock failure is caused by the
    difference between training-time temporal sampling and webcam
    consecutive-frame buffering.

This script DOES NOT:
    - retrain the model
    - modify the dataset
    - modify src/inference/realtime.py
    - modify the generalized model

It compares:
    1. Training-style uniform 60-frame sampling
    2. First consecutive 60 frames
    3. Middle consecutive 60 frames
    4. Last consecutive 60 frames
    5. Sliding consecutive 60-frame windows

Expected Clock files:
    existing24__MVI_4959.npy
    include__MVI_4959.npy
    existing24__MVI_4960.npy
    include__MVI_4960.npy
"""

from pathlib import Path
import json

import numpy as np
import tensorflow as tf

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

DATASET_DIR = Path(
    "dataset/processed/generalized_landmarks_legacy"
)

SEQUENCE_LENGTH = 60

# Step between sliding windows.
# 10 means windows:
# 0-59, 10-69, 20-79, ...
SLIDING_STEP = 10


# ============================================================
# LABELS
# ============================================================

def load_class_names():

    with open(
        LABEL_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        mapping = json.load(f)

    return [
        label
        for label, index
        in sorted(
            mapping.items(),
            key=lambda item: item[1]
        )
    ]


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_frames(frames):

    data = np.asarray(
        frames,
        dtype=np.float32
    )

    if data.ndim != 2:
        raise ValueError(
            f"Expected 2D array, got {data.shape}"
        )

    if data.shape[1] != 258:
        raise ValueError(
            f"Expected 258 features, got {data.shape[1]}"
        )

    data = HandLandmarkInterpolator().interpolate(data)

    data = LandmarkNormalizer().normalize(data)

    # If exactly 60 frames are supplied, SequenceGenerator
    # returns those frames unchanged.
    data = SequenceGenerator().generate(data)

    return data.astype(np.float32)


# ============================================================
# PREDICTION
# ============================================================

def predict_frames(model, class_names, frames):

    sequence = preprocess_frames(frames)

    probabilities = model.predict(
        sequence[np.newaxis, ...],
        verbose=0
    )[0]

    order = np.argsort(
        probabilities
    )[::-1]

    return [
        (
            class_names[int(index)],
            float(probabilities[int(index)])
        )
        for index in order[:5]
    ]


# ============================================================
# DISPLAY
# ============================================================

def print_prediction(name, frame_range, results):

    label, confidence = results[0]

    print(
        f"{name:<22} "
        f"{frame_range:<18} "
        f"{label:<30} "
        f"{confidence * 100:6.2f}%"
    )

    print(
        f"    Top-5: "
        f"{' | '.join(label + ' ' + f'{conf * 100:.1f}%' for label, conf in results)}"
    )


# ============================================================
# FIND CLOCK FILES
# ============================================================

def find_clock_files():

    target_names = {
        "MVI_4959.npy",
        "MVI_4960.npy"
    }

    matches = []

    for path in DATASET_DIR.rglob("*.npy"):

        if (
            path.name in target_names
            or any(
                path.name.endswith("__" + name)
                for name in target_names
            )
        ):

            if path.parent.name == "51. Clock":

                matches.append(path)

    return sorted(matches)


# ============================================================
# TEST ONE VIDEO
# ============================================================

def test_video(model, class_names, path):

    data = np.load(path)

    print("\n" + "=" * 78)
    print(f"FILE: {path}")
    print(f"Frames available: {len(data)}")
    print("=" * 78)

    if data.ndim != 2 or data.shape[1] != 258:
        print(
            f"INVALID SHAPE: {data.shape}"
        )
        return

    total_frames = len(data)

    if total_frames < SEQUENCE_LENGTH:

        print(
            f"Too short for a 60-frame window: "
            f"{total_frames}"
        )

        return

    # --------------------------------------------------------
    # Training-style uniform sampling
    # --------------------------------------------------------

    uniform_indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH
    ).astype(int)

    uniform_frames = data[
        uniform_indices
    ]

    results = predict_frames(
        model,
        class_names,
        uniform_frames
    )

    print_prediction(
        "Training-style",
        "uniform 60",
        results
    )

    # --------------------------------------------------------
    # First consecutive 60
    # --------------------------------------------------------

    results = predict_frames(
        model,
        class_names,
        data[:SEQUENCE_LENGTH]
    )

    print_prediction(
        "Consecutive",
        "first 0-59",
        results
    )

    # --------------------------------------------------------
    # Middle consecutive 60
    # --------------------------------------------------------

    middle_start = max(
        0,
        (total_frames - SEQUENCE_LENGTH) // 2
    )

    results = predict_frames(
        model,
        class_names,
        data[
            middle_start:
            middle_start + SEQUENCE_LENGTH
        ]
    )

    print_prediction(
        "Consecutive",
        f"middle {middle_start}-"
        f"{middle_start + SEQUENCE_LENGTH - 1}",
        results
    )

    # --------------------------------------------------------
    # Last consecutive 60
    # --------------------------------------------------------

    last_start = (
        total_frames - SEQUENCE_LENGTH
    )

    results = predict_frames(
        model,
        class_names,
        data[
            last_start:
            last_start + SEQUENCE_LENGTH
        ]
    )

    print_prediction(
        "Consecutive",
        f"last {last_start}-"
        f"{total_frames - 1}",
        results
    )

    # --------------------------------------------------------
    # Sliding windows
    # --------------------------------------------------------

    print("\nSliding windows:")

    window_results = []

    starts = list(
        range(
            0,
            total_frames - SEQUENCE_LENGTH + 1,
            SLIDING_STEP
        )
    )

    # Always include the final possible window.
    final_start = (
        total_frames - SEQUENCE_LENGTH
    )

    if final_start not in starts:
        starts.append(final_start)

    starts = sorted(set(starts))

    for start in starts:

        end = (
            start + SEQUENCE_LENGTH - 1
        )

        results = predict_frames(
            model,
            class_names,
            data[start:end + 1]
        )

        label, confidence = results[0]

        window_results.append(
            (
                start,
                end,
                label,
                confidence
            )
        )

        print(
            f"  {start:4d}-{end:4d} "
            f"-> {label:<30} "
            f"{confidence * 100:6.2f}%"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    clock_windows = [
        item
        for item in window_results
        if item[2] == "51. Clock"
    ]

    truck_windows = [
        item
        for item in window_results
        if item[2] == "12. Truck"
    ]

    print("\nWindow summary:")
    print(
        f"  Total windows : {len(window_results)}"
    )
    print(
        f"  Clock         : {len(clock_windows)}"
    )
    print(
        f"  Truck         : {len(truck_windows)}"
    )

    if clock_windows:

        best_clock = max(
            clock_windows,
            key=lambda item: item[3]
        )

        print(
            "  Best Clock    : "
            f"{best_clock[0]}-"
            f"{best_clock[1]} "
            f"({best_clock[3] * 100:.2f}%)"
        )

    if truck_windows:

        best_truck = max(
            truck_windows,
            key=lambda item: item[3]
        )

        print(
            "  Best Truck    : "
            f"{best_truck[0]}-"
            f"{best_truck[1]} "
            f"({best_truck[3] * 100:.2f}%)"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "GENERALIZED 114-CLASS MODEL - "
        "TEMPORAL WINDOW EXPERIMENT"
    )
    print("=" * 78)

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    class_names = load_class_names()

    print(
        f"Model loaded. Classes: {len(class_names)}"
    )

    files = find_clock_files()

    print(
        f"Clock files found: {len(files)}"
    )

    if not files:

        print(
            "ERROR: No Clock landmark files found."
        )
        return

    for path in files:

        test_video(
            model,
            class_names,
            path
        )

    print("\n" + "=" * 78)
    print("TEMPORAL WINDOW EXPERIMENT COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()