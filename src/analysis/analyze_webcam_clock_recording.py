from pathlib import Path
import json
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "saved_models" / "isl_lstm_generalized_114_legacy.keras"
LABEL_PATH = ROOT / "outputs" / "generalized_114_label_mapping.json"

WEBCAM_PATH = ROOT / "outputs" / "webcam_test" / "clock_webcam_20260920_222015.npy"

SEQ_LEN = 60


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    label_mapping = json.load(f)

# label_mapping is expected to be: label -> index
class_names = [None] * len(label_mapping)

for label, index in label_mapping.items():
    class_names[int(index)] = label


print(f"Classes loaded: {len(class_names)}")


# ---------------------------------------------------------
# LOAD WEBCAM LANDMARKS
# ---------------------------------------------------------

data = np.load(WEBCAM_PATH)

print("\nWebcam recording:")
print(f"Shape: {data.shape}")

if data.ndim != 2 or data.shape[1] != 258:
    raise ValueError(
        f"Expected shape (frames, 258), got {data.shape}"
    )

num_frames = data.shape[0]

print(f"Frames: {num_frames}")
print(f"Features: {data.shape[1]}")


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

def prepare_sequence(sequence):
    """
    Prepare exactly 60 frames for the generalized model.

    If more than 60 frames are supplied:
        uniformly sample 60 frames.

    If exactly 60:
        use them directly.

    If fewer than 60:
        zero-pad.
    """

    sequence = np.asarray(sequence, dtype=np.float32)

    if len(sequence) > SEQ_LEN:

        indices = np.linspace(
            0,
            len(sequence) - 1,
            SEQ_LEN
        ).astype(int)

        sequence = sequence[indices]

    elif len(sequence) < SEQ_LEN:

        padding = np.zeros(
            (SEQ_LEN - len(sequence), sequence.shape[1]),
            dtype=np.float32
        )

        sequence = np.vstack([sequence, padding])

    return sequence


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

def predict_sequence(sequence):

    sequence = prepare_sequence(sequence)

    X = np.expand_dims(sequence, axis=0)

    probabilities = model.predict(
        X,
        verbose=0
    )[0]

    top_indices = np.argsort(
        probabilities
    )[::-1][:5]

    results = []

    for index in top_indices:

        results.append(
            (
                class_names[index],
                float(probabilities[index])
            )
        )

    return results


# ---------------------------------------------------------
# DISPLAY RESULT
# ---------------------------------------------------------

def show_result(name, sequence):

    results = predict_sequence(sequence)

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    for rank, (label, confidence) in enumerate(results, start=1):

        print(
            f"{rank}. {label:<30} "
            f"{confidence * 100:6.2f}%"
        )

    return results[0][0]


# ---------------------------------------------------------
# TEST 1
# TRAINING-STYLE UNIFORM SAMPLING
# ---------------------------------------------------------

print("\n\nTEST 1: Training-style uniform sampling")

uniform_prediction = show_result(
    "Full recording -> uniformly sampled to 60 frames",
    data
)


# ---------------------------------------------------------
# TEST 2
# FIRST 60 FRAMES
# ---------------------------------------------------------

first_prediction = show_result(
    "First 60 frames",
    data[:60]
)


# ---------------------------------------------------------
# TEST 3
# MIDDLE 60 FRAMES
# ---------------------------------------------------------

middle_start = max(
    0,
    (num_frames - SEQ_LEN) // 2
)

middle_end = middle_start + SEQ_LEN

middle_prediction = show_result(
    f"Middle 60 frames ({middle_start}-{middle_end - 1})",
    data[middle_start:middle_end]
)


# ---------------------------------------------------------
# TEST 4
# LAST 60 FRAMES
# ---------------------------------------------------------

last_start = num_frames - SEQ_LEN

last_prediction = show_result(
    f"Last 60 frames ({last_start}-{num_frames - 1})",
    data[last_start:]
)


# ---------------------------------------------------------
# TEST 5
# SLIDING WINDOWS
# ---------------------------------------------------------

print("\n\n")
print("=" * 65)
print("TEST 5: Sliding 60-frame windows")
print("=" * 65)

sliding_results = []

for start in range(
    0,
    num_frames - SEQ_LEN + 1
):

    end = start + SEQ_LEN

    results = predict_sequence(
        data[start:end]
    )

    top_label = results[0][0]
    top_confidence = results[0][1]

    sliding_results.append(
        (
            start,
            end - 1,
            top_label,
            top_confidence
        )
    )

    print(
        f"Frames {start:02d}-{end - 1:02d} -> "
        f"{top_label:<30} "
        f"{top_confidence * 100:6.2f}%"
    )


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print("\n\n")
print("=" * 65)
print("SUMMARY")
print("=" * 65)

print(
    f"Training-style : {uniform_prediction}"
)

print(
    f"First 60       : {first_prediction}"
)

print(
    f"Middle 60      : {middle_prediction}"
)

print(
    f"Last 60        : {last_prediction}"
)

print("\nSliding-window predictions:")

for start, end, label, confidence in sliding_results:

    print(
        f"  {start:02d}-{end:02d}: "
        f"{label:<30} "
        f"{confidence * 100:6.2f}%"
    )


# ---------------------------------------------------------
# CLOCK / TRUCK CHECK
# ---------------------------------------------------------

all_predictions = [
    uniform_prediction,
    first_prediction,
    middle_prediction,
    last_prediction
]

all_predictions.extend(
    result[2]
    for result in sliding_results
)

clock_count = sum(
    1
    for label in all_predictions
    if label.lower() == "clock"
)

truck_count = sum(
    1
    for label in all_predictions
    if "truck" in label.lower()
)

print("\n" + "=" * 65)
print("CLOCK / TRUCK DIAGNOSTIC")
print("=" * 65)

print(f"Clock predictions : {clock_count}")
print(f"Truck predictions : {truck_count}")
print(f"Total tests       : {len(all_predictions)}")

print("\nInterpretation:")

if truck_count > 0 and clock_count > 0:

    print(
        "The recording is sensitive to temporal window selection."
    )

elif truck_count == len(all_predictions):

    print(
        "The recorded webcam landmarks are consistently "
        "being classified as Truck."
    )

elif clock_count == len(all_predictions):

    print(
        "The recorded webcam landmarks are consistently "
        "being classified as Clock."
    )

else:

    print(
        "The recording produces mixed predictions. "
        "Further analysis is required."
    )