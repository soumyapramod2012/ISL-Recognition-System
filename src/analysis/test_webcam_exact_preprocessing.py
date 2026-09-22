from pathlib import Path
import json
import numpy as np
import tensorflow as tf

from src.training.hand_landmark_interpolator import HandLandmarkInterpolator
from src.training.landmark_normalizer import LandmarkNormalizer
from src.training.sequence_generator import SequenceGenerator


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "saved_models"
    / "isl_lstm_generalized_114_legacy.keras"
)

LABEL_PATH = (
    ROOT
    / "outputs"
    / "generalized_114_label_mapping.json"
)

WEBCAM_PATH = (
    ROOT
    / "outputs"
    / "webcam_test"
    / "clock_webcam_20260920_222015.npy"
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

with open(
    LABEL_PATH,
    "r",
    encoding="utf-8"
) as f:
    label_mapping = json.load(f)

class_names = [None] * len(label_mapping)

for label, index in label_mapping.items():
    class_names[int(index)] = label

print(
    f"Classes loaded: {len(class_names)}"
)


# ---------------------------------------------------------
# LOAD EXACT PREPROCESSING CLASSES
# ---------------------------------------------------------

interpolator = HandLandmarkInterpolator(
    max_gap=5
)

normalizer = LandmarkNormalizer()

sequence_generator = SequenceGenerator()


# ---------------------------------------------------------
# LOAD WEBCAM LANDMARKS
# ---------------------------------------------------------

landmarks = np.load(
    WEBCAM_PATH
)

print("\nRaw webcam landmarks:")
print(
    f"Shape: {landmarks.shape}"
)


# ---------------------------------------------------------
# EXACT TRAINING PIPELINE
# ---------------------------------------------------------

print("\nApplying exact preprocessing...")

# Step 1
landmarks = interpolator.interpolate(
    landmarks
)

print(
    f"After interpolation: {landmarks.shape}"
)


# Step 2
landmarks = normalizer.normalize(
    landmarks
)

print(
    f"After normalization: {landmarks.shape}"
)


# Step 3
sequence = sequence_generator.generate(
    landmarks
)

print(
    f"After sequence generation: {sequence.shape}"
)


# ---------------------------------------------------------
# MODEL PREDICTION
# ---------------------------------------------------------

X = np.expand_dims(
    sequence,
    axis=0
)

probabilities = model.predict(
    X,
    verbose=0
)[0]


# ---------------------------------------------------------
# TOP 10
# ---------------------------------------------------------

top_indices = np.argsort(
    probabilities
)[::-1][:10]


print("\n")
print("=" * 65)
print("EXACT PREPROCESSING RESULT")
print("=" * 65)

for rank, index in enumerate(
    top_indices,
    start=1
):

    print(
        f"{rank:2d}. "
        f"{class_names[index]:<30} "
        f"{probabilities[index] * 100:6.2f}%"
    )


# ---------------------------------------------------------
# SPECIFIC CLASS CHECK
# ---------------------------------------------------------

print("\n")
print("=" * 65)
print("SPECIFIC CLASS CHECK")
print("=" * 65)

for target in [
    "51. Clock",
    "15. Boat",
    "12. Truck"
]:

    if target in label_mapping:

        index = int(
            label_mapping[target]
        )

        print(
            f"{target:<20} "
            f"{probabilities[index] * 100:6.2f}%"
        )


# ---------------------------------------------------------
# BASIC DATA CHECK
# ---------------------------------------------------------

print("\n")
print("=" * 65)
print("DATA CHECK")
print("=" * 65)

print(
    "NaN values:",
    np.isnan(sequence).sum()
)

print(
    "Inf values:",
    np.isinf(sequence).sum()
)

print(
    "Minimum:",
    np.min(sequence)
)

print(
    "Maximum:",
    np.max(sequence)
)

print(
    "Mean:",
    np.mean(sequence)
)

print(
    "Std:",
    np.std(sequence)
)