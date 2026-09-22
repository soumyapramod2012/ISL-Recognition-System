from pathlib import Path
import sys
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_dataset import TrainDataset


ORIGINAL_MODEL = (
    PROJECT_ROOT
    / "saved_models"
    / "isl_lstm_class_weighted_legacy.keras"
)

COMPATIBLE_MODEL = (
    PROJECT_ROOT
    / "saved_models"
    / "isl_lstm_class_weighted_legacy_compatible.h5"
)

SPLIT_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "split_legacy.json"
)

DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "landmarks_legacy"
)


print("=" * 70)
print("MODEL PARITY CHECK")
print("=" * 70)

print("Loading test data...")

dataset = TrainDataset(DATASET)
dataset.split_file = SPLIT_FILE

(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    encoder,
) = dataset.build()

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print()
print("Loading original Keras model...")
original = tf.keras.models.load_model(
    ORIGINAL_MODEL,
    compile=False,
)

print("Loading compatible H5 model...")
compatible = tf.keras.models.load_model(
    COMPATIBLE_MODEL,
    compile=False,
)

print()
print("Original output :", original.output_shape)
print("Compatible output:", compatible.output_shape)

print()
print("Running predictions...")

p_original = original.predict(
    X_test,
    verbose=0,
)

p_compatible = compatible.predict(
    X_test,
    verbose=0,
)

# ------------------------------------------------------------
# Numerical comparison
# ------------------------------------------------------------

max_diff = float(
    np.max(
        np.abs(
            p_original - p_compatible
        )
    )
)

mean_diff = float(
    np.mean(
        np.abs(
            p_original - p_compatible
        )
    )
)

original_pred = np.argmax(
    p_original,
    axis=1,
)

compatible_pred = np.argmax(
    p_compatible,
    axis=1,
)

same_predictions = np.mean(
    original_pred == compatible_pred
)

original_accuracy = np.mean(
    original_pred == y_test
)

compatible_accuracy = np.mean(
    compatible_pred == y_test
)

print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"Maximum probability difference : "
    f"{max_diff:.10f}"
)

print(
    f"Mean probability difference    : "
    f"{mean_diff:.10f}"
)

print(
    f"Identical predicted classes    : "
    f"{same_predictions * 100:.2f}%"
)

print(
    f"Original model accuracy        : "
    f"{original_accuracy * 100:.2f}%"
)

print(
    f"Compatible model accuracy     : "
    f"{compatible_accuracy * 100:.2f}%"
)

print()

if (
    max_diff < 1e-5
    and same_predictions == 1.0
):
    print("PASS: Models are prediction-equivalent.")
else:
    print("FAIL: Model conversion changed predictions.")