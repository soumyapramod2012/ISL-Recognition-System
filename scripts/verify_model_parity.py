from pathlib import Path
import sys
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

COMPATIBLE_MODEL = (
    PROJECT_ROOT
    / "saved_models"
    / "isl_lstm_class_weighted_legacy_compatible.h5"
)

PARITY_DATA = (
    PROJECT_ROOT
    / "outputs"
    / "parity_test_data.npz"
)


print("=" * 70)
print("MODEL PARITY CHECK")
print("=" * 70)

# ------------------------------------------------------------
# Check parity artifact
# ------------------------------------------------------------

if not PARITY_DATA.exists():
    raise FileNotFoundError(
        f"Parity data not found: {PARITY_DATA}"
    )

print("Loading saved parity data...")

data = np.load(PARITY_DATA)

required = {
    "X_test",
    "y_test",
    "original_predictions",
}

missing = required.difference(data.files)

if missing:
    raise RuntimeError(
        f"Missing arrays in parity data: {sorted(missing)}"
    )

X_test = data["X_test"]
y_test = data["y_test"]
original_predictions = data["original_predictions"]

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
print("Original predictions:", original_predictions.shape)

# ------------------------------------------------------------
# Basic validation
# ------------------------------------------------------------

if X_test.ndim != 3:
    raise RuntimeError(
        f"Unexpected X_test dimensions: {X_test.shape}"
    )

if X_test.shape[1:] != (60, 258):
    raise RuntimeError(
        f"Unexpected X_test shape: {X_test.shape}"
    )

if len(X_test) != len(y_test):
    raise RuntimeError(
        "X_test and y_test sample counts do not match."
    )

if original_predictions.shape[0] != len(X_test):
    raise RuntimeError(
        "Prediction count does not match X_test."
    )

print()
print("Test samples:", len(X_test))
print("Feature shape:", X_test.shape[1:])

# ------------------------------------------------------------
# Original model accuracy from saved predictions
# ------------------------------------------------------------

original_pred = np.argmax(
    original_predictions,
    axis=1,
)

original_accuracy = np.mean(
    original_pred == y_test
)

print()
print(
    f"Original saved-model accuracy : "
    f"{original_accuracy * 100:.2f}%"
)

# ------------------------------------------------------------
# Load compatible model
# ------------------------------------------------------------

print()
print("Loading compatible H5 model...")

if not COMPATIBLE_MODEL.exists():
    raise FileNotFoundError(
        f"Compatible model not found: {COMPATIBLE_MODEL}"
    )

compatible = tf.keras.models.load_model(
    COMPATIBLE_MODEL,
    compile=False,
)

print("Compatible model loaded.")
print("Input shape :", compatible.input_shape)
print("Output shape:", compatible.output_shape)

# ------------------------------------------------------------
# Compatible predictions
# ------------------------------------------------------------

print()
print("Running compatible-model predictions...")

compatible_predictions = compatible.predict(
    X_test,
    verbose=0,
)

# ------------------------------------------------------------
# Numerical comparison
# ------------------------------------------------------------

max_diff = float(
    np.max(
        np.abs(
            original_predictions - compatible_predictions
        )
    )
)

mean_diff = float(
    np.mean(
        np.abs(
            original_predictions - compatible_predictions
        )
    )
)

compatible_pred = np.argmax(
    compatible_predictions,
    axis=1,
)

same_predictions = np.mean(
    original_pred == compatible_pred
)

compatible_accuracy = np.mean(
    compatible_pred == y_test
)

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

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
    raise SystemExit(0)

print("FAIL: Model conversion changed predictions.")
raise SystemExit(1)