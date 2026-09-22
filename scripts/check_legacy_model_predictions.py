from pathlib import Path
import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "isl_lstm_class_weighted_legacy_compatible.h5"
)

DATA_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "parity_test_data.npz"
)


data = np.load(DATA_PATH)

X_test = data["X_test"]
y_test = data["y_test"]
original_predictions = data["original_predictions"]

print("Loading compatible model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

print("Compatible model loaded.")
print("Input :", model.input_shape)
print("Output:", model.output_shape)

compatible_predictions = model.predict(
    X_test,
    verbose=0,
)

original_classes = np.argmax(
    original_predictions,
    axis=1,
)

compatible_classes = np.argmax(
    compatible_predictions,
    axis=1,
)

max_diff = np.max(
    np.abs(
        original_predictions
        - compatible_predictions
    )
)

mean_diff = np.mean(
    np.abs(
        original_predictions
        - compatible_predictions
    )
)

same_classes = np.mean(
    original_classes
    == compatible_classes
)

original_accuracy = np.mean(
    original_classes == y_test
)

compatible_accuracy = np.mean(
    compatible_classes == y_test
)

print()
print("=" * 70)
print("MODEL PARITY RESULT")
print("=" * 70)

print(
    f"Maximum probability difference : {max_diff:.10f}"
)

print(
    f"Mean probability difference    : {mean_diff:.10f}"
)

print(
    f"Identical predicted classes    : "
    f"{same_classes * 100:.2f}%"
)

print(
    f"Original model accuracy        : "
    f"{original_accuracy * 100:.2f}%"
)

print(
    f"Compatible model accuracy     : "
    f"{compatible_accuracy * 100:.2f}%"
)

if same_classes == 1.0 and max_diff < 1e-5:
    print()
    print("PASS: Model conversion preserved predictions.")
else:
    print()
    print("FAIL: Model conversion changed predictions.")