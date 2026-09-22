"""
Train the experimental 114-class generalized ISL model.

IMPORTANT:
- Does not modify the existing 24-class model.
- Does not modify src/inference/realtime.py.
- Uses the generalized stratified split.
- Uses the existing 60 x 258 preprocessing pipeline.
"""

import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.models import Model

from src.training.train_dataset import TrainDataset
import src.training.config as config


# ============================================================
# Reproducibility
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# Paths
# ============================================================

DATASET_PATH = Path("dataset/processed/generalized_landmarks_legacy")
SPLIT_PATH = Path("dataset/split_generalized_legacy_stratified.json")

MODEL_PATH = Path(
    "saved_models/isl_lstm_generalized_114_legacy.keras"
)

LABEL_PATH = Path(
    "outputs/generalized_114_label_mapping.json"
)

HISTORY_PATH = Path(
    "outputs/generalized_114_training_history.json"
)

RESULT_PATH = Path(
    "outputs/generalized_114_training_result.txt"
)


# ============================================================
# Model
# ============================================================

def build_model(num_classes):
    model = tf.keras.Sequential(
        [
            Input(shape=(60, 258)),

            Bidirectional(
                LSTM(128, return_sequences=True)
            ),
            Dropout(0.30),

            Bidirectional(
                LSTM(64, return_sequences=False)
            ),
            Dropout(0.30),

            Dense(64, activation="relu"),
            Dropout(0.20),

            Dense(num_classes, activation="softmax"),
        ],
        name="ISL_Generalized_114_BiLSTM",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("GENERALIZED ISL - 114 CLASS TRAINING")
    print("=" * 60)

    print("\nDataset:", DATASET_PATH)
    print("Split  :", SPLIT_PATH)
    print("Model  :", MODEL_PATH)

    # --------------------------------------------------------
    # Load and preprocess dataset
    # --------------------------------------------------------

    dataset = TrainDataset(DATASET_PATH)
    dataset.split_file = SPLIT_PATH

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        encoder,
    ) = dataset.build()

    num_classes = len(encoder.classes)

    print("\nVerified dataset:")
    print("  Train      :", X_train.shape)
    print("  Validation :", X_val.shape)
    print("  Test       :", X_test.shape)
    print("  Classes    :", num_classes)

    if num_classes != 114:
        raise RuntimeError(
            f"Expected 114 classes, found {num_classes}"
        )

    if X_train.shape[1:] != (60, 258):
        raise RuntimeError(
            f"Unexpected input shape: {X_train.shape[1:]}"
        )

    # --------------------------------------------------------
    # Save label mapping
    # --------------------------------------------------------

    LABEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    encoder.save(LABEL_PATH)

    print("\nLabel mapping saved to:")
    print(" ", LABEL_PATH)

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    classes = np.arange(num_classes)

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(classes, class_weights_array)
    }

    print("\nClass weighting enabled.")
    print(
        "Weight range:",
        f"{min(class_weights.values()):.3f}",
        "to",
        f"{max(class_weights.values()):.3f}",
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model(num_classes)

    print("\nModel:")
    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),

        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=1,
    )

    print(f"\nTest Loss     : {test_loss:.6f}")
    print(f"Test Accuracy : {test_accuracy:.6f}")

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_data = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)

    # --------------------------------------------------------
    # Save result summary
    # --------------------------------------------------------

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        f.write("GENERALIZED ISL 114-CLASS TRAINING RESULT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Classes: {num_classes}\n")
        f.write(f"Train samples: {len(X_train)}\n")
        f.write(f"Validation samples: {len(X_val)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Sequence length: {X_train.shape[1]}\n")
        f.write(f"Feature size: {X_train.shape[2]}\n")
        f.write(f"Test loss: {test_loss:.6f}\n")
        f.write(f"Test accuracy: {test_accuracy:.6f}\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Labels: {LABEL_PATH}\n")

    print("\nArtifacts:")
    print("  Model   :", MODEL_PATH)
    print("  Labels  :", LABEL_PATH)
    print("  History :", HISTORY_PATH)
    print("  Result  :", RESULT_PATH)

    print("\nTRAINING COMPLETE")


if __name__ == "__main__":
    main()