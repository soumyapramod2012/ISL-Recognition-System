"""
Experimental 136-class generalized ISL training.

IMPORTANT:
- Does NOT modify the existing 24-class or 114-class models.
- Uses the filtered dataset created by create_filtered_generalized_dataset.py.
- Legacy representation: MediaPipe Holistic 0.10.21, 258 features/frame, 60 frames.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_dataset import TrainDataset
from src.training.model import ISLModel


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATASET_PATH = "dataset/processed/generalized_filtered_legacy"
SPLIT_PATH = Path("dataset/split_generalized_filtered_legacy.json")

OUTPUT_MODEL = Path(
    "saved_models/isl_lstm_generalized_filtered_136_legacy.keras"
)
OUTPUT_MAPPING = Path(
    "outputs/generalized_filtered_136_label_mapping.json"
)
OUTPUT_HISTORY = Path(
    "outputs/generalized_filtered_136_training_history.json"
)
OUTPUT_RESULT = Path(
    "outputs/generalized_filtered_136_training_result.txt"
)

EXPECTED_CLASSES = 136
SEQUENCE_LENGTH = 60
FEATURE_SIZE = 258


def main():
    print("=" * 70)
    print("GENERALIZED ISL - FILTERED 136 CLASS TRAINING")
    print("=" * 70)
    print(f"Dataset : {DATASET_PATH}")
    print(f"Split   : {SPLIT_PATH}")
    print(f"Model   : {OUTPUT_MODEL}")

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

    num_classes = len(encoder.label_to_index)

    if num_classes != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Expected {EXPECTED_CLASSES} classes, found {num_classes}"
        )

    if X_train.shape[1:] != (SEQUENCE_LENGTH, FEATURE_SIZE):
        raise RuntimeError(
            f"Unexpected training shape: {X_train.shape}"
        )

    class_names = [
        encoder.decode(i)
        for i in range(num_classes)
    ]

    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MAPPING.parent.mkdir(parents=True, exist_ok=True)

    # Save the exact mapping used by this experiment.
    OUTPUT_MAPPING.write_text(
        json.dumps(
            encoder.label_to_index,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Balanced class weights.
    counts = np.bincount(
        y_train,
        minlength=num_classes,
    ).astype(np.float32)

    if np.any(counts == 0):
        missing = [
            class_names[i]
            for i, count in enumerate(counts)
            if count == 0
        ]
        raise RuntimeError(
            f"Classes missing from training split: {missing}"
        )

    class_weights = (
        counts.sum()
        / (num_classes * counts)
    )

    print("\nDataset summary")
    print("-" * 50)
    print("Classes        :", num_classes)
    print("Train          :", X_train.shape)
    print("Validation     :", X_val.shape)
    print("Test           :", X_test.shape)
    print("Feature size   :", X_train.shape[2])
    print("Sequence length:", X_train.shape[1])
    print(
        "Weight range   : "
        f"{class_weights.min():.4f} - "
        f"{class_weights.max():.4f}"
    )

    # Use the established ISL BiLSTM architecture.
    model = ISLModel().build(num_classes)

    print("\nModel")
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            str(OUTPUT_MODEL),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        class_weight={
            i: float(class_weights[i])
            for i in range(num_classes)
        },
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate the restored/best model.
    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0,
    )

    probabilities = model.predict(
        X_test,
        verbose=0,
    )
    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    top3 = np.any(
        np.argsort(probabilities, axis=1)[:, -3:] == y_test[:, None],
        axis=1,
    ).mean()

    top5 = np.any(
        np.argsort(probabilities, axis=1)[:, -5:] == y_test[:, None],
        axis=1,
    ).mean()

    # Best epoch information.
    val_losses = history.history.get("val_loss", [])
    val_accs = history.history.get("val_accuracy", [])

    best_epoch = (
        int(np.argmin(val_losses)) + 1
        if val_losses else None
    )
    best_val_loss = (
        float(min(val_losses))
        if val_losses else None
    )
    best_val_accuracy = (
        float(val_accs[best_epoch - 1])
        if val_accs and best_epoch
        else None
    )

    result_lines = [
        "GENERALIZED ISL - FILTERED 136 CLASS TRAINING",
        "",
        f"Dataset: {DATASET_PATH}",
        f"Split: {SPLIT_PATH}",
        f"Classes: {num_classes}",
        f"Train samples: {len(y_train)}",
        f"Validation samples: {len(y_val)}",
        f"Test samples: {len(y_test)}",
        f"Sequence length: {SEQUENCE_LENGTH}",
        f"Feature size: {FEATURE_SIZE}",
        "",
        f"Best epoch: {best_epoch}",
        f"Best validation loss: {best_val_loss}",
        f"Best validation accuracy: {best_val_accuracy}",
        "",
        f"Test loss: {loss:.6f}",
        f"Test top-1 accuracy: {accuracy:.6%}",
        f"Test top-3 accuracy: {top3:.6%}",
        f"Test top-5 accuracy: {top5:.6%}",
        "",
        f"Model: {OUTPUT_MODEL}",
        f"Mapping: {OUTPUT_MAPPING}",
        f"History: {OUTPUT_HISTORY}",
    ]

    OUTPUT_RESULT.write_text(
        "\n".join(result_lines),
        encoding="utf-8",
    )

    # Save Keras history in a JSON-safe form.
    OUTPUT_HISTORY.write_text(
        json.dumps(
            {
                key: [float(v) for v in values]
                for key, values in history.history.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best epoch          : {best_epoch}")
    print(f"Best val loss       : {best_val_loss:.6f}")
    print(f"Best val accuracy   : {best_val_accuracy:.6%}")
    print(f"Test top-1          : {accuracy:.6%}")
    print(f"Test top-3          : {top3:.6%}")
    print(f"Test top-5          : {top5:.6%}")
    print(f"\nSaved model         : {OUTPUT_MODEL}")
    print(f"Saved mapping       : {OUTPUT_MAPPING}")
    print(f"Saved history       : {OUTPUT_HISTORY}")
    print(f"Saved result        : {OUTPUT_RESULT}")


if __name__ == "__main__":
    main()
