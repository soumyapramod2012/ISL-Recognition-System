"""
Targeted training experiment for the filtered 136-class ISL model.

Purpose:
    Test whether targeted oversampling + class weighting improves the
    identified confusion groups without modifying the existing 136-class
    model, dataset, or realtime inference.

Target classes:
    42. he
    43. she
    58. Son
    59. Daughter
    47. Red
    52. Pink
    53. Fan
    52. Lamp
    64. Fall

IMPORTANT:
    - This creates a NEW experimental model.
    - Existing 136-class model is NOT overwritten.
    - Existing dataset is NOT modified.
    - realtime.py is NOT modified.
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

import src.training.config as config
from src.training.train_dataset import TrainDataset
from src.training.model import ISLModel


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATASET_PATH = "dataset/processed/generalized_filtered_legacy"
SPLIT_PATH = Path("dataset/split_generalized_filtered_legacy.json")

OUTPUT_MODEL = (
    "saved_models/isl_lstm_generalized_filtered_136_targeted.keras"
)
OUTPUT_MAPPING = Path(
    "outputs/generalized_filtered_136_targeted_label_mapping.json"
)
OUTPUT_HISTORY = Path(
    "outputs/generalized_filtered_136_targeted_training_history.json"
)
OUTPUT_RESULT = Path(
    "outputs/generalized_filtered_136_targeted_training_result.txt"
)

TARGET_CLASSES = {
    "42. he",
    "43. she",
    "58. Son",
    "59. Daughter",
    "47. Red",
    "52. Pink",
    "53. Fan",
    "52. Lamp",
    "64. Fall",
}

# Moderate oversampling:
# target samples are duplicated once, so their training presence is doubled.
OVERSAMPLE_FACTOR = 2.0

# Moderate additional class-weight multiplier.
TARGET_WEIGHT_MULTIPLIER = 1.5


def main():
    print("=" * 80)
    print("TARGETED 136-CLASS TRAINING EXPERIMENT")
    print("=" * 80)
    print("Existing model will NOT be overwritten.")
    print("Existing dataset will NOT be modified.")
    print("realtime.py will NOT be modified.")
    print()

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
    class_names = [
        encoder.decode(i)
        for i in range(num_classes)
    ]

    print("Dataset:")
    print(f"  Train : {X_train.shape}")
    print(f"  Val   : {X_val.shape}")
    print(f"  Test  : {X_test.shape}")
    print(f"  Classes: {num_classes}")

    # ------------------------------------------------------------
    # 1. Balanced class weights
    # ------------------------------------------------------------
    counts = np.bincount(
        y_train,
        minlength=num_classes,
    ).astype(np.float32)

    if np.any(counts == 0):
        raise RuntimeError("At least one class has zero training samples.")

    weights = counts.sum() / (num_classes * counts)

    # Additional weight for the nine target classes.
    for name in TARGET_CLASSES:
        if name not in encoder.label_to_index:
            raise RuntimeError(
                f"Target class missing from encoder: {name}"
            )

        idx = encoder.label_to_index[name]
        weights[idx] *= TARGET_WEIGHT_MULTIPLIER

    # Keep average weight approximately 1.
    weights = weights / np.mean(weights)

    # ------------------------------------------------------------
    # 2. Targeted oversampling ONLY on training data
    # ------------------------------------------------------------
    target_indices = {
        encoder.label_to_index[name]
        for name in TARGET_CLASSES
    }

    selected = [
        i for i, label in enumerate(y_train)
        if int(label) in target_indices
    ]

    if not selected:
        raise RuntimeError("No target training samples found.")

    X_extra = X_train[selected]
    y_extra = y_train[selected]

    X_train_aug = np.concatenate(
        [X_train, X_extra],
        axis=0,
    )

    y_train_aug = np.concatenate(
        [y_train, y_extra],
        axis=0,
    )

    # Deterministic shuffle.
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(y_train_aug))

    X_train_aug = X_train_aug[order]
    y_train_aug = y_train_aug[order]

    print()
    print("Target classes:")
    for name in sorted(TARGET_CLASSES):
        idx = encoder.label_to_index[name]
        original_count = int(counts[idx])
        print(
            f"  {name:20s} "
            f"original={original_count:3d} "
            f"after_oversampling={original_count * int(OVERSAMPLE_FACTOR):3d} "
            f"weight={weights[idx]:.4f}"
        )

    print()
    print(f"Original training samples : {len(y_train)}")
    print(f"Augmented training samples: {len(y_train_aug)}")

    # ------------------------------------------------------------
    # 3. Build NEW model
    # ------------------------------------------------------------
    model = ISLModel().build(num_classes)

    checkpoint = ModelCheckpoint(
        OUTPUT_MODEL,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        checkpoint,
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train_aug,
        y_train_aug,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight={
            i: float(weights[i])
            for i in range(num_classes)
        },
        callbacks=callbacks,
        verbose=1,
    )

    # ------------------------------------------------------------
    # 4. Test evaluation
    # ------------------------------------------------------------
    test_loss, test_accuracy = model.evaluate(
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

    top3 = np.argsort(
        probabilities,
        axis=1,
    )[:, -3:]

    top5 = np.argsort(
        probabilities,
        axis=1,
    )[:, -5:]

    top3_accuracy = np.mean(
        [
            int(y in row)
            for y, row in zip(y_test, top3)
        ]
    )

    top5_accuracy = np.mean(
        [
            int(y in row)
            for y, row in zip(y_test, top5)
        ]
    )

    # ------------------------------------------------------------
    # 5. Target-class evaluation
    # ------------------------------------------------------------
    target_results = {}

    for name in sorted(TARGET_CLASSES):
        idx = encoder.label_to_index[name]
        mask = y_test == idx

        if not np.any(mask):
            continue

        target_probs = probabilities[mask]
        target_pred = predictions[mask]

        target_results[name] = {
            "test_samples": int(np.sum(mask)),
            "top1_accuracy": float(
                np.mean(target_pred == idx)
            ),
            "top3_accuracy": float(
                np.mean([
                    idx in row
                    for row in top3[mask]
                ])
            ),
            "top5_accuracy": float(
                np.mean([
                    idx in row
                    for row in top5[mask]
                ])
            ),
            "average_true_confidence": float(
                np.mean(target_probs[:, idx])
            ),
        }

    # ------------------------------------------------------------
    # 6. Save mapping/history/result
    # ------------------------------------------------------------
    OUTPUT_MAPPING.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_MAPPING,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                str(i): name
                for i, name in enumerate(class_names)
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    OUTPUT_HISTORY.write_text(
        json.dumps(
            {
                key: [float(x) for x in values]
                for key, values in history.history.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    best_epoch = int(
        np.argmin(history.history["val_loss"]) + 1
    )

    best_val_loss = float(
        min(history.history["val_loss"])
    )

    best_val_accuracy = float(
        max(history.history["val_accuracy"])
    )

    lines = [
        "=" * 80,
        "TARGETED 136-CLASS TRAINING RESULT",
        "=" * 80,
        "",
        f"Classes              : {num_classes}",
        f"Original train       : {len(y_train)}",
        f"Augmented train      : {len(y_train_aug)}",
        f"Validation samples   : {len(y_val)}",
        f"Test samples         : {len(y_test)}",
        "",
        f"Oversample factor    : {OVERSAMPLE_FACTOR}",
        f"Target weight factor : {TARGET_WEIGHT_MULTIPLIER}",
        "",
        f"Best epoch           : {best_epoch}",
        f"Best validation loss : {best_val_loss:.6f}",
        f"Best validation acc  : {best_val_accuracy * 100:.6f}%",
        "",
        f"Test loss            : {test_loss:.6f}",
        f"Test Top-1 accuracy  : {test_accuracy * 100:.6f}%",
        f"Test Top-3 accuracy  : {top3_accuracy * 100:.6f}%",
        f"Test Top-5 accuracy  : {top5_accuracy * 100:.6f}%",
        "",
        "TARGET CLASS RESULTS",
        "-" * 80,
    ]

    for name in sorted(target_results):
        r = target_results[name]
        lines.append(
            f"{name:20s} "
            f"n={r['test_samples']:2d} "
            f"Top1={r['top1_accuracy'] * 100:6.2f}% "
            f"Top3={r['top3_accuracy'] * 100:6.2f}% "
            f"Top5={r['top5_accuracy'] * 100:6.2f}% "
            f"TrueConf={r['average_true_confidence'] * 100:6.2f}%"
        )

    lines.extend([
        "",
        f"Model  : {OUTPUT_MODEL}",
        f"Mapping: {OUTPUT_MAPPING}",
        f"History: {OUTPUT_HISTORY}",
    ])

    OUTPUT_RESULT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
