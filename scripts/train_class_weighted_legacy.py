import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.training.config as config
from src.training.train_dataset import TrainDataset
from src.training.model import ISLModel


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATASET_PATH = "dataset/processed/landmarks_legacy"
SPLIT_PATH = Path("dataset/split_legacy.json")

OUTPUT_MODEL = "saved_models/isl_lstm_class_weighted_legacy.keras"
OUTPUT_LOG = Path("outputs/class_weighted_legacy_results.txt")


def main():
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
        encoder.decode(i) for i in range(num_classes)
    ]

    # Balanced class weights: inverse frequency, normalized around 1.
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights = counts.sum() / (num_classes * counts)

    print("\nClass weights:")
    for i, name in enumerate(class_names):
        print(f"{name:15s}: {class_weights[i]:.4f}  (n={int(counts[i])})")

    model = ISLModel().build(num_classes)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            OUTPUT_MODEL,
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
            patience=5,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight={
            i: float(class_weights[i])
            for i in range(num_classes)
        },
        callbacks=callbacks,
        verbose=1,
    )

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0,
    )

    probabilities = model.predict(X_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    print("\n" + "=" * 60)
    print("CLASS-WEIGHTED LEGACY EXPERIMENT")
    print("=" * 60)
    print("Legacy baseline accuracy : 60.49%")
    print(f"Weighted model accuracy : {accuracy * 100:.2f}%")
    print(f"Weighted model loss     : {loss:.4f}")
    print(f"Change                  : {(accuracy - 0.6049) * 100:+.2f} pp")
    print("=" * 60)

    lines = [
        "CLASS-WEIGHTED LEGACY EXPERIMENT",
        "Legacy baseline accuracy: 60.49%",
        f"Weighted model accuracy: {accuracy * 100:.2f}%",
        f"Weighted model loss: {loss:.4f}",
        f"Change: {(accuracy - 0.6049) * 100:+.2f} pp",
        "",
        "Class accuracy:",
    ]

    print("\nClass accuracy:")
    for idx, name in enumerate(class_names):
        mask = y_test == idx
        if np.any(mask):
            class_acc = float(
                np.mean(predictions[mask] == y_test[mask])
            )
            line = f"{name:15s}: {class_acc * 100:6.2f}%"
            print(line)
            lines.append(line)

    OUTPUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_LOG.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"\nSaved model : {OUTPUT_MODEL}")
    print(f"Saved log   : {OUTPUT_LOG}")


if __name__ == "__main__":
    main()
