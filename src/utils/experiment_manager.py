import json
import os
from pathlib import Path


class ExperimentManager:

    def __init__(self, config):

        self.config = config

        os.makedirs(config.MODEL_DIR, exist_ok=True)

    def save_config(self):

        data = {

            "sequence_length": self.config.SEQUENCE_LENGTH,
            "landmark_size": self.config.LANDMARK_SIZE,

            "lstm_units_1": self.config.LSTM_UNITS_1,
            "lstm_units_2": self.config.LSTM_UNITS_2,

            "dense_units": self.config.DENSE_UNITS,

            "dropout_1": self.config.DROPOUT_1,
            "dropout_2": self.config.DROPOUT_2,
            "dropout_3": self.config.DROPOUT_3,

            "batch_size": self.config.BATCH_SIZE,
            "epochs": self.config.EPOCHS,

            "learning_rate": self.config.LEARNING_RATE,
        }

        with open(
            Path(self.config.MODEL_DIR) / "config_snapshot.json",
            "w",
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
            )

    def save_dataset_info(
        self,
        encoder,
        X_train,
        X_val,
        X_test,
    ):

        data = {

            "classes": len(encoder.label_to_index),

            "train_samples": len(X_train),

            "validation_samples": len(X_val),

            "test_samples": len(X_test),

            "sequence_length": self.config.SEQUENCE_LENGTH,

            "feature_size": self.config.LANDMARK_SIZE,
        }

        with open(
            Path(self.config.MODEL_DIR) / "dataset_info.json",
            "w",
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
            )