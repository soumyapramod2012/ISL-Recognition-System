import json
from pathlib import Path
import csv
import os
from datetime import datetime


class ExperimentLogger:

    def __init__(
            self,
            csv_path,
            experiments_dir, 
        ):

        self.filepath = csv_path
        self.experiments_dir = Path(experiments_dir)

        os.makedirs(
            os.path.dirname(csv_path),
            exist_ok=True,
        )

        if not os.path.exists(self.filepath):

            with open(
                self.filepath,
                "w",
                newline="",
                encoding="utf-8",
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    "Timestamp",
                    "Version",
                    "Model",
                    "Accuracy",
                    "Loss",
                    "Parameters",
                    "Best Epoch",
                    "Best Validation Accuracy",
                    "Epochs Trained",
                ])

    def log(
        self,
        version,
        model_name,
        accuracy,
        loss,
        parameters,
        best_epoch,
        best_val_accuracy,
        epochs_trained,
    ):
        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # ---------------- CSV ----------------

        with open(
            self.filepath,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                timestamp,
                version,
                model_name,
                f"{accuracy:.4f}",
                f"{loss:.4f}",
                parameters,
                best_epoch,
                f"{best_val_accuracy:.4f}",
                epochs_trained,
            ])

        # ---------------- JSON ----------------

        experiment = {
            "timestamp": timestamp,
            "version": version,
            "model": model_name,
            "accuracy": float(accuracy),
            "loss": float(loss),
            "parameters": int(parameters),
            "best_epoch": int(best_epoch),
            "best_validation_accuracy": float(best_val_accuracy),
            "epochs_trained": int(epochs_trained),
        }

        self.experiments_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        safe_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = self.experiments_dir / (
            f"{version}_{safe_timestamp}.json"
        )

        with open(
            filename,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                experiment,
                file,
                indent=4,
            )