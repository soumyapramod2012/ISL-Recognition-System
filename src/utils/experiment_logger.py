import csv
import os
from datetime import datetime


class ExperimentLogger:

    def __init__(self, filepath):

        self.filepath = filepath

        os.makedirs(
            os.path.dirname(filepath),
            exist_ok=True,
        )

        if not os.path.exists(filepath):

            with open(
                filepath,
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
                ])

    def log(
        self,
        version,
        model_name,
        accuracy,
        loss,
        parameters,
    ):

        with open(
            self.filepath,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                version,
                model_name,
                f"{accuracy:.4f}",
                f"{loss:.4f}",
                parameters,
            ])