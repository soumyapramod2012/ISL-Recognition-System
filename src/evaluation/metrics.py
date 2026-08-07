import json
import os

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


class Metrics:

    def __init__(self, output_dir):

        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

    def evaluate(
        self,
        y_true,
        y_pred,
        class_names,
    ):

        # ---------------- Accuracy ----------------

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        # ---------------- Classification Report ----------------

        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )

        with open(
            os.path.join(
                self.output_dir,
                "classification_report.txt",
            ),
            "w",
            encoding="utf-8",
        ) as file:

            file.write(report)

        # ---------------- Confusion Matrix ----------------

        cm = confusion_matrix(
            y_true,
            y_pred,
        )

        df = pd.DataFrame(
            cm,
            index=class_names,
            columns=class_names,
        )

        df.to_csv(
            os.path.join(
                self.output_dir,
                "confusion_matrix.csv",
            )
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names,
        )

        fig, ax = plt.subplots(figsize=(8, 8))

        disp.plot(
            ax=ax,
            cmap="Blues",
            colorbar=False,
        )

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_dir,
                "confusion_matrix.png",
            )
        )

        plt.close()

        # ---------------- Evaluation JSON ----------------

        summary = {

            "accuracy": float(accuracy),

            "samples": len(y_true),

            "classes": len(class_names),

        }

        with open(
            os.path.join(
                self.output_dir,
                "evaluation.json",
            ),
            "w",
        ) as file:

            json.dump(
                summary,
                file,
                indent=4,
            )

        return accuracy