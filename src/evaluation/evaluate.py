from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import Metrics

from src.training.train_dataset import TrainDataset

import src.training.config as config


# -----------------------------
# Load Dataset
# -----------------------------

dataset = TrainDataset(
    "dataset/processed/landmarks"
)

(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    encoder,
) = dataset.build()


# -----------------------------
# Load Best Model
# -----------------------------

evaluator = Evaluator(
    config.BEST_MODEL
)

predictions = evaluator.predict(
    X_test
)


# -----------------------------
# Evaluate
# -----------------------------

metrics = Metrics(
    "outputs"
)

accuracy = metrics.evaluate(

    y_true=y_test,

    y_pred=predictions,

    class_names=encoder.classes,

)

print()
print("=" * 50)
print("EVALUATION COMPLETE")
print("=" * 50)
print(f"Test Accuracy : {accuracy:.4f}")
print("=" * 50)