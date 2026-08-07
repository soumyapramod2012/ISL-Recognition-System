import random
import numpy as np
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

from src.utils.experiment_logger import ExperimentLogger
from src.training.train_dataset import TrainDataset
from src.training.model import ISLModel
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import Metrics

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

'''X_train, X_test, y_train, y_test, encoder = dataset.build()'''

num_classes = len(encoder.label_to_index)

model = ISLModel().build(num_classes)


from src.utils.experiment_manager import ExperimentManager
from src.utils.training_utils import TrainingUtils
import src.training.config as config

manager = ExperimentManager(config)

TrainingUtils.create_directories(config)

TrainingUtils.save_model_summary(
    model,
    config.MODEL_SUMMARY,
)

TrainingUtils.save_model_plot(
    model,
    config.MODEL_PLOT,
)

manager.save_config()

manager.save_dataset_info(
    encoder,
    X_train,
    X_val,
    X_test,
)

model.summary()

from src.training.trainer import Trainer

trainer = Trainer(config)

history = trainer.train(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
)


TrainingUtils.save_history(
    history,
    config,
)

TrainingUtils.plot_accuracy(
    history,
    config,
)

TrainingUtils.plot_loss(
    history,
    config,
)


loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1,
)

print()
print("=" * 50)
print("FINAL TEST RESULTS")
print("=" * 50)
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")
print("=" * 50)


# ---------------- Evaluation ----------------

evaluator = Evaluator(model)

y_pred = evaluator.predict(X_test)

metrics = Metrics("outputs")

metrics.evaluate(
    y_true=y_test,
    y_pred=y_pred,
    class_names=[
        encoder.decode(i)
        for i in range(num_classes)
    ],
)

logger = ExperimentLogger(
    config.EXPERIMENT_LOG
)

logger.log(
    version=config.PROJECT_VERSION,
    model_name=model.name,
    accuracy=accuracy,
    loss=loss,
    parameters=model.count_params(),
)