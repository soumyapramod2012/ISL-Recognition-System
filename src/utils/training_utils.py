import json
import os

import matplotlib.pyplot as plt

from tensorflow.keras.utils import plot_model


class TrainingUtils:

    @staticmethod
    def create_directories(config):

        os.makedirs(config.MODEL_DIR, exist_ok=True)
        os.makedirs(config.TENSORBOARD_LOGS, exist_ok=True)

    @staticmethod
    def save_model_summary(model, filepath):

        with open(filepath, "w", encoding="utf-8") as f:
            model.summary(print_fn=lambda x: f.write(x + "\n"))

    @staticmethod
    def save_model_plot(model, filepath):

        try:
            plot_model(
                model,
                to_file=filepath,
                show_shapes=True,
                show_layer_names=True,
            )

        except Exception as e:

            print("Unable to generate model diagram.")
            print(e)

    @staticmethod
    def save_history(history, config):

        with open(config.TRAINING_HISTORY, "w") as f:

            json.dump(history.history, f, indent=4)

    @staticmethod
    def plot_accuracy(history, config):

        plt.figure(figsize=(8, 5))

        plt.plot(history.history["accuracy"])

        plt.plot(history.history["val_accuracy"])

        plt.title("Training Accuracy")

        plt.xlabel("Epoch")

        plt.ylabel("Accuracy")

        plt.legend(["Train", "Validation"])

        plt.grid(True)

        plt.savefig(config.ACCURACY_PLOT)

        plt.close()

    @staticmethod
    def plot_loss(history, config):

        plt.figure(figsize=(8, 5))

        plt.plot(history.history["loss"])

        plt.plot(history.history["val_loss"])

        plt.title("Training Loss")

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.legend(["Train", "Validation"])

        plt.grid(True)

        plt.savefig(config.LOSS_PLOT)

        plt.close()