import numpy as np

from tensorflow.keras.models import load_model


class Evaluator:

    def __init__(self, model_path):

        self.model = load_model(model_path)

    def predict(self, X):

        probabilities = self.model.predict(
            X,
            verbose=0,
        )

        predictions = np.argmax(
            probabilities,
            axis=1,
        )

        return predictions