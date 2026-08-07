import numpy as np


class Evaluator:

    def __init__(self, model):

        self.model = model

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