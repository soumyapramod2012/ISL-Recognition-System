import numpy as np

from src.training.config import SEQUENCE_LENGTH


class SequenceGenerator:

    def __init__(self):

        self.sequence_length = SEQUENCE_LENGTH

    def generate(self, landmarks):

        total_frames = len(landmarks)

        if total_frames >= self.sequence_length:

            indices = np.linspace(
                0,
                total_frames - 1,
                self.sequence_length,
                dtype=int,
            )

            return landmarks[indices]

        sequence = np.zeros(
            (
                self.sequence_length,
                landmarks.shape[1],
            ),
            dtype=np.float32,
        )

        sequence[:total_frames] = landmarks

        return sequence