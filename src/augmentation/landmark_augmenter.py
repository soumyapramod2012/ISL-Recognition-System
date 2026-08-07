import numpy as np


class LandmarkAugmenter:

    def __init__(self, noise_std=0.005):

        self.noise_std = noise_std


    def augment(self, sequence):

        return self.add_gaussian_noise(sequence)
    

    def add_gaussian_noise(self, sequence):

        """
        sequence shape:
        (sequence_length, feature_size)

        Example:
        (60, 258)
        """

        noise = np.random.normal(
            loc=0.0,
            scale=self.noise_std,
            size=sequence.shape,
        )

        augmented = sequence + noise

        return augmented.astype(np.float32)