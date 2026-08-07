import numpy as np

from src.augmentation.landmark_augmenter import LandmarkAugmenter


sequence = np.load(
    "dataset/processed/landmarks/61. Summer/MVI_4565.npy"
)

augmenter = LandmarkAugmenter()

augmented = augmenter.add_gaussian_noise(sequence)

print("Original Shape :", sequence.shape)
print("Augmented Shape:", augmented.shape)

print()
print("First Landmark")

print("Original :", sequence[0][:10])
print("Augmented:", augmented[0][:10])