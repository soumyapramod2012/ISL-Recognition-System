import numpy as np

from src.training.sequence_generator import SequenceGenerator

data = np.load(
    r"dataset/processed/landmarks/61. Summer/MVI_4565.npy"
)

print("Original :", data.shape)

generator = SequenceGenerator()

sequence = generator.generate(data)

print("Generated :", sequence.shape)