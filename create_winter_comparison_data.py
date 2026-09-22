import numpy as np

from src.training.hand_landmark_interpolator import HandLandmarkInterpolator
from src.training.landmark_normalizer import LandmarkNormalizer
from src.training.sequence_generator import SequenceGenerator


input_path = r"outputs\webcam_recordings\winter_legacy_mp_0.10.21_landmarks.npy"
output_path = r"outputs\webcam_recordings\winter_legacy_normalized_60.npy"


# Load landmarks extracted with MediaPipe 0.10.21
landmarks = np.load(input_path)

print("Original shape:", landmarks.shape)


# 1. Interpolate short missing-hand gaps
interpolator = HandLandmarkInterpolator(max_gap=5)
landmarks = interpolator.interpolate(landmarks)

print("After interpolation:", landmarks.shape)


# 2. Apply the exact training normalization
normalizer = LandmarkNormalizer()
landmarks = normalizer.normalize(landmarks)

print("After normalization:", landmarks.shape)


# 3. Generate the 60-frame model sequence
generator = SequenceGenerator()
sequence = generator.generate(landmarks)

print("Final sequence:", sequence.shape)


# Save
np.save(output_path, sequence)

print()
print("Saved:")
print(output_path)