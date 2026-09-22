import numpy as np
from pathlib import Path

from src.training.hand_landmark_interpolator import HandLandmarkInterpolator
from src.training.landmark_normalizer import LandmarkNormalizer
from src.training.sequence_generator import SequenceGenerator


USER_FILE = Path(
    r"outputs\webcam_recordings\winter_legacy_normalized_60.npy"
)

WINTER_DIR = Path(
    r"dataset\processed\landmarks\63. Winter"
)


# Load user's normalized 60-frame sequence
user_sequence = np.load(USER_FILE)


# Training preprocessing
interpolator = HandLandmarkInterpolator(max_gap=5)
normalizer = LandmarkNormalizer()
generator = SequenceGenerator()


results = []


for file in sorted(WINTER_DIR.glob("*.npy")):

    data = np.load(file)

    data = interpolator.interpolate(data)
    data = normalizer.normalize(data)
    data = generator.generate(data)

    # Mean Euclidean distance across all 60 frames/features
    distance = np.mean(
        np.linalg.norm(user_sequence - data, axis=1)
    )

    results.append((file.name, distance))


results.sort(key=lambda x: x[1])


print()
print("=== USER vs WINTER TRAINING SEQUENCES ===")
print()

for name, distance in results:
    print(f"{name:15s} {distance:.6f}")

print()
print("Closest Winter sample:")
print(results[0][0], f"{results[0][1]:.6f}")

print()
print("Average distance to Winter:")
print(
    f"{np.mean([d for _, d in results]):.6f}"
)