from src.training.dataset_loader import DatasetLoader
from src.training.label_encoder import LabelEncoder

loader = DatasetLoader(
    "dataset/processed/landmarks"
)

samples = loader.load()

encoder = LabelEncoder()

encoder.fit(samples)

print()

print("Total Classes :", len(encoder.label_to_index))

print()

for label, index in encoder.label_to_index.items():

    print(f"{index:2d} -> {label}")

encoder.save(
    "outputs/label_mapping.json"
)

print()

print("Saved to outputs/label_mapping.json")