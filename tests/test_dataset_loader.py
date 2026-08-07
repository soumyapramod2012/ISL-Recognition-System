from src.training.dataset_loader import DatasetLoader

loader = DatasetLoader(
    "dataset/processed/landmarks"
)

samples = loader.load()

print("Samples :", len(samples))

print()

for sample in samples[:10]:

    print(sample["label"])

    print(sample["path"])

    print()