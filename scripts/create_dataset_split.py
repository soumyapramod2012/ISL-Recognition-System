from pathlib import Path
import json

from sklearn.model_selection import train_test_split


DATASET_PATH = Path("dataset/processed/landmarks")

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.25
RANDOM_STATE = 42


def main():
    files = sorted(
        file
        for file in DATASET_PATH.rglob("*.npy")
        if file.parent.name != "Extra"
    )

    if not files:
        raise RuntimeError("No landmark files found.")

    # Group files by class
    class_files = {}

    for file in files:
        label = file.parent.name
        class_files.setdefault(label, []).append(str(file))

    train_files = []
    val_files = []
    test_files = []

    for label, samples in sorted(class_files.items()):

        if len(samples) < 3:
            raise RuntimeError(
                f"Class '{label}' has too few samples: {len(samples)}"
            )

        # First: train+validation vs test
        train_val, test = train_test_split(
            samples,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        # Second: train vs validation
        train, val = train_test_split(
            train_val,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
        )

        train_files.extend(train)
        val_files.extend(val)
        test_files.extend(test)

    split = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "validation_size": VALIDATION_SIZE,
        "train": sorted(train_files),
        "validation": sorted(val_files),
        "test": sorted(test_files),
    }

    output = Path("dataset/split.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    print("=" * 60)
    print("DATASET SPLIT")
    print("=" * 60)

    print(f"Total      : {len(files)}")
    print(f"Train      : {len(train_files)}")
    print(f"Validation : {len(val_files)}")
    print(f"Test       : {len(test_files)}")

    print()
    print(f"Saved to   : {output}")


if __name__ == "__main__":
    main()