"""
Prepare a generalized legacy-landmark dataset for the ISL project.

This script:
1. Combines the INCLUDE legacy landmarks with the existing 24-class
   legacy landmarks.
2. Keeps the existing class folder names as the canonical labels.
3. Prefixes copied filenames with their source to avoid collisions.
4. Creates a stratified train/validation/test split:
      train = 60%
      validation = 20%
      test = 20%
   per class, using RANDOM_SEED=42.
5. Does NOT modify the existing datasets or split files.

Run from the project root:
    venv_mp_legacy\Scripts\activate
    python src\preprocessing\prepare_generalized_legacy_dataset.py
"""

import json
import random
import shutil
from pathlib import Path

import numpy as np


INCLUDE_ROOT = Path("dataset/processed/include_landmarks_legacy")
EXISTING_ROOT = Path("dataset/processed/landmarks_legacy")

OUTPUT_ROOT = Path("dataset/processed/generalized_landmarks_legacy")
SPLIT_FILE = Path("dataset/split_generalized_legacy.json")
LABEL_MAPPING_FILE = Path("outputs/generalized_label_mapping.json")

RANDOM_SEED = 42
EXPECTED_FEATURES = 258


# These are the 24 classes already present in the existing model.
# INCLUDE uses the same folder names for these semantic classes.
EXISTING_24 = {
    "1. Dog",
    "2. Cat",
    "3. Fish",
    "4. Bird",
    "5. Cow",
    "51. Clock",
    "52. Lamp",
    "53. Fan",
    "54. Cell phone",
    "55. Computer",
    "56. Laptop",
    "57. Screen",
    "58. Camera",
    "59. Television",
    "6. Mouse",
    "60. Radio",
    "61. Summer",
    "62. Spring",
    "63. Winter",
    "64. Fall",
    "65. Season",
    "7. Horse",
    "8. Animal",
    "Ex. Monsoon",
}


def validate_npy(path):
    """Validate a landmark file before copying it."""
    data = np.load(path, mmap_mode="r")

    if data.ndim != 2:
        raise ValueError(f"{path}: expected 2D array, found {data.shape}")

    if data.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"{path}: expected {EXPECTED_FEATURES} features, "
            f"found {data.shape[1]}"
        )

    if data.shape[0] == 0:
        raise ValueError(f"{path}: empty sequence")

    # mmap_mode supports these checks without unnecessarily loading
    # the whole array into RAM.
    if np.isnan(data).any():
        raise ValueError(f"{path}: contains NaN")

    if np.isinf(data).any():
        raise ValueError(f"{path}: contains Inf")


def collect_source(root, source_name):
    """Return {label: [npy paths]} from one processed dataset root."""
    result = {}

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue

        label = label_dir.name

        # 'Extra' was an intermediate raw-dataset folder and must never
        # become a class.
        if label == "Extra":
            continue

        files = sorted(label_dir.glob("*.npy"))
        if not files:
            continue

        result.setdefault(label, []).extend(
            (source_name, path) for path in files
        )

    return result


def copy_dataset(include_data, existing_data):
    """Copy both sources into one canonical class-folder structure."""
    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"{OUTPUT_ROOT} already exists.\n"
            "Delete it manually only if you intentionally want to rebuild."
        )

    OUTPUT_ROOT.mkdir(parents=True)

    combined = {}

    for label, items in include_data.items():
        combined.setdefault(label, []).extend(items)

    for label, items in existing_data.items():
        combined.setdefault(label, []).extend(items)

    for label in sorted(combined):
        output_dir = OUTPUT_ROOT / label
        output_dir.mkdir(parents=True, exist_ok=True)

        for source_name, source_path in combined[label]:
            validate_npy(source_path)

            destination = (
                output_dir
                / f"{source_name}__{source_path.name}"
            )

            shutil.copy2(source_path, destination)

    return combined


def make_split(combined):
    """
    Stratified split by class.

    The split is deterministic and performed after combining both sources.
    """
    rng = random.Random(RANDOM_SEED)

    train = []
    validation = []
    test = []

    for label in sorted(combined):
        paths = sorted(
            OUTPUT_ROOT / label / f"{source_name}__{file.name}"
            for source_name, file in combined[label]
        )

        rng.shuffle(paths)

        n = len(paths)

        if n < 3:
            raise ValueError(
                f"Class '{label}' has only {n} samples; "
                "cannot create train/validation/test splits."
            )

        # 20% test, 20% validation, remainder train.
        n_test = max(1, round(n * 0.20))
        n_val = max(1, round(n * 0.20))

        # Keep at least one training sample.
        if n_test + n_val >= n:
            n_test = 1
            n_val = 1

        test_paths = paths[:n_test]
        val_paths = paths[n_test:n_test + n_val]
        train_paths = paths[n_test + n_val:]

        if not train_paths:
            raise ValueError(
                f"Class '{label}' has no training samples after split."
            )

        train.extend(str(p).replace("\\", "/") for p in train_paths)
        validation.extend(str(p).replace("\\", "/") for p in val_paths)
        test.extend(str(p).replace("\\", "/") for p in test_paths)

    # Shuffle each split globally while keeping class stratification intact.
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)

    split = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(SPLIT_FILE, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    return split


def create_label_mapping(split):
    labels = sorted(
        {
            Path(path).parent.name
            for paths in split.values()
            for path in paths
        }
    )

    label_to_index = {
        label: index
        for index, label in enumerate(labels)
    }

    LABEL_MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(LABEL_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(label_to_index, f, indent=4)

    return labels, label_to_index


def print_summary(combined, split, labels):
    print()
    print("=" * 70)
    print("GENERALIZED LEGACY DATASET")
    print("=" * 70)

    print(f"Classes              : {len(labels)}")
    print(f"Total samples        : {sum(len(v) for v in combined.values())}")
    print(f"Train samples        : {len(split['train'])}")
    print(f"Validation samples   : {len(split['validation'])}")
    print(f"Test samples         : {len(split['test'])}")

    print()
    print("Class counts")
    print("-" * 70)

    for label in labels:
        total = len(combined[label])
        train_count = sum(
            1 for p in split["train"]
            if Path(p).parent.name == label
        )
        val_count = sum(
            1 for p in split["validation"]
            if Path(p).parent.name == label
        )
        test_count = sum(
            1 for p in split["test"]
            if Path(p).parent.name == label
        )

        print(
            f"{label:30s} "
            f"total={total:3d}  "
            f"train={train_count:3d}  "
            f"val={val_count:3d}  "
            f"test={test_count:3d}"
        )

    print()
    print(f"Dataset : {OUTPUT_ROOT}")
    print(f"Split   : {SPLIT_FILE}")
    print(f"Labels  : {LABEL_MAPPING_FILE}")
    print("=" * 70)


def main():
    include_data = collect_source(
        INCLUDE_ROOT,
        "include",
    )

    existing_data = collect_source(
        EXISTING_ROOT,
        "existing24",
    )

    include_labels = set(include_data)
    existing_labels = set(existing_data)

    missing_existing = sorted(
        label for label in EXISTING_24
        if label not in include_labels
    )

    if missing_existing:
        raise ValueError(
            "Expected existing 24 classes were not found in INCLUDE:\n"
            + "\n".join(missing_existing)
        )

    print(f"INCLUDE classes : {len(include_labels)}")
    print(f"Existing classes: {len(existing_labels)}")
    print(
        "INCLUDE/existing semantic overlap:",
        len(include_labels & existing_labels),
    )

    combined = copy_dataset(
        include_data,
        existing_data,
    )

    split = make_split(combined)

    labels, _ = create_label_mapping(split)

    print_summary(
        combined,
        split,
        labels,
    )


if __name__ == "__main__":
    main()