"""
Create a filtered generalized ISL legacy-landmark dataset.

Purpose:
- Start from the already validated 262-class generalized dataset.
- Keep every class with >= 20 samples.
- Always keep the original 24 classes, even if a class has <20 samples.
- Do NOT modify the existing generalized dataset, split, model, or realtime.py.
- Create a completely separate filtered dataset and split.

Run from the project root:
    venv_mp_legacy\Scripts\activate
    python scripts\create_filtered_generalized_dataset.py
"""

import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


SEED = 42
MIN_SAMPLES = 20

SOURCE_ROOT = Path("dataset/processed/generalized_landmarks_legacy")
OUTPUT_ROOT = Path("dataset/processed/generalized_filtered_legacy")

SPLIT_FILE = Path("dataset/split_generalized_filtered_legacy.json")
LABEL_MAPPING_FILE = Path("outputs/generalized_filtered_label_mapping.json")
REPORT_FILE = Path("outputs/generalized_filtered_dataset_report.txt")

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


def collect_classes():
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {SOURCE_ROOT}"
        )

    class_files = {}

    for class_dir in sorted(SOURCE_ROOT.iterdir()):
        if not class_dir.is_dir():
            continue

        files = sorted(class_dir.glob("*.npy"))
        if files:
            class_files[class_dir.name] = files

    return class_files


def select_classes(class_files):
    selected = {
        label
        for label, files in class_files.items()
        if len(files) >= MIN_SAMPLES
    }

    # The original 24-class vocabulary is mandatory.
    missing_original = sorted(
        label for label in EXISTING_24
        if label not in class_files
    )

    if missing_original:
        raise RuntimeError(
            "Original 24 classes missing from generalized dataset:\n"
            + "\n".join(missing_original)
        )

    selected.update(EXISTING_24)
    return selected


def validate_file(path):
    data = np.load(path, mmap_mode="r")

    if data.ndim != 2 or data.shape[1] != 258:
        raise ValueError(
            f"Invalid landmark shape: {path} -> {data.shape}"
        )

    if data.shape[0] == 0:
        raise ValueError(f"Empty landmark sequence: {path}")

    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError(f"NaN/Inf detected: {path}")


def create_dataset(class_files, selected):
    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"{OUTPUT_ROOT} already exists.\n"
            "Delete it manually only if you intentionally want to rebuild."
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    selected_counts = {}

    for label in sorted(selected):
        destination_dir = OUTPUT_ROOT / label
        destination_dir.mkdir(parents=True, exist_ok=True)

        files = class_files[label]
        selected_counts[label] = len(files)

        for source in files:
            validate_file(source)

            # Keep the already-prefixed filename from the generalized dataset.
            destination = destination_dir / source.name
            shutil.copy2(source, destination)

    return selected_counts


def make_split(selected_counts):
    rng = random.Random(SEED)

    train = []
    validation = []
    test = []

    for label in sorted(selected_counts):
        files = sorted((OUTPUT_ROOT / label).glob("*.npy"))

        rng.shuffle(files)

        n = len(files)

        if n < 3:
            raise RuntimeError(
                f"Class '{label}' has only {n} samples."
            )

        n_test = max(1, round(n * 0.20))
        n_val = max(1, round(n * 0.20))

        if n_test + n_val >= n:
            n_test = 1
            n_val = 1

        test_files = files[:n_test]
        val_files = files[n_test:n_test + n_val]
        train_files = files[n_test + n_val:]

        if not train_files:
            raise RuntimeError(
                f"Class '{label}' has no training samples."
            )

        train.extend(str(p).replace("\\", "/") for p in train_files)
        validation.extend(str(p).replace("\\", "/") for p in val_files)
        test.extend(str(p).replace("\\", "/") for p in test_files)

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


def create_mapping(split):
    labels = sorted(
        {
            Path(path).parent.name
            for paths in split.values()
            for path in paths
        }
    )

    mapping = {
        label: index
        for index, label in enumerate(labels)
    }

    LABEL_MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(LABEL_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4)

    return labels


def verify_split(split, labels):
    all_paths = [
        path
        for paths in split.values()
        for path in paths
    ]

    missing = [
        path
        for path in all_paths
        if not Path(path).exists()
    ]

    if missing:
        raise RuntimeError(
            f"{len(missing)} split paths do not exist. First missing:\n"
            + "\n".join(missing[:10])
        )

    split_counts = {
        key: Counter(Path(path).parent.name for path in paths)
        for key, paths in split.items()
    }

    # Every class must occur in all three splits.
    for label in labels:
        for split_name in ("train", "validation", "test"):
            if split_counts[split_name][label] == 0:
                raise RuntimeError(
                    f"Class '{label}' missing from {split_name} split."
                )


def write_report(source_classes, selected, selected_counts, split, labels):
    report = []

    report.append("FILTERED GENERALIZED LEGACY DATASET")
    report.append("=" * 70)
    report.append(f"Source classes                 : {len(source_classes)}")
    report.append(f"Selected classes               : {len(selected)}")
    report.append(f"Threshold                      : >= {MIN_SAMPLES} samples")
    report.append("Original 24 classes            : always retained")
    report.append("")
    report.append(f"Total samples                  : {sum(selected_counts.values())}")
    report.append(f"Train samples                  : {len(split['train'])}")
    report.append(f"Validation samples             : {len(split['validation'])}")
    report.append(f"Test samples                   : {len(split['test'])}")
    report.append("")
    report.append("Selected classes and counts")
    report.append("-" * 70)

    for label in labels:
        report.append(
            f"{label:35s} {selected_counts[label]:4d}"
        )

    report.append("")
    report.append(f"Dataset : {OUTPUT_ROOT}")
    report.append(f"Split   : {SPLIT_FILE}")
    report.append(f"Labels  : {LABEL_MAPPING_FILE}")

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("\n".join(report))


def main():
    class_files = collect_classes()

    selected = select_classes(class_files)

    print("=" * 70)
    print("FILTERED GENERALIZED DATASET")
    print("=" * 70)
    print(f"Source classes : {len(class_files)}")
    print(f"Selected       : {len(selected)}")
    print(f"Rule           : >= {MIN_SAMPLES} samples")
    print("Original 24    : ALWAYS retained")

    selected_counts = create_dataset(class_files, selected)
    split = make_split(selected_counts)
    labels = create_mapping(split)

    verify_split(split, labels)

    write_report(
        source_classes=class_files,
        selected=selected,
        selected_counts=selected_counts,
        split=split,
        labels=labels,
    )

    print("\nDataset creation and verification completed successfully.")


if __name__ == "__main__":
    main()