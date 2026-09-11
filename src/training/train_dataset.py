import json
from pathlib import Path

import numpy as np

from src.training.landmark_normalizer import LandmarkNormalizer
from src.training.hand_landmark_interpolator import (
    HandLandmarkInterpolator,
)

from src.training.config import LANDMARK_SIZE

from src.training.dataset_loader import DatasetLoader
from src.training.label_encoder import LabelEncoder
from src.training.sequence_generator import SequenceGenerator


class TrainDataset:

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

        self.loader = DatasetLoader(dataset_path)
        self.encoder = LabelEncoder()
        self.generator = SequenceGenerator()
        self.normalizer = LandmarkNormalizer()
        self.interpolator = HandLandmarkInterpolator(
            max_gap=5
        )

        self.split_file = Path("dataset/split.json")

    def _load_split(self):

        if not self.split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {self.split_file}"
            )

        with open(
            self.split_file,
            "r",
            encoding="utf-8",
        ) as f:

            split = json.load(f)

        required = ["train", "validation", "test"]

        for key in required:

            if key not in split:
                raise ValueError(
                    f"Missing split: {key}"
                )

        return split

    def _prepare_samples(self, paths):

        X = []
        y = []

        invalid_files = []

        for path in paths:

            path = Path(path)

            try:

                if not path.exists():
                    invalid_files.append(
                        (str(path), "File not found")
                    )
                    continue

                label = path.parent.name

                landmarks = np.load(path)

                if landmarks.ndim != 2:
                    invalid_files.append(
                        (
                            str(path),
                            "Not a 2D array",
                        )
                    )
                    continue

                if landmarks.shape[1] != LANDMARK_SIZE:
                    invalid_files.append(
                        (
                            str(path),
                            (
                                f"Expected {LANDMARK_SIZE} "
                                f"features, found "
                                f"{landmarks.shape[1]}"
                            ),
                        )
                    )
                    continue

                if landmarks.shape[0] == 0:
                    invalid_files.append(
                        (
                            str(path),
                            "Empty sequence",
                        )
                    )
                    continue

                if np.isnan(landmarks).any():
                    invalid_files.append(
                        (
                            str(path),
                            "Contains NaN values",
                        )
                    )
                    continue

                if np.isinf(landmarks).any():
                    invalid_files.append(
                        (
                            str(path),
                            "Contains Infinite values",
                        )
                    )
                    continue

                landmarks = self.interpolator.interpolate(
                    landmarks
                )

                landmarks = self.normalizer.normalize(
                    landmarks
                )

                sequence = self.generator.generate(
                    landmarks
                )

                X.append(sequence)

                y.append(
                    self.encoder.encode(label)
                )

            except Exception as e:

                invalid_files.append(
                    (
                        str(path),
                        str(e),
                    )
                )

        return (
            np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int32),
            invalid_files,
        )

    def build(self):

        split = self._load_split()

        # Collect every sample from the split file
        all_paths = (
            split["train"]
            + split["validation"]
            + split["test"]
        )

        # Create samples for LabelEncoder
        samples = [
            {
                "label": Path(path).parent.name,
                "path": Path(path),
            }
            for path in all_paths
        ]

        self.encoder.fit(samples)

        X_train, y_train, invalid_train = (
            self._prepare_samples(split["train"])
        )

        X_val, y_val, invalid_val = (
            self._prepare_samples(split["validation"])
        )

        X_test, y_test, invalid_test = (
            self._prepare_samples(split["test"])
        )

        invalid_files = (
            invalid_train
            + invalid_val
            + invalid_test
        )

        print()
        print("Dataset Shapes")
        print("-" * 50)
        print("X_train :", X_train.shape)
        print("X_val   :", X_val.shape)
        print("X_test  :", X_test.shape)

        print()
        print("=" * 50)
        print("DATASET SUMMARY")
        print("=" * 50)

        print(
            f"Classes                 : "
            f"{len(self.encoder.label_to_index)}"
        )

        print(
            f"Videos in Split         : "
            f"{len(all_paths)}"
        )

        print(
            f"Valid Samples           : "
            f"{len(X_train) + len(X_val) + len(X_test)}"
        )

        print(
            f"Invalid Samples         : "
            f"{len(invalid_files)}"
        )

        print(
            f"Sequence Length         : "
            f"{X_train.shape[1]}"
        )

        print(
            f"Feature Size            : "
            f"{X_train.shape[2]}"
        )

        print(
            f"Train Samples           : "
            f"{len(X_train)}"
        )

        print(
            f"Validation Samples      : "
            f"{len(X_val)}"
        )

        print(
            f"Test Samples            : "
            f"{len(X_test)}"
        )

        print("=" * 50)

        if invalid_files:

            print()
            print("INVALID FILES")
            print("-" * 50)

            for path, reason in invalid_files:

                print(path)
                print("Reason :", reason)
                print()

            raise RuntimeError(
                "Invalid files found. "
                "Training aborted."
            )

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            self.encoder,
        )