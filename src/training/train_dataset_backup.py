import numpy as np
from src.training.landmark_normalizer import LandmarkNormalizer
from src.training.hand_landmark_interpolator import (
    HandLandmarkInterpolator
)

from sklearn.model_selection import train_test_split

from src.training.config import (
    TEST_SIZE,
    VALIDATION_SIZE,
    RANDOM_STATE,
    LANDMARK_SIZE,
)

from src.training.dataset_loader import DatasetLoader
from src.training.label_encoder import LabelEncoder
from src.training.sequence_generator import SequenceGenerator
#from src.augmentation.landmark_augmenter import LandmarkAugmenter


class TrainDataset:

    def __init__(self, dataset_path):

        self.loader = DatasetLoader(dataset_path)
        self.encoder = LabelEncoder()
        self.generator = SequenceGenerator()
        self.normalizer = LandmarkNormalizer()
        self.interpolator = HandLandmarkInterpolator(
            max_gap=5
        )
        
        '''self.augmenter = LandmarkAugmenter(
            noise_std=0.005,
        )'''


    def build(self):

        samples = self.loader.load()

        self.encoder.fit(samples)

        X = []
        y = []

        invalid_files = []

        for sample in samples:

            try:

                landmarks = np.load(sample["path"])

                if landmarks.ndim != 2:
                    invalid_files.append(
                        (sample["path"], "Not a 2D array")
                    )
                    continue

                if landmarks.shape[1] != LANDMARK_SIZE:
                    invalid_files.append(
                        (
                            sample["path"],
                            f"Expected {LANDMARK_SIZE} features, found {landmarks.shape[1]}"
                        )
                    )
                    continue

                if landmarks.shape[0] == 0:
                    invalid_files.append(
                        (
                            sample["path"],
                            "Empty sequence"
                        )
                    )
                    continue

                if np.isnan(landmarks).any():
                    invalid_files.append(
                        (
                            sample["path"],
                            "Contains NaN values"
                        )
                    )
                    continue

                if np.isinf(landmarks).any():
                    invalid_files.append(
                        (
                            sample["path"],
                            "Contains Infinite values"
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
                    self.encoder.encode(sample["label"])
                )

            except Exception as e:

                invalid_files.append(
                    (
                        sample["path"],
                        str(e)
                    )
                )

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        zero_frames = np.all(
            X == 0,
            axis=2,
        ).sum()

        print(
            f"Zero Frames          : {zero_frames}"
        )

        # First split: Train + Test

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        # Second split: Train + Validation

        X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
        )

        #original_train_samples = len(X_train)

        '''augmented_X = []
        augmented_y = []

        for sequence, label in zip(X_train, y_train):
            augmented_X.append(sequence)
            augmented_y.append(label)

            augmented_sequence = self.augmenter.augment(sequence)

            augmented_X.append(augmented_sequence)
            augmented_y.append(label)

        X_train = np.array(augmented_X, dtype=np.float32)
        y_train = np.array(augmented_y, dtype=np.int32)'''

        # Augmentation disabled (Baseline)
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)

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
        print(f"Classes                 : {len(self.encoder.label_to_index)}")
        print(f"Videos Found            : {len(samples)}")
        print(f"Valid Samples           : {len(X)}")
        print(f"Invalid Samples         : {len(invalid_files)}")
        print(f"Sequence Length         : {X.shape[1]}")
        print(f"Feature Size            : {X.shape[2]}")
        #print(f"Original Train Samples  : {original_train_samples}")
        #print(f"Augmented Train Samples : {len(X_train)}")
        print(f"Train Samples           : {len(X_train)}")
        print(f"Validation Samples      : {len(X_val)}")
        print(f"Test Samples            : {len(X_test)}")
        print("=" * 50)

        if invalid_files:

            print()
            print("INVALID FILES")
            print("-" * 50)

            for path, reason in invalid_files:

                print(path)
                print("Reason :", reason)
                print()

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            self.encoder,
            )
    