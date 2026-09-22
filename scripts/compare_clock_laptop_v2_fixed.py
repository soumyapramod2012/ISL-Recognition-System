from pathlib import Path
import sys
import json
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.hand_landmark_interpolator import HandLandmarkInterpolator
from src.training.landmark_normalizer import LandmarkNormalizer
from src.training.sequence_generator import SequenceGenerator

MODEL_PATH = (
    PROJECT_ROOT / "saved_models" /
    "isl_lstm_class_weighted_legacy_compatible.h5"
)

CLOCK_DIR = (
    PROJECT_ROOT / "dataset" / "raw" / "INCLUDE" /
    "Electronics" / "51. Clock"
)

LAPTOP_DIR = (
    PROJECT_ROOT / "dataset" / "raw" / "INCLUDE" /
    "Electronics" / "56. Laptop"
)

SPLIT_PATH = PROJECT_ROOT / "dataset" / "split_legacy.json"

LIVE_CLOCK = (
    PROJECT_ROOT / "outputs" / "webcam_recordings" /
    "clock_live_landmarks.npy"
)

OUTPUT = (
    PROJECT_ROOT / "outputs" /
    "clock_laptop_comparison_v2.txt"
)

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv"}


class Extractor:
    def __init__(self):
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def hand_points(hand):
        if hand is None:
            return [0.0] * 63

        values = []
        for lm in hand.landmark:
            values.extend([lm.x, lm.y, lm.z])
        return values

    def extract(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)

        values = []

        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                values.extend([
                    lm.x, lm.y, lm.z, lm.visibility
                ])
        else:
            values.extend([0.0] * 132)

        values.extend(self.hand_points(results.left_hand_landmarks))
        values.extend(self.hand_points(results.right_hand_landmarks))

        return np.asarray(values, dtype=np.float32)

    def close(self):
        self.holistic.close()


def extract_sequence(path, extractor, interpolator, normalizer, generator):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(extractor.extract(frame))

    cap.release()

    landmarks = np.asarray(frames, dtype=np.float32)
    interpolated = interpolator.interpolate(landmarks)
    normalized = normalizer.normalize(interpolated)
    sequence = generator.generate(normalized)

    return landmarks, sequence


def sequence_from_landmarks(path, interpolator, normalizer, generator):
    landmarks = np.load(path).astype(np.float32)
    interpolated = interpolator.interpolate(landmarks)
    normalized = normalizer.normalize(interpolated)
    sequence = generator.generate(normalized)
    return landmarks, sequence


def predict_sequence(sequence, model):
    probabilities = model.predict(
        np.expand_dims(sequence, axis=0),
        verbose=0,
    )[0]
    return probabilities


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def relpath_from_split(p):
    return PROJECT_ROOT / Path(p.replace("\\", "/"))


def main():
    labels = sorted(
        p.name
        for p in (
            PROJECT_ROOT / "dataset" /
            "processed" / "landmarks_legacy"
        ).iterdir()
        if p.is_dir() and p.name != "Extra"
    )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)

    extractor = Extractor()
    interpolator = HandLandmarkInterpolator(max_gap=5)
    normalizer = LandmarkNormalizer()
    generator = SequenceGenerator()

    # Exact Clock training samples from split_legacy.json
    clock_train_paths = [
        relpath_from_split(p)
        for p in split["train"]
        if "\\51. Clock\\" in p
    ]

    laptop_train_paths = [
        relpath_from_split(p)
        for p in split["train"]
        if "\\56. Laptop\\" in p
    ]

    print(f"Clock training samples: {len(clock_train_paths)}")
    print(f"Laptop training samples: {len(laptop_train_paths)}")

    cache = {}

    def get_training_sequence(path):
        key = str(path)
        if key not in cache:
            _, cache[key] = sequence_from_landmarks(
                path,
                interpolator,
                normalizer,
                generator,
            )
        return cache[key]

    clock_train_sequences = {
        p.name: get_training_sequence(p)
        for p in clock_train_paths
    }

    laptop_train_sequences = {
        p.name: get_training_sequence(p)
        for p in laptop_train_paths
    }

    validation_clock_names = [
        "MVI_4959.MOV",
        "MVI_4960.MOV",
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("CLOCK vs LAPTOP DIAGNOSTIC V2\n")
        f.write("=" * 80 + "\n\n")

        for name in validation_clock_names:
            path = CLOCK_DIR / name
            _, seq = extract_sequence(
                path,
                extractor,
                interpolator,
                normalizer,
                generator,
            )
            probs = predict_sequence(seq, model)

            f.write(f"{name}\n")
            f.write(f"Model top prediction: {labels[np.argmax(probs)]}\n")
            f.write(f"Model confidence: {np.max(probs):.6f}\n\n")

            clock_distances = sorted(
                (
                    (sample, mae(seq, train_seq))
                    for sample, train_seq
                    in clock_train_sequences.items()
                ),
                key=lambda x: x[1],
            )

            laptop_distances = sorted(
                (
                    (sample, mae(seq, train_seq))
                    for sample, train_seq
                    in laptop_train_sequences.items()
                ),
                key=lambda x: x[1],
            )

            f.write("Nearest CLOCK training samples:\n")
            for rank, (sample, distance) in enumerate(
                clock_distances, start=1
            ):
                f.write(f"{rank:02d}. {sample:15s} {distance:.6f}\n")

            f.write("\nNearest LAPTOP training samples:\n")
            for rank, (sample, distance) in enumerate(
                laptop_distances, start=1
            ):
                f.write(f"{rank:02d}. {sample:15s} {distance:.6f}\n")

            f.write("\n")

        # Optional live Clock comparison
        if LIVE_CLOCK.exists():
            landmarks, live_seq = sequence_from_landmarks(
                LIVE_CLOCK,
                interpolator,
                normalizer,
                generator,
            )

            probs = predict_sequence(live_seq, model)

            f.write("\nLIVE CLOCK COMPARISON\n")
            f.write("=" * 80 + "\n")
            f.write(f"Frames: {landmarks.shape[0]}\n")
            f.write(
                f"Model top prediction: "
                f"{labels[np.argmax(probs)]}\n"
            )
            f.write(
                f"Model confidence: "
                f"{np.max(probs):.6f}\n\n"
            )

            clock_distances = sorted(
                (
                    (sample, mae(live_seq, train_seq))
                    for sample, train_seq
                    in clock_train_sequences.items()
                ),
                key=lambda x: x[1],
            )

            laptop_distances = sorted(
                (
                    (sample, mae(live_seq, train_seq))
                    for sample, train_seq
                    in laptop_train_sequences.items()
                ),
                key=lambda x: x[1],
            )

            f.write("Nearest CLOCK training samples:\n")
            for rank, (sample, distance) in enumerate(
                clock_distances, start=1
            ):
                f.write(f"{rank:02d}. {sample:15s} {distance:.6f}\n")

            f.write("\nNearest LAPTOP training samples:\n")
            for rank, (sample, distance) in enumerate(
                laptop_distances, start=1
            ):
                f.write(f"{rank:02d}. {sample:15s} {distance:.6f}\n")
        else:
            f.write(
                "\nLIVE CLOCK FILE NOT FOUND:\n"
                f"{LIVE_CLOCK}\n"
            )

    extractor.close()

    print()
    print("=" * 80)
    print("Saved:")
    print(OUTPUT)
    print("=" * 80)


if __name__ == "__main__":
    main()