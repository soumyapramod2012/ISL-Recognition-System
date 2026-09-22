from pathlib import Path
import sys

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
    PROJECT_ROOT
    / "saved_models"
    / "isl_lstm_class_weighted_legacy_compatible.h5"
)

LANDMARK_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "landmarks_legacy"
)

CLOCK_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "INCLUDE"
    / "Electronics"
    / "51. Clock"
)

OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "clock_audit"
    / "clock_predictions.txt"
)


VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv"}


def load_labels():
    return sorted(
        p.name
        for p in LANDMARK_ROOT.iterdir()
        if p.is_dir() and p.name != "Extra"
    )


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
                    lm.x,
                    lm.y,
                    lm.z,
                    lm.visibility,
                ])
        else:
            values.extend([0.0] * 132)

        values.extend(
            self.hand_points(
                results.left_hand_landmarks
            )
        )

        values.extend(
            self.hand_points(
                results.right_hand_landmarks
            )
        )

        return np.asarray(
            values,
            dtype=np.float32,
        )

    def close(self):
        self.holistic.close()


def create_contact_sheet(videos, output_path):
    thumbnails = []

    for video in videos:
        cap = cv2.VideoCapture(str(video))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count <= 0:
            cap.release()
            continue

        sample_indices = np.linspace(
            0,
            frame_count - 1,
            5,
            dtype=int,
        )

        row = []

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()

            if not ok:
                continue

            frame = cv2.resize(frame, (320, 180))

            cv2.putText(
                frame,
                f"{video.name} #{idx + 1}",
                (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            row.append(frame)

        cap.release()

        if row:
            while len(row) < 5:
                row.append(np.zeros_like(row[0]))

            thumbnails.append(
                np.hstack(row)
            )

    if not thumbnails:
        return

    contact_sheet = np.vstack(thumbnails)

    cv2.imwrite(
        str(output_path),
        contact_sheet,
    )


def process_video(path, extractor, interpolator, normalizer, generator, model, labels):

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")

    frames = []

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frames.append(
            extractor.extract(frame)
        )

    cap.release()

    landmarks = np.asarray(
        frames,
        dtype=np.float32,
    )

    interpolated = interpolator.interpolate(
        landmarks
    )

    normalized = normalizer.normalize(
        interpolated
    )

    sequence = generator.generate(
        normalized
    )

    probabilities = model.predict(
        np.expand_dims(sequence, axis=0),
        verbose=0,
    )[0]

    top = np.argsort(probabilities)[::-1][:5]

    return landmarks.shape[0], [
        (
            labels[i],
            float(probabilities[i])
        )
        for i in top
    ]


def main():

    labels = load_labels()

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    extractor = Extractor()
    interpolator = HandLandmarkInterpolator(max_gap=5)
    normalizer = LandmarkNormalizer()
    generator = SequenceGenerator()

    videos = sorted(
        p for p in CLOCK_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    CONTACT_SHEET = (
        PROJECT_ROOT
        / "outputs"
        / "clock_audit"
        / "clock_contact_sheet.jpg"
    )

    CONTACT_SHEET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    create_contact_sheet(
        videos,
        CONTACT_SHEET,
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "CLOCK VIDEO MODEL AUDIT\n"
        )
        f.write("=" * 80 + "\n\n")

        for number, video in enumerate(
            videos,
            start=1,
        ):

            frames, top5 = process_video(
                video,
                extractor,
                interpolator,
                normalizer,
                generator,
                model,
                labels,
            )

            print()
            print(
                f"{number:02d}. {video.name} "
                f"({frames} frames)"
            )

            f.write(
                f"{video.name}\n"
            )
            f.write(
                f"Frames: {frames}\n"
            )

            for rank, (
                label,
                confidence,
            ) in enumerate(
                top5,
                start=1,
            ):

                print(
                    f"   {rank}. "
                    f"{label:20s} "
                    f"{confidence:.4f}"
                )

                f.write(
                    f"{rank}. "
                    f"{label}\t"
                    f"{confidence:.6f}\n"
                )

            f.write("\n")

    extractor.close()

    print()
    print("=" * 80)
    print("Saved:")
    print(OUTPUT)
    print("Contact sheet:")
    print(CONTACT_SHEET)
    print("=" * 80)


if __name__ == "__main__":
    main()