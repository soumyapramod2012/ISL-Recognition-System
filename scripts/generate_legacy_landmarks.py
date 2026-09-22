import cv2
import sys
import numpy as np

from pathlib import Path

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dataset_processor import DatasetProcessor
from src.preprocessing.legacy_landmark_extractor import (
    LegacyLandmarkExtractor,
)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

DATASET = Path("dataset/raw/INCLUDE")
OUTPUT = Path("dataset/processed/landmarks_legacy")


# ------------------------------------------------------------
# Intended labelled classes
# ------------------------------------------------------------

EXCLUDE_LABELS = {
    "Extra",
}


def main():

    print("=" * 70)
    print("LEGACY MEDIAPIPE LANDMARK GENERATION")
    print("=" * 70)

    processor = DatasetProcessor(str(DATASET))

    videos = processor.get_video_files()

    # Exclude the 5 unlabeled Extra files
    videos = [
        video
        for video in videos
        if video["label"] not in EXCLUDE_LABELS
    ]

    print(f"Videos Found : {len(videos)}")
    print(f"Output       : {OUTPUT}")
    print()

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    extractor = LegacyLandmarkExtractor()

    processed = 0
    skipped = 0
    failed = 0

    try:

        for index, video in enumerate(
            videos,
            start=1,
        ):

            label = video["label"]
            video_path = Path(video["path"])

            output_file = (
                OUTPUT
                / label
                / f"{video_path.stem}.npy"
            )

            # Skip if already generated
            if output_file.exists():

                print(
                    f"[{index}/{len(videos)}] "
                    f"SKIP - {label} - "
                    f"{video_path.name}"
                )

                skipped += 1
                continue

            print(
                f"[{index}/{len(videos)}] "
                f"{label} - "
                f"{video_path.name}"
            )

            cap = cv2.VideoCapture(
                str(video_path)
            )

            if not cap.isOpened():

                print(
                    "    ERROR: Could not open video"
                )

                failed += 1
                continue

            frames = []

            while True:

                success, frame = cap.read()

                if not success:
                    break

                vector = extractor.extract(frame)

                if vector.shape != (258,):

                    print(
                        f"    ERROR: Invalid landmark shape "
                        f"{vector.shape}"
                    )

                    frames = []
                    break

                frames.append(vector)

            cap.release()

            if not frames:

                print(
                    "    ERROR: No valid frames extracted"
                )

                failed += 1
                continue

            frames = np.asarray(
                frames,
                dtype=np.float32,
            )

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            if frames.ndim != 2:

                print(
                    f"    ERROR: Invalid array shape "
                    f"{frames.shape}"
                )

                failed += 1
                continue

            if frames.shape[1] != 258:

                print(
                    f"    ERROR: Expected 258 features, "
                    f"found {frames.shape[1]}"
                )

                failed += 1
                continue

            if np.isnan(frames).any():

                print(
                    "    ERROR: NaN values detected"
                )

                failed += 1
                continue

            if np.isinf(frames).any():

                print(
                    "    ERROR: Infinite values detected"
                )

                failed += 1
                continue

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            output_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.save(
                output_file,
                frames,
            )

            print(
                f"    Saved: {output_file} "
                f"({frames.shape[0]} frames)"
            )

            processed += 1

    finally:

        extractor.close()

    print()
    print("=" * 70)
    print("LEGACY MEDIAPIPE PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Videos Found : {len(videos)}")
    print(f"Processed    : {processed}")
    print(f"Skipped      : {skipped}")
    print(f"Failed       : {failed}")
    print("=" * 70)


if __name__ == "__main__":
    main()