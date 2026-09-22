import cv2
import numpy as np
from pathlib import Path

from src.preprocessing.legacy_landmark_extractor import LegacyLandmarkExtractor


INPUT_ROOT = Path("dataset/raw/INCLUDE")
OUTPUT_ROOT = Path("dataset/processed/include_landmarks_legacy")


VIDEO_EXTENSIONS = {".MOV", ".MP4", ".mov", ".mp4"}


def get_videos():
    """Find all INCLUDE videos recursively."""
    return sorted(
        p for p in INPUT_ROOT.rglob("*")
        if p.is_file() and p.suffix in VIDEO_EXTENSIONS
    )


def main():

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    videos = get_videos()

    print("=" * 70)
    print("INCLUDE LEGACY LANDMARK GENERATION")
    print("=" * 70)
    print(f"Videos found : {len(videos)}")
    print(f"Output       : {OUTPUT_ROOT}")
    print()

    extractor = LegacyLandmarkExtractor()

    processed = 0
    skipped = 0
    failed = 0

    try:

        for index, video_path in enumerate(videos, start=1):

            # Immediate parent folder is the class label
            label = video_path.parent.name

            output_dir = OUTPUT_ROOT / label
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / f"{video_path.stem}.npy"

            if output_file.exists():
                print(
                    f"[{index}/{len(videos)}] "
                    f"SKIP - {label} - {video_path.name}"
                )
                skipped += 1
                continue

            print(
                f"[{index}/{len(videos)}] "
                f"{label} - {video_path.name}"
            )

            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                print("    ERROR: Could not open video")
                failed += 1
                continue

            frames = []

            try:
                while True:

                    success, frame = cap.read()

                    if not success:
                        break

                    vector = extractor.extract(frame)
                    frames.append(vector)

            finally:
                cap.release()

            if not frames:
                print("    ERROR: No frames extracted")
                failed += 1
                continue

            frames = np.asarray(frames, dtype=np.float32)

            # Validate shape
            if frames.ndim != 2:
                print(
                    f"    ERROR: Invalid shape {frames.shape}"
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

            # Validate numerical values
            if np.isnan(frames).any():
                print("    ERROR: NaN values detected")
                failed += 1
                continue

            if np.isinf(frames).any():
                print("    ERROR: Infinite values detected")
                failed += 1
                continue

            np.save(output_file, frames)

            print(
                f"    Saved: {output_file} "
                f"({frames.shape[0]} frames)"
            )

            processed += 1

    finally:
        extractor.close()

    print()
    print("=" * 70)
    print("INCLUDE LEGACY LANDMARK GENERATION COMPLETE")
    print("=" * 70)
    print(f"Videos found : {len(videos)}")
    print(f"Processed    : {processed}")
    print(f"Skipped      : {skipped}")
    print(f"Failed       : {failed}")
    print("=" * 70)


if __name__ == "__main__":
    main()