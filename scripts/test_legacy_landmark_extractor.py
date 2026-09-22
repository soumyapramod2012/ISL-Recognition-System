import argparse
import cv2
import sys
import numpy as np
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.legacy_landmark_extractor import LegacyLandmarkExtractor


DEFAULT_VIDEOS = [
    PROJECT_ROOT / "dataset/raw/INCLUDE/Animals/4. Bird/MVI_2987.MOV",
    PROJECT_ROOT / "dataset/raw/INCLUDE/Animals/7. Horse/MVI_2996.MOV",
]


def test_video(video_path, extractor):
    print("\n" + "=" * 70)
    print(f"Testing: {video_path}")
    print("=" * 70)

    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return False

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return False

    total_frames = 0
    left_frames = 0
    right_frames = 0
    both_frames = 0
    invalid_shape = 0
    nan_frames = 0
    inf_frames = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            total_frames += 1

            landmarks = extractor.extract(frame)

            # Check feature shape
            if landmarks.shape != (258,):
                invalid_shape += 1
                print(
                    f"ERROR: Frame {total_frames}: "
                    f"expected (258,), got {landmarks.shape}"
                )
                continue

            # Check NaN / Inf
            if np.isnan(landmarks).any():
                nan_frames += 1

            if np.isinf(landmarks).any():
                inf_frames += 1

            # Feature layout:
            # Pose       = 0:132
            # Left hand  = 132:195
            # Right hand = 195:258
            left_hand = landmarks[132:195]
            right_hand = landmarks[195:258]

            left_present = not np.allclose(left_hand, 0.0)
            right_present = not np.allclose(right_hand, 0.0)

            if left_present:
                left_frames += 1

            if right_present:
                right_frames += 1

            if left_present and right_present:
                both_frames += 1

    finally:
        cap.release()

    print(f"Total frames       : {total_frames}")
    print(f"Feature dimension  : 258")

    if total_frames > 0:
        print(
            f"Left hand detected : {left_frames}/{total_frames} "
            f"({left_frames / total_frames * 100:.2f}%)"
        )
        print(
            f"Right hand detected: {right_frames}/{total_frames} "
            f"({right_frames / total_frames * 100:.2f}%)"
        )
        print(
            f"Both hands detected: {both_frames}/{total_frames} "
            f"({both_frames / total_frames * 100:.2f}%)"
        )

    print(f"Invalid shape      : {invalid_shape}")
    print(f"NaN frames         : {nan_frames}")
    print(f"Inf frames         : {inf_frames}")

    passed = (
        total_frames > 0
        and invalid_shape == 0
        and nan_frames == 0
        and inf_frames == 0
    )

    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")

    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Regression test for the legacy 258-feature landmark extractor."
    )

    parser.add_argument(
        "videos",
        nargs="*",
        type=Path,
        help="Video files to test. Defaults to Bird and Horse samples.",
    )

    args = parser.parse_args()

    videos = args.videos if args.videos else DEFAULT_VIDEOS

    extractor = LegacyLandmarkExtractor()

    try:
        results = [
            test_video(video_path, extractor)
            for video_path in videos
        ]
    finally:
        extractor.close()

    print("\n" + "=" * 70)
    print(f"Passed: {sum(results)}/{len(results)}")
    print("=" * 70)

    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())