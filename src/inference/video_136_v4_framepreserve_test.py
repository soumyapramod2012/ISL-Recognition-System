"""
136-class V4 diagnostic: preserve ORIGINAL video-frame positions.

Purpose:
    Test whether deleting no-hand frames is damaging temporal information.

Pipeline:
    video frames
        -> extract 258 landmarks for EVERY frame
        -> preserve original frame positions
        -> HandLandmarkInterpolator(max_gap=5)
        -> LandmarkNormalizer
        -> SequenceGenerator
        -> frozen 136-class targeted model

IMPORTANT:
    - Does NOT modify realtime.py.
    - Does NOT modify the trained model.
    - This is a diagnostic for isolated dataset clips.
    - For clips < 60 frames, SequenceGenerator zero-pads.
    - For clips >= 60 frames, SequenceGenerator resamples the full
      original-position sequence to 60 frames.

Run from project root:
    python src\inference\video_136_v4_framepreserve_test.py "path\to\video.MOV" --threshold 0.50 --max-gap 5 --save-json "outputs\video_predictions\Laptop_MVI_4974_v4.json"
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.legacy_landmark_extractor import LegacyLandmarkExtractor
from src.training.hand_landmark_interpolator import HandLandmarkInterpolator
from src.training.landmark_normalizer import LandmarkNormalizer
from src.training.sequence_generator import SequenceGenerator


MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "isl_lstm_generalized_filtered_136_targeted.keras"
)

MAPPING_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "generalized_filtered_136_label_mapping.json"
)

SEQUENCE_LENGTH = 60
FEATURES = 258


def load_mapping():
    raw = json.loads(MAPPING_FILE.read_text(encoding="utf-8"))

    if all(str(k).isdigit() for k in raw.keys()):
        return {int(k): v for k, v in raw.items()}

    return {int(v): k for k, v in raw.items()}


def hand_detected(landmarks):
    """Return True when either hand contains non-zero landmarks."""
    return not np.allclose(landmarks[132:258], 0.0)


def extract_video(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    extractor = LegacyLandmarkExtractor()

    frames = []
    hand_flags = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            landmarks = extractor.extract(frame)

            if landmarks.shape != (FEATURES,):
                raise ValueError(
                    f"Unexpected landmark shape: {landmarks.shape}; "
                    f"expected ({FEATURES},)"
                )

            frames.append(landmarks)
            hand_flags.append(hand_detected(landmarks))

    finally:
        cap.release()
        extractor.close()

    if not frames:
        raise RuntimeError("No frames were extracted.")

    return (
        np.asarray(frames, dtype=np.float32),
        np.asarray(hand_flags, dtype=bool),
        fps,
    )


def find_hand_gaps(hand_flags):
    """Return consecutive no-hand gaps as (start, end, length), 1-based."""
    gaps = []
    start = None

    for i, has_hand in enumerate(hand_flags):
        if not has_hand and start is None:
            start = i

        if has_hand and start is not None:
            gaps.append((start + 1, i, i - start))
            start = None

    if start is not None:
        gaps.append((start + 1, len(hand_flags), len(hand_flags) - start))

    return gaps


def make_sequence(raw_landmarks, max_gap):
    interpolator = HandLandmarkInterpolator(max_gap=max_gap)
    normalizer = LandmarkNormalizer()
    generator = SequenceGenerator()

    # CRITICAL:
    # Interpolate while retaining ALL original video-frame positions.
    interpolated = interpolator.interpolate(raw_landmarks)

    normalized = normalizer.normalize(interpolated)

    sequence = generator.generate(normalized)

    return np.asarray(sequence, dtype=np.float32)


def predict(model, sequence, mapping):
    start = time.perf_counter()

    probabilities = model.predict(
        np.expand_dims(sequence, axis=0),
        verbose=0,
    )[0]

    inference_ms = (time.perf_counter() - start) * 1000.0

    indices = np.argsort(probabilities)[::-1][:5]

    top5 = [
        {
            "label": mapping[int(i)],
            "confidence": float(probabilities[i]),
        }
        for i in indices
    ]

    best_index = int(indices[0])
    best_label = mapping[best_index]
    best_confidence = float(probabilities[best_index])

    return best_label, best_confidence, top5, inference_ms


def main():
    parser = argparse.ArgumentParser(
        description="136-class V4 diagnostic preserving original frame positions."
    )

    parser.add_argument("video", help="Path to input video")

    parser.add_argument(
        "--max-gap",
        type=int,
        default=5,
        help="Maximum missing-hand gap to interpolate (default: 5)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Confidence threshold (default: 0.50)",
    )

    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional JSON output path",
    )

    args = parser.parse_args()

    video_path = Path(args.video)

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    if args.max_gap < 0:
        raise ValueError("--max-gap must be >= 0")

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    print("=" * 78)
    print("136-CLASS V4 FRAME-PRESERVING DATASET VIDEO TEST")
    print("=" * 78)
    print(f"Video       : {video_path}")
    print(f"Model       : {MODEL_PATH.name}")
    print(f"Max gap     : {args.max_gap} frames")
    print(f"Threshold   : {args.threshold:.2f}")
    print("Frame mode  : PRESERVE ALL ORIGINAL VIDEO FRAME POSITIONS")
    print()

    mapping = load_mapping()

    print("Loading targeted 136-class model...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    if model.output_shape[-1] != len(mapping):
        raise RuntimeError(
            f"Model outputs {model.output_shape[-1]} classes, "
            f"but mapping contains {len(mapping)} labels."
        )

    print("Extracting landmarks from EVERY video frame...")
    raw, hand_flags, fps = extract_video(video_path)

    total_frames = len(raw)
    hand_count = int(hand_flags.sum())
    no_hand_count = total_frames - hand_count
    gaps = find_hand_gaps(hand_flags)

    print()
    print("-" * 78)
    print("FRAME EXTRACTION")
    print("-" * 78)
    print(f"FPS                 : {fps:.2f}")
    print(f"Total frames        : {total_frames}")
    print(f"Hand detected       : {hand_count}/{total_frames}")
    print(f"No-hand frames      : {no_hand_count}")
    print(f"Hand ratio          : {hand_count / total_frames:.2%}")

    print()
    print("No-hand gaps:")
    if not gaps:
        print("  None")
    else:
        for start, end, length in gaps:
            status = "INTERPOLATE" if length <= args.max_gap else "KEEP GAP"
            print(
                f"  frames {start}-{end} ({length} frames) -> {status}"
            )

    print()
    print("-" * 78)
    print("SEQUENCE GENERATION")
    print("-" * 78)
    print("IMPORTANT: no-hand frames were NOT deleted.")
    print(f"Source sequence length : {total_frames}")

    if total_frames < SEQUENCE_LENGTH:
        sequence_mode = "zero_pad"
    else:
        sequence_mode = "resample"

    print(f"Sequence mode         : {sequence_mode}")
    print(f"Model sequence length : {SEQUENCE_LENGTH}")

    sequence = make_sequence(raw, args.max_gap)

    print(f"Generated shape       : {sequence.shape}")

    print()
    print("-" * 78)
    print("MODEL PREDICTION")
    print("-" * 78)

    label, confidence, top5, inference_ms = predict(
        model,
        sequence,
        mapping,
    )

    if confidence >= args.threshold:
        final_label = label
        status = "CONFIDENT"
    else:
        final_label = "Uncertain"
        status = "UNCERTAIN"

    print(f"FINAL                 : {final_label} {confidence:.2%}")
    print(f"STATUS                : {status}")
    print("TOP 5:")

    for item in top5:
        print(
            f"  {item['label']:<30} "
            f"{item['confidence']:.2%}"
        )

    print(f"Inference             : {inference_ms:.1f} ms")

    result = {
        "video": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "hand_detected_frames": hand_count,
        "no_hand_frames": no_hand_count,
        "hand_ratio": float(hand_count / total_frames),
        "max_gap": args.max_gap,
        "sequence_length": SEQUENCE_LENGTH,
        "source_length": total_frames,
        "sequence_mode": sequence_mode,
        "frame_positions_preserved": True,
        "no_hand_gaps": [
            {
                "start_frame": start,
                "end_frame": end,
                "length": length,
                "interpolated": length <= args.max_gap,
            }
            for start, end, length in gaps
        ],
        "final_label": final_label,
        "raw_best_label": label,
        "confidence": confidence,
        "status": status,
        "top5": top5,
        "inference_ms": inference_ms,
    }

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        print()
        print(f"JSON saved            : {output_path}")

    print()
    print("=" * 78)
    print("TEST COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
