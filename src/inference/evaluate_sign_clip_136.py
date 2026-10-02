"""
Indian Sign Language - 136-Class Individual Sign Evaluator

Purpose
-------
Evaluate ONE isolated sign clip with the FINAL frozen 136-class model.

This file is intentionally separate from realtime.py and video_136.py.
It does NOT modify the model, dataset, main branch, or realtime inference.

Pipeline
--------
Video clip
  -> optional time range
  -> Legacy MediaPipe Holistic
  -> 258 features/frame
  -> 60-frame sliding windows
  -> HandLandmarkInterpolator(max_gap=5)
  -> LandmarkNormalizer
  -> SequenceGenerator
  -> FINAL 136-class targeted model
  -> RAW Top-1 / Top-5
  -> mean-probability aggregate
  -> majority Top-1

Important
---------
TemporalStabilizer is NOT used here. This is deliberate: for an isolated
sign, we want to measure what the model itself predicts in each 60-frame
window without the realtime display logic carrying an earlier prediction
into the next sign.

Usage from the project root
---------------------------

python src\\inference\\evaluate_sign_clip_136.py ^
  "outputs\\external_tests\\isl_family_members.mp4" ^
  --start 25 --end 30 ^
  --name "Son"

The --start and --end values are seconds in the source video.

Optional:
  --step 3
  --threshold 0.50
  --save-json "outputs\\external_tests\\son_136_eval.json"
  --save-csv  "outputs\\external_tests\\son_136_eval.csv"

The script reports:
  - number of frames
  - hand-detected frames
  - number of valid 60-frame windows
  - per-window Top-5
  - mean probability across all valid windows
  - majority Top-1
  - mean Top-1 confidence
  - accepted/rejected using the selected confidence threshold

This is a real-video evaluation only when the selected clip is known to
contain the named sign throughout the evaluated interval.
"""

import argparse
import csv
import json
import sys
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

LABEL_MAPPING_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "generalized_filtered_136_label_mapping.json"
)

SEQUENCE_LENGTH = 60
FEATURE_SIZE = 258
DEFAULT_STEP = 3
DEFAULT_THRESHOLD = 0.50


def load_class_names():
    with LABEL_MAPPING_PATH.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    # The project mapping is normally index -> label, but handle either
    # JSON orientation defensively.
    if all(str(k).isdigit() for k in mapping.keys()):
        return [mapping[str(i)] for i in range(len(mapping))]

    if all(isinstance(v, (int, float)) for v in mapping.values()):
        ordered = sorted(mapping.items(), key=lambda item: int(item[1]))
        return [item[0] for item in ordered]

    raise ValueError("Unsupported label mapping format.")


def hand_detected(vector):
    # Hand blocks are:
    # pose 0:132, left hand 132:195, right hand 195:258.
    left = np.any(np.abs(vector[132:195]) > 1e-8)
    right = np.any(np.abs(vector[195:258]) > 1e-8)
    return bool(left or right)


def preprocess_window(raw_window, interpolator, normalizer, generator):
    recovered = interpolator.interpolate(raw_window)
    normalized = normalizer.normalize(recovered)
    sequence = generator.generate(normalized)

    if sequence.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
        raise ValueError(
            f"Unexpected processed sequence shape: {sequence.shape}"
        )

    return sequence.astype(np.float32)


def top5(probabilities, class_names):
    indices = np.argsort(probabilities)[::-1][:5]
    return [
        {
            "label": class_names[int(i)],
            "confidence": float(probabilities[int(i)]),
        }
        for i in indices
    ]


def parse_time_range(cap, start_sec, end_sec):
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        raise RuntimeError("Could not determine video FPS.")

    duration = total / fps

    start_sec = 0.0 if start_sec is None else float(start_sec)
    end_sec = duration if end_sec is None else float(end_sec)

    if start_sec < 0:
        raise ValueError("--start must be >= 0.")
    if end_sec <= start_sec:
        raise ValueError("--end must be greater than --start.")
    if start_sec >= duration:
        raise ValueError(
            f"--start {start_sec:.2f}s is beyond video duration "
            f"{duration:.2f}s."
        )

    end_sec = min(end_sec, duration)

    start_frame = int(round(start_sec * fps))
    end_frame = int(round(end_sec * fps)) - 1

    start_frame = max(0, start_frame)
    end_frame = min(total - 1, end_frame)

    return fps, total, duration, start_frame, end_frame


def extract_clip(video_path, start_sec, end_sec):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video:\n{video_path}")

    fps, total, duration, start_frame, end_frame = parse_time_range(
        cap, start_sec, end_sec
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    extractor = LegacyLandmarkExtractor()

    vectors = []
    hand_flags = []
    frame_numbers = []

    try:
        current = start_frame

        while current <= end_frame:
            ok, frame = cap.read()
            if not ok:
                break

            vector = extractor.extract(frame)

            if vector.shape != (FEATURE_SIZE,):
                raise ValueError(
                    f"Frame {current + 1}: expected {(FEATURE_SIZE,)}, "
                    f"got {vector.shape}"
                )

            vectors.append(vector)
            hand_flags.append(hand_detected(vector))
            frame_numbers.append(current)

            current += 1
    finally:
        extractor.close()
        cap.release()

    if not vectors:
        raise RuntimeError("No frames were extracted from the selected range.")

    raw = np.asarray(vectors, dtype=np.float32)
    flags = np.asarray(hand_flags, dtype=bool)

    return {
        "fps": fps,
        "total_frames": total,
        "duration": duration,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "raw": raw,
        "hand_flags": flags,
        "frame_numbers": frame_numbers,
    }


def evaluate(raw, hand_flags, model, class_names, step):
    if len(raw) < SEQUENCE_LENGTH:
        raise RuntimeError(
            f"Only {len(raw)} frames available; "
            f"{SEQUENCE_LENGTH} are required for one prediction."
        )

    interpolator = HandLandmarkInterpolator(max_gap=5)
    normalizer = LandmarkNormalizer()
    generator = SequenceGenerator()

    probabilities = []
    records = []

    # Only use windows where a hand is present in the majority of frames.
    # This prevents a transition/blank interval from dominating the result.
    for start in range(0, len(raw) - SEQUENCE_LENGTH + 1, step):
        end = start + SEQUENCE_LENGTH
        window_flags = hand_flags[start:end]

        if float(window_flags.mean()) < 0.50:
            continue

        sequence = preprocess_window(
            raw[start:end],
            interpolator,
            normalizer,
            generator,
        )

        batch = np.expand_dims(sequence, axis=0)
        probs = model.predict(batch, verbose=0)[0]
        probs = np.asarray(probs, dtype=np.float64)

        probabilities.append(probs)

        best = int(np.argmax(probs))
        records.append(
            {
                "window_start_frame": int(start),
                "window_end_frame": int(end - 1),
                "top1_label": class_names[best],
                "top1_confidence": float(probs[best]),
                "top5": top5(probs, class_names),
                "hand_fraction": float(window_flags.mean()),
            }
        )

    if not probabilities:
        raise RuntimeError(
            "No valid 60-frame windows remained. "
            "Try a longer clip or a better crop/range."
        )

    matrix = np.vstack(probabilities)
    mean_probs = matrix.mean(axis=0)

    mean_best = int(np.argmax(mean_probs))
    mean_top5 = top5(mean_probs, class_names)

    top1_indices = [
        int(np.argmax(row))
        for row in matrix
    ]

    counts = np.bincount(
        top1_indices,
        minlength=len(class_names),
    )
    majority_index = int(np.argmax(counts))

    return {
        "records": records,
        "mean_top5": mean_top5,
        "mean_top1_label": class_names[mean_best],
        "mean_top1_confidence": float(mean_probs[mean_best]),
        "majority_top1_label": class_names[majority_index],
        "majority_top1_windows": int(counts[majority_index]),
        "total_valid_windows": len(records),
        "mean_probabilities": mean_probs,
    }


def save_json(path, result, metadata):
    output = {
        "metadata": metadata,
        "summary": {
            "mean_top1_label": result["mean_top1_label"],
            "mean_top1_confidence": result["mean_top1_confidence"],
            "majority_top1_label": result["majority_top1_label"],
            "majority_top1_windows": result["majority_top1_windows"],
            "total_valid_windows": result["total_valid_windows"],
            "mean_top5": result["mean_top5"],
        },
        "windows": result["records"],
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "window_start_frame",
                "window_end_frame",
                "top1_label",
                "top1_confidence",
                "hand_fraction",
                "top2_label",
                "top2_confidence",
                "top3_label",
                "top3_confidence",
                "top4_label",
                "top4_confidence",
                "top5_label",
                "top5_confidence",
            ]
        )

        for record in records:
            row = [
                record["window_start_frame"],
                record["window_end_frame"],
                record["top1_label"],
                f'{record["top1_confidence"]:.6f}',
                f'{record["hand_fraction"]:.6f}',
            ]

            for item in record["top5"]:
                row.extend(
                    [
                        item["label"],
                        f'{item["confidence"]:.6f}',
                    ]
                )

            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one isolated ISL sign with the final 136-class model."
    )

    parser.add_argument("video", help="Path to the MP4/video file.")
    parser.add_argument("--start", type=float, default=None,
                        help="Start time in seconds.")
    parser.add_argument("--end", type=float, default=None,
                        help="End time in seconds.")
    parser.add_argument("--name", default="Unknown",
                        help="Expected sign name, e.g. Son.")
    parser.add_argument("--step", type=int, default=DEFAULT_STEP,
                        help=f"Prediction window step. Default: {DEFAULT_STEP}")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Confidence threshold. Default: {DEFAULT_THRESHOLD}")
    parser.add_argument("--save-json", default=None,
                        help="Optional JSON output path.")
    parser.add_argument("--save-csv", default=None,
                        help="Optional CSV output path.")

    args = parser.parse_args()

    if args.step < 1:
        raise ValueError("--step must be >= 1.")
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1.")

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found:\n{video_path}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")
    if not LABEL_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Label mapping not found:\n{LABEL_MAPPING_PATH}"
        )

    class_names = load_class_names()
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    if model.input_shape[-2:] != (SEQUENCE_LENGTH, FEATURE_SIZE):
        raise RuntimeError(
            f"Model input is {model.input_shape}; "
            f"expected (*, {SEQUENCE_LENGTH}, {FEATURE_SIZE})."
        )

    if model.output_shape[-1] != 136:
        raise RuntimeError(
            f"Model output is {model.output_shape}; expected 136."
        )

    print("=" * 78)
    print("INDIAN SIGN LANGUAGE - 136 CLASS INDIVIDUAL SIGN EVALUATION")
    print("=" * 78)
    print(f"Model      : {MODEL_PATH}")
    print(f"Expected   : {args.name}")
    print(f"Video      : {video_path}")
    print(f"Time range : {args.start} -> {args.end} seconds")
    print(f"Step       : {args.step}")
    print(f"Threshold  : {args.threshold:.2f}")
    print()

    data = extract_clip(video_path, args.start, args.end)

    raw = data["raw"]
    flags = data["hand_flags"]

    print(f"Video FPS              : {data['fps']:.2f}")
    print(f"Selected frames        : {len(raw)}")
    print(f"Hand-detected frames   : {int(flags.sum())}/{len(flags)}")
    print(f"Hand-detection ratio   : {flags.mean() * 100:.1f}%")
    print()

    result = evaluate(
        raw,
        flags,
        model,
        class_names,
        args.step,
    )

    print("-" * 78)
    print("PER-WINDOW PREDICTIONS")
    print("-" * 78)

    for i, record in enumerate(result["records"], 1):
        print(
            f"{i:03d} | "
            f"frames {record['window_start_frame']:4d}-"
            f"{record['window_end_frame']:4d} | "
            f"{record['top1_label']:<25} "
            f"{record['top1_confidence'] * 100:6.2f}%"
        )

    print()
    print("-" * 78)
    print("AGGREGATED RESULT")
    print("-" * 78)

    print(
        f"Mean-probability Top-1 : "
        f"{result['mean_top1_label']} "
        f"({result['mean_top1_confidence'] * 100:.2f}%)"
    )
    print(
        f"Majority Top-1         : "
        f"{result['majority_top1_label']} "
        f"({result['majority_top1_windows']}/"
        f"{result['total_valid_windows']} windows)"
    )

    print()
    print("Mean-probability Top-5:")
    for rank, item in enumerate(result["mean_top5"], 1):
        print(
            f"  {rank}. {item['label']:<25} "
            f"{item['confidence'] * 100:6.2f}%"
        )

    accepted = (
        result["mean_top1_label"] == args.name
        and result["mean_top1_confidence"] >= args.threshold
    )

    print()
    print(
        "EVALUATION STATUS : "
        + ("PASS" if accepted else "CHECK")
    )
    print(
        f"Reason: expected='{args.name}', "
        f"predicted='{result['mean_top1_label']}', "
        f"confidence={result['mean_top1_confidence'] * 100:.2f}%"
    )

    metadata = {
        "expected_sign": args.name,
        "video": str(video_path),
        "start_seconds": args.start,
        "end_seconds": args.end,
        "step": args.step,
        "threshold": args.threshold,
        "model": str(MODEL_PATH),
        "label_mapping": str(LABEL_MAPPING_PATH),
        "sequence_length": SEQUENCE_LENGTH,
        "feature_size": FEATURE_SIZE,
        "fps": data["fps"],
        "selected_frames": len(raw),
        "hand_detected_frames": int(flags.sum()),
        "hand_detection_ratio": float(flags.mean()),
        "evaluation_status": "PASS" if accepted else "CHECK",
    }

    if args.save_json:
        save_json(args.save_json, result, metadata)
        print(f"\nJSON saved: {args.save_json}")

    if args.save_csv:
        save_csv(args.save_csv, result["records"])
        print(f"CSV saved : {args.save_csv}")


if __name__ == "__main__":
    main()
