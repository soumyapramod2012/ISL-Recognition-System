"""
Indian Sign Language - 136-Class Uploaded Video Inference

Purpose
-------
Run the FINAL 136-class targeted ISL model on an uploaded video.

This is a separate inference utility.
It does NOT modify:
    - main
    - src/inference/realtime.py
    - the trained model
    - the dataset
    - the label mapping

Pipeline
--------
Video
  -> Legacy MediaPipe Holistic
  -> 258 features/frame
  -> continuous hand-detected segments
  -> 60-frame sliding windows
  -> HandLandmarkInterpolator(max_gap=5)
  -> LandmarkNormalizer
  -> SequenceGenerator
  -> FINAL 136-class targeted model
  -> confidence threshold
  -> TemporalStabilizer
  -> window-by-window results

Usage
-----
From the project root:

    python src/inference/video_136.py "path\\to\\video.mp4"

Optional:
    --step 3
        Number of frames between prediction windows.
        Default 3, matching the current realtime inference interval.

    --threshold 0.50
        Confidence threshold. Default 0.50.

    --display
        Play the video with the latest prediction overlaid.

    --save-video "outputs\\video_predictions\\result.mp4"
        Save an annotated copy of the video.

    --save-json "outputs\\video_predictions\\result.json"
        Save detailed predictions.

    --crop x1 y1 x2 y2
        Optional crop applied before MediaPipe processing.
        Useful when a TV/news video has a small interpreter window.

Example:
    python src/inference/video_136.py ^
        "D:\\Videos\\isl_news.mp4" ^
        --display ^
        --save-video "outputs\\video_predictions\\isl_result.mp4"

Notes
-----
The 60-frame requirement is preserved intentionally so that uploaded-video
inference remains compatible with the final trained model and the current
136-class realtime pipeline.

For accuracy evaluation, the source video should have known/annotated signs.
For an arbitrary news video, the output is a functional prediction stream,
not a ground-truth accuracy measurement.
"""

import argparse
import csv
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.temporal_stabilizer import TemporalStabilizer
from src.preprocessing.legacy_landmark_extractor import LegacyLandmarkExtractor
from src.training.hand_landmark_interpolator import HandLandmarkInterpolator
from src.training.landmark_normalizer import LandmarkNormalizer
from src.training.sequence_generator import SequenceGenerator


# ---------------------------------------------------------------------
# FINAL 136-CLASS CONFIGURATION
# ---------------------------------------------------------------------

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

# Match the current realtime inference cadence.
DEFAULT_STEP = 3
DEFAULT_THRESHOLD = 0.50

# Current realtime stabilizer settings.
HISTORY_SIZE = 5
INITIAL_MIN_VOTES = 2
TRANSITION_MIN_VOTES = 3


# ---------------------------------------------------------------------
# LABELS
# ---------------------------------------------------------------------

def load_class_names():
    if not LABEL_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Label mapping not found:\n{LABEL_MAPPING_PATH}"
        )

    with LABEL_MAPPING_PATH.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    if not isinstance(mapping, dict):
        raise ValueError("Expected the label mapping to be a JSON object.")

    # Support either:
    #   label -> index
    # or
    #   index -> label
    if all(str(key).isdigit() for key in mapping.keys()):
        index_to_label = {
            int(key): str(value)
            for key, value in mapping.items()
        }
    elif all(isinstance(value, int) for value in mapping.values()):
        index_to_label = {
            int(value): str(key)
            for key, value in mapping.items()
        }
    else:
        raise ValueError(
            "Unsupported label mapping format. "
            "Expected label->index or index->label."
        )

    class_names = [
        index_to_label[index]
        for index in sorted(index_to_label)
    ]

    if len(class_names) != 136:
        raise ValueError(
            f"Expected 136 classes, found {len(class_names)}."
        )

    return class_names


# ---------------------------------------------------------------------
# HAND DETECTION
# ---------------------------------------------------------------------

def hand_detected(landmarks):
    """
    The 258-feature representation is:

        pose        : 0:132
        left hand   : 132:195
        right hand  : 195:258
    """
    if landmarks.shape != (FEATURE_SIZE,):
        raise ValueError(
            f"Expected ({FEATURE_SIZE},), got {landmarks.shape}"
        )

    hands = landmarks[132:258]
    return not np.allclose(hands, 0.0)


# ---------------------------------------------------------------------
# VIDEO CROP
# ---------------------------------------------------------------------

def parse_crop(values):
    if values is None:
        return None

    if len(values) != 4:
        raise ValueError("Crop requires four integers: x1 y1 x2 y2")

    x1, y1, x2, y2 = values

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            "Invalid crop. Expected x2 > x1 and y2 > y1."
        )

    return x1, y1, x2, y2


def apply_crop(frame, crop):
    if crop is None:
        return frame

    x1, y1, x2, y2 = crop

    h, w = frame.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(1, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return frame

    return frame[y1:y2, x1:x2]


# ---------------------------------------------------------------------
# EXTRACTION
# ---------------------------------------------------------------------

def extract_video(video_path, crop=None):
    """
    Extract all 258-feature landmark vectors and hand-presence flags.

    The raw video is read once. A second pass is used later only when
    display/save-video is requested.
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    extractor = LegacyLandmarkExtractor()

    landmarks = []
    hand_flags = []

    frame_index = 0

    print()
    print("=" * 78)
    print("VIDEO PROCESSING")
    print("=" * 78)
    print(f"Video        : {video_path}")
    print(f"FPS          : {fps:.2f}")
    print(f"Total frames : {total_frames}")
    print()

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame_index += 1

            frame_for_mp = apply_crop(frame, crop)
            vector = extractor.extract(frame_for_mp)

            if vector.shape != (FEATURE_SIZE,):
                raise ValueError(
                    f"Frame {frame_index}: unexpected landmark shape "
                    f"{vector.shape}"
                )

            landmarks.append(vector)
            hand_flags.append(hand_detected(vector))

            if (
                frame_index % 100 == 0
                or frame_index == total_frames
            ):
                print(
                    f"\rFrames processed: "
                    f"{frame_index}/{total_frames}",
                    end="",
                    flush=True,
                )

    finally:
        cap.release()
        extractor.close()

    print()

    if not landmarks:
        raise RuntimeError("No video frames were extracted.")

    raw = np.asarray(landmarks, dtype=np.float32)
    hand_flags = np.asarray(hand_flags, dtype=bool)

    print(f"Raw landmark shape: {raw.shape}")
    print(
        f"Hand detected    : "
        f"{int(hand_flags.sum())}/{len(hand_flags)} frames"
    )

    return raw, hand_flags, fps, total_frames


# ---------------------------------------------------------------------
# CONTINUOUS HAND SEGMENTS
# ---------------------------------------------------------------------

def continuous_segments(hand_flags, minimum_length=SEQUENCE_LENGTH):
    """
    Find continuous runs where a hand is detected.

    This mirrors the realtime pipeline more closely than allowing a
    60-frame prediction window to cross a period where no hand exists.
    """
    segments = []

    start = None

    for index, present in enumerate(hand_flags):
        if present and start is None:
            start = index

        elif not present and start is not None:
            end = index - 1

            if end - start + 1 >= minimum_length:
                segments.append((start, end))

            start = None

    if start is not None:
        end = len(hand_flags) - 1

        if end - start + 1 >= minimum_length:
            segments.append((start, end))

    return segments


# ---------------------------------------------------------------------
# PREPROCESS ONE 60-FRAME WINDOW
# ---------------------------------------------------------------------

def preprocess_window(raw_window, interpolator, normalizer, generator):
    if raw_window.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
        raise ValueError(
            "Expected window shape "
            f"({SEQUENCE_LENGTH}, {FEATURE_SIZE}), "
            f"got {raw_window.shape}"
        )

    recovered = interpolator.interpolate(raw_window)
    normalized = normalizer.normalize(recovered)
    sequence = generator.generate(normalized)

    if sequence.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
        raise ValueError(
            "SequenceGenerator returned "
            f"{sequence.shape}; expected "
            f"({SEQUENCE_LENGTH}, {FEATURE_SIZE})"
        )

    return sequence.astype(np.float32)


# ---------------------------------------------------------------------
# TOP-5
# ---------------------------------------------------------------------

def top5_from_probabilities(probabilities, class_names):
    indices = np.argsort(probabilities)[::-1][:5]

    return [
        {
            "label": class_names[int(index)],
            "confidence": float(probabilities[int(index)]),
        }
        for index in indices
    ]


# ---------------------------------------------------------------------
# VIDEO PREDICTION
# ---------------------------------------------------------------------

def predict_video(
    raw,
    hand_flags,
    model,
    class_names,
    step=DEFAULT_STEP,
    threshold=DEFAULT_THRESHOLD,
):
    """
    Run sliding-window prediction and temporal stabilization.

    Returns a list of prediction records. Each record refers to the
    final frame of its 60-frame window.
    """
    if step < 1:
        raise ValueError("step must be >= 1")

    interpolator = HandLandmarkInterpolator(max_gap=5)
    normalizer = LandmarkNormalizer()
    generator = SequenceGenerator()

    stabilizer = TemporalStabilizer(
        history_size=HISTORY_SIZE,
        initial_min_votes=INITIAL_MIN_VOTES,
        transition_min_votes=TRANSITION_MIN_VOTES,
    )

    segments = continuous_segments(hand_flags)

    print()
    print("=" * 78)
    print("PREDICTION")
    print("=" * 78)
    print(f"Sequence length : {SEQUENCE_LENGTH}")
    print(f"Window step     : {step}")
    print(f"Threshold       : {threshold:.2f}")
    print(f"Hand segments   : {len(segments)}")

    if not segments:
        print()
        print(
            "No continuous hand-detected segment contains "
            f"{SEQUENCE_LENGTH} frames."
        )
        return [], segments

    records = []

    total_windows = 0

    for start, end in segments:
        length = end - start + 1

        starts = list(
            range(
                start,
                end - SEQUENCE_LENGTH + 2,
                step,
            )
        )

        final_start = end - SEQUENCE_LENGTH + 1

        if starts and starts[-1] != final_start:
            starts.append(final_start)

        total_windows += len(starts)

    print(f"Prediction windows: {total_windows}")
    print()

    processed_windows = 0

    for segment_number, (segment_start, segment_end) in enumerate(
        segments,
        start=1,
    ):
        # New hand segment = new temporal context.
        stabilizer = TemporalStabilizer(
            history_size=HISTORY_SIZE,
            initial_min_votes=INITIAL_MIN_VOTES,
            transition_min_votes=TRANSITION_MIN_VOTES,
        )

        starts = list(
            range(
                segment_start,
                segment_end - SEQUENCE_LENGTH + 2,
                step,
            )
        )

        final_start = segment_end - SEQUENCE_LENGTH + 1

        if starts and starts[-1] != final_start:
            starts.append(final_start)

        for window_start in starts:
            window_end = window_start + SEQUENCE_LENGTH - 1

            raw_window = raw[window_start:window_end + 1]

            sequence = preprocess_window(
                raw_window,
                interpolator,
                normalizer,
                generator,
            )

            start_time = time.perf_counter()

            probabilities = model.predict(
                np.expand_dims(sequence, axis=0),
                verbose=0,
            )[0]

            inference_ms = (
                time.perf_counter() - start_time
            ) * 1000.0

            top_entries = top5_from_probabilities(
                probabilities,
                class_names,
            )

            top_label = top_entries[0]["label"]
            top_confidence = top_entries[0]["confidence"]

            raw_label = (
                top_label
                if top_confidence >= threshold
                else "Uncertain"
            )

            stable_label, stable_confidence = stabilizer.update(
                raw_label,
                top_confidence,
                hand_detected=True,
            )

            record = {
                "segment": segment_number,
                "window_start": int(window_start),
                "window_end": int(window_end),
                "raw_label": raw_label,
                "raw_confidence": float(top_confidence),
                "stable_label": stable_label,
                "stable_confidence": float(stable_confidence),
                "inference_ms": float(inference_ms),
                "top5": top_entries,
            }

            records.append(record)

            processed_windows += 1

            print(
                f"\rWindow {processed_windows:4d}/"
                f"{total_windows:4d} | "
                f"Frame {window_end:6d} | "
                f"RAW: {raw_label:<25} "
                f"{top_confidence * 100:6.2f}% | "
                f"STABLE: {stable_label:<25}",
                end="",
                flush=True,
            )

    print()

    return records, segments


# ---------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------

def summarize(records):
    print()
    print("=" * 78)
    print("VIDEO PREDICTION SUMMARY")
    print("=" * 78)

    if not records:
        print("No prediction windows were produced.")
        return

    from collections import Counter

    raw_labels = [
        r["raw_label"]
        for r in records
        if r["raw_label"] != "Uncertain"
    ]

    stable_labels = [
        r["stable_label"]
        for r in records
        if r["stable_label"] != "Uncertain"
    ]

    print(f"Prediction windows : {len(records)}")
    print(f"RAW non-uncertain  : {len(raw_labels)}")
    print(f"STABLE non-uncertain: {len(stable_labels)}")

    if raw_labels:
        print()
        print("RAW TOP PREDICTIONS")
        print("-" * 78)

        counts = Counter(raw_labels)

        for label, count in counts.most_common(10):
            confidences = [
                r["raw_confidence"]
                for r in records
                if r["raw_label"] == label
            ]

            average = float(np.mean(confidences))

            print(
                f"{label:<32} "
                f"{count:4d} windows   "
                f"avg {average * 100:6.2f}%"
            )

    if stable_labels:
        print()
        print("STABLE PREDICTIONS")
        print("-" * 78)

        counts = Counter(stable_labels)

        for label, count in counts.most_common(10):
            confidences = [
                r["stable_confidence"]
                for r in records
                if r["stable_label"] == label
            ]

            average = float(np.mean(confidences))

            print(
                f"{label:<32} "
                f"{count:4d} windows   "
                f"avg {average * 100:6.2f}%"
            )

    inference_times = [
        r["inference_ms"]
        for r in records
    ]

    print()
    print(
        f"Average inference time: "
        f"{np.mean(inference_times):.2f} ms"
    )
    print(
        f"Median inference time : "
        f"{np.median(inference_times):.2f} ms"
    )


# ---------------------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------------------

def save_json(
    path,
    video_path,
    records,
    segments,
    fps,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "video": str(video_path),
        "fps": float(fps),
        "sequence_length": SEQUENCE_LENGTH,
        "window_step": None,
        "model": str(MODEL_PATH),
        "label_mapping": str(LABEL_MAPPING_PATH),
        "segments": [
            {
                "start_frame": int(start),
                "end_frame": int(end),
                "length": int(end - start + 1),
            }
            for start, end in segments
        ],
        "predictions": records,
    }

    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(f"\nJSON saved to:\n{path}")


def save_csv(path, records, fps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "segment",
        "window_start",
        "window_end",
        "time_seconds",
        "raw_label",
        "raw_confidence",
        "stable_label",
        "stable_confidence",
        "inference_ms",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for record in records:
            row = {
                key: record.get(key)
                for key in fields
            }

            row["time_seconds"] = (
                record["window_end"] / fps
                if fps > 0
                else None
            )

            writer.writerow(row)

    print(f"\nCSV saved to:\n{path}")


# ---------------------------------------------------------------------
# ANNOTATED VIDEO
# ---------------------------------------------------------------------

def annotate_video(
    video_path,
    output_path,
    records,
    crop=None,
    display=False,
):
    """
    Second pass through the video.

    The prediction associated with a frame is the latest prediction
    whose 60-frame window has ended at or before that frame.
    """
    output_path = (
        Path(output_path)
        if output_path
        else None
    )

    if output_path is not None:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not reopen video:\n{video_path}"
        )

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None

    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height),
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError(
                f"Could not create output video:\n{output_path}"
            )

    by_end = {
        int(record["window_end"]): record
        for record in records
    }

    latest_record = None
    frame_number = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame_number += 1

            if frame_number - 1 in by_end:
                latest_record = by_end[frame_number - 1]

            display_frame = frame.copy()

            if latest_record is None:
                label = "Waiting for 60 frames..."
                confidence = 0.0
            else:
                label = latest_record["stable_label"]
                confidence = latest_record["stable_confidence"]

            cv2.rectangle(
                display_frame,
                (10, 10),
                (min(width - 10, 760), 125),
                (20, 20, 20),
                -1,
            )

            cv2.putText(
                display_frame,
                "INDIAN SIGN LANGUAGE - 136 CLASSES",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                display_frame,
                f"STABLE: {label}",
                (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 0)
                if label != "Uncertain"
                else (0, 165, 255),
                2,
            )

            cv2.putText(
                display_frame,
                f"Confidence: {confidence:.2f}",
                (25, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
            )

            if crop is not None:
                x1, y1, x2, y2 = crop

                cv2.rectangle(
                    display_frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 0),
                    2,
                )

            if writer is not None:
                writer.write(display_frame)

            if display:
                cv2.imshow(
                    "ISL Video - 136 Classes",
                    display_frame,
                )

                key = cv2.waitKey(
                    max(1, int(1000 / fps))
                ) & 0xFF

                if key == ord("q"):
                    break

    finally:
        cap.release()

        if writer is not None:
            writer.release()

        if display:
            cv2.destroyAllWindows()

    if output_path is not None:
        print(
            f"\nAnnotated video saved to:\n"
            f"{output_path}"
        )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the final 136-class Indian Sign Language "
            "model on an uploaded video."
        )
    )

    parser.add_argument(
        "video",
        help="Path to the input video.",
    )

    parser.add_argument(
        "--step",
        type=int,
        default=DEFAULT_STEP,
        help=(
            "Frame step between prediction windows. "
            f"Default: {DEFAULT_STEP}"
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "Confidence threshold. "
            f"Default: {DEFAULT_THRESHOLD}"
        ),
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the annotated video during processing.",
    )

    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Save an annotated MP4.",
    )

    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Save detailed JSON predictions.",
    )

    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="Save a CSV prediction table.",
    )

    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
        help=(
            "Optional interpreter crop: "
            "X1 Y1 X2 Y2"
        ),
    )

    args = parser.parse_args()

    video_path = Path(args.video)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found:\n{video_path}"
        )

    if args.step < 1:
        raise ValueError("--step must be >= 1")

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(
            "--threshold must be between 0.0 and 1.0"
        )

    crop = parse_crop(args.crop)

    print("=" * 78)
    print("INDIAN SIGN LANGUAGE - 136 CLASS VIDEO INFERENCE")
    print("=" * 78)
    print(f"Model         : {MODEL_PATH}")
    print(f"Label mapping : {LABEL_MAPPING_PATH}")
    print(f"Sequence      : {SEQUENCE_LENGTH} frames")
    print(f"Features      : {FEATURE_SIZE}")
    print(f"Window step   : {args.step}")
    print(f"Threshold     : {args.threshold:.2f}")

    if crop is not None:
        print(f"Crop          : {crop}")

    print()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final 136-class model not found:\n{MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    class_names = load_class_names()

    if model.input_shape[-2:] != (
        SEQUENCE_LENGTH,
        FEATURE_SIZE,
    ):
        raise RuntimeError(
            f"Model input is {model.input_shape}; "
            f"expected (*, {SEQUENCE_LENGTH}, {FEATURE_SIZE})."
        )

    if model.output_shape[-1] != 136:
        raise RuntimeError(
            f"Model output is {model.output_shape}; "
            "expected 136 classes."
        )

    print("Model compatibility: PASS")
    print(f"Classes loaded   : {len(class_names)}")

    raw, hand_flags, fps, total_frames = extract_video(
        video_path,
        crop=crop,
    )

    records, segments = predict_video(
        raw,
        hand_flags,
        model,
        class_names,
        step=args.step,
        threshold=args.threshold,
    )

    summarize(records)

    if args.save_json:
        save_json(
            args.save_json,
            video_path,
            records,
            segments,
            fps,
        )

    if args.save_csv:
        save_csv(
            args.save_csv,
            records,
            fps,
        )

    if args.display or args.save_video:
        annotate_video(
            video_path,
            args.save_video,
            records,
            crop=crop,
            display=args.display,
        )

    print()
    print("=" * 78)
    print("VIDEO INFERENCE COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
