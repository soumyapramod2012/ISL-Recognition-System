"""
Indian Sign Language - 136-Class Gap-Tolerant Diagnostic

Purpose:
    Diagnostic only. Tests whether short hand-detection gaps are causing
    the current uploaded-video pipeline to reject otherwise valid signs.

Pipeline:
    Video -> Legacy MediaPipe Holistic -> 258 features/frame
          -> allow short hand gaps (default 5 frames)
          -> interpolate missing hand landmarks
          -> normalize
          -> SequenceGenerator (variable length -> exactly 60 frames)
          -> frozen 136-class targeted model

Does NOT modify realtime.py, the trained model, or the dataset.
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

from src.inference.temporal_stabilizer import TemporalStabilizer
from src.preprocessing.legacy_landmark_extractor import LegacyLandmarkExtractor
from src.training.hand_landmark_interpolator import HandLandmarkInterpolator
from src.training.landmark_normalizer import LandmarkNormalizer
from src.training.sequence_generator import SequenceGenerator

MODEL_PATH = PROJECT_ROOT / "saved_models" / "isl_lstm_generalized_filtered_136_targeted.keras"
LABEL_MAPPING_PATH = PROJECT_ROOT / "outputs" / "generalized_filtered_136_label_mapping.json"
FEATURE_SIZE = 258
SEQUENCE_LENGTH = 60
DEFAULT_THRESHOLD = 0.50
DEFAULT_MAX_GAP = 5


def load_class_names():
    with LABEL_MAPPING_PATH.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    if all(str(k).isdigit() for k in mapping):
        idx_to_label = {int(k): str(v) for k, v in mapping.items()}
    else:
        idx_to_label = {int(v): str(k) for k, v in mapping.items()}
    return [idx_to_label[i] for i in sorted(idx_to_label)]


def hand_detected(v):
    return not np.allclose(v[132:258], 0.0)


def tolerant_segments(flags, max_gap=5, min_length=1):
    """Group hand-present runs separated by <= max_gap absent frames."""
    segments = []
    start = None
    last_hand = None
    gap = 0

    for i, present in enumerate(flags):
        if present:
            if start is None:
                start = i
            last_hand = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                end = last_hand
                if end - start + 1 >= min_length:
                    segments.append((start, end))
                start = None
                last_hand = None
                gap = 0

    if start is not None and last_hand is not None:
        if last_hand - start + 1 >= min_length:
            segments.append((start, last_hand))

    return segments


def extract(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extractor = LegacyLandmarkExtractor()
    raw = []
    flags = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            v = extractor.extract(frame)
            if v.shape != (FEATURE_SIZE,):
                raise ValueError(f"Unexpected feature shape: {v.shape}")
            raw.append(v)
            flags.append(hand_detected(v))
    finally:
        cap.release()
        extractor.close()

    return np.asarray(raw, dtype=np.float32), np.asarray(flags, dtype=bool), fps, total


def top5(probs, names):
    idx = np.argsort(probs)[::-1][:5]
    return [(names[int(i)], float(probs[int(i)])) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--max-gap", type=int, default=DEFAULT_MAX_GAP)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--save-json", default=None)
    args = ap.parse_args()

    video = Path(args.video)
    names = load_class_names()
    model = tf.keras.models.load_model(MODEL_PATH)

    print("=" * 78)
    print("136-CLASS GAP-TOLERANT DIAGNOSTIC")
    print("=" * 78)
    print(f"Video       : {video}")
    print(f"Model       : {MODEL_PATH.name}")
    print(f"Max hand gap: {args.max_gap} frames")
    print(f"Threshold   : {args.threshold:.2f}")
    print()

    raw, flags, fps, total = extract(video)
    print(f"FPS             : {fps:.2f}")
    print(f"Total frames    : {total}")
    print(f"Hand detected   : {int(flags.sum())}/{len(flags)}")
    print()

    segments = tolerant_segments(flags, args.max_gap, min_length=SEQUENCE_LENGTH)
    print(f"Tolerant segments >= 60 frames: {len(segments)}")
    for n, (s, e) in enumerate(segments, 1):
        print(f"  Segment {n}: frames {s+1}-{e+1} ({e-s+1} frames)")

    if not segments:
        print("\nNo qualifying tolerant segment. Try --max-gap 8 or 10 for diagnosis.")
        return

    interpolator = HandLandmarkInterpolator(max_gap=args.max_gap)
    normalizer = LandmarkNormalizer()
    generator = SequenceGenerator()
    stabilizer = TemporalStabilizer(history_size=5, initial_min_votes=2, transition_min_votes=3)

    results = []
    print("\nPREDICTIONS")
    print("-" * 78)

    for n, (s, e) in enumerate(segments, 1):
        segment_raw = raw[s:e+1]
        recovered = interpolator.interpolate(segment_raw)
        normalized = normalizer.normalize(recovered)
        sequence = generator.generate(normalized).astype(np.float32)

        if sequence.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
            raise ValueError(f"Sequence shape {sequence.shape}")

        t0 = time.perf_counter()
        probs = model.predict(np.expand_dims(sequence, 0), verbose=0)[0]
        ms = (time.perf_counter() - t0) * 1000

        entries = top5(probs, names)
        label, conf = entries[0]
        raw_label = label if conf >= args.threshold else "Uncertain"
        stable_label, stable_conf = stabilizer.update(raw_label, conf, hand_detected=True)

        print(f"Segment {n}: RAW    {raw_label:25s} {conf*100:6.2f}%")
        print(f"           STABLE {stable_label:25s} {stable_conf*100:6.2f}%")
        print("           TOP5:")
        for lab, c in entries:
            print(f"             {lab:25s} {c*100:6.2f}%")
        print(f"           source frames: {s+1}-{e+1}; resampled to 60; {ms:.1f} ms")

        results.append({
            "segment": n,
            "source_start_frame": s + 1,
            "source_end_frame": e + 1,
            "source_length": e - s + 1,
            "raw_label": raw_label,
            "raw_confidence": conf,
            "stable_label": stable_label,
            "stable_confidence": stable_conf,
            "top5": [{"label": lab, "confidence": c} for lab, c in entries],
            "inference_ms": ms,
        })

    output = {
        "video": str(video),
        "fps": fps,
        "total_frames": total,
        "hand_detected_frames": int(flags.sum()),
        "max_gap": args.max_gap,
        "sequence_length": SEQUENCE_LENGTH,
        "results": results,
    }

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\nJSON saved: {out}")


if __name__ == "__main__":
    main()
