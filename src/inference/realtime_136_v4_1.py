"""
Indian Sign Language - 136 Class Real-Time V4
Frame-preserving, gap-tolerant gesture inference.

IMPORTANT
---------
- This is a NEW inference file.
- It does NOT modify src/inference/realtime.py.
- It does NOT modify the trained model.
- It uses the frozen targeted 136-class model.
- It preserves EVERY camera-frame position inside a gesture.
- Short no-hand gaps (<= MAX_INTERPOLATION_GAP) are retained and
  repaired by HandLandmarkInterpolator.
- Gestures shorter than 60 frames are supported through SequenceGenerator
  zero-padding.
- Gestures longer than 60 frames are resampled by SequenceGenerator.
- A longer no-hand period finalizes the current gesture.

Run from project root:
    python src\inference\realtime_136_v4.py

Press Q to quit.
"""

import json
import sys
import time
from pathlib import Path

from collections import deque

import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.legacy_landmark_extractor import LegacyLandmarkExtractor
from src.training.hand_landmark_interpolator import HandLandmarkInterpolator
from src.training.landmark_normalizer import LandmarkNormalizer
from src.training.sequence_generator import SequenceGenerator
from src.inference.temporal_stabilizer import TemporalStabilizer


# ============================================================
# FROZEN PRODUCTION MODEL
# ============================================================

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


# ============================================================
# MODEL / TEMPORAL CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 60
FEATURES = 258

CONFIDENCE_THRESHOLD = 0.50

# Run a prediction every N camera frames after minimum gesture
# length has been reached.
PREDICTION_INTERVAL = 3

# Do not try to classify extremely short accidental hand detections.
# 30 frames ~= 1 second at 30 FPS.
MIN_GESTURE_FRAMES = 30

# Short missing-hand gaps are part of the same gesture.
MAX_INTERPOLATION_GAP = 5

# A longer continuous no-hand period means the gesture has ended.
#
# Important:
#   5 frames = maximum interpolation gap
#   8 frames = gesture finalization threshold
#
# This prevents a single/noisy hand-detection failure from resetting
# the gesture.
GESTURE_END_GAP = 8

# Prevent an excessively long live buffer.
MAX_GESTURE_FRAMES = 180


# ============================================================
# LABELS
# ============================================================

def load_labels():
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(MAPPING_FILE)

    raw = json.loads(
        MAPPING_FILE.read_text(
            encoding="utf-8"
        )
    )

    # Support:
    #   index -> label
    # or:
    #   label -> index
    if all(str(k).isdigit() for k in raw.keys()):
        labels = {
            int(k): str(v)
            for k, v in raw.items()
        }
    else:
        labels = {
            int(v): str(k)
            for k, v in raw.items()
        }

    if len(labels) != 136:
        raise RuntimeError(
            f"Expected 136 labels, found {len(labels)}"
        )

    return labels


# ============================================================
# HELPERS
# ============================================================

def hand_detected(landmarks):
    """
    Hand is considered present if either hand has non-zero
    63-feature landmark data.

    Pose-only data does not count as hand presence.
    """
    return not np.allclose(
        landmarks[132:258],
        0.0,
    )


def confidence_quality(confidence, label):
    if label == "Uncertain" or confidence < CONFIDENCE_THRESHOLD:
        return "UNCERTAIN"

    if confidence < 0.70:
        return "MEDIUM"

    return "HIGH"


def format_label(label):
    if label == "Uncertain":
        return label

    return label


# ============================================================
# REAL-TIME ENGINE
# ============================================================

def main():

    print("=" * 78)
    print("INDIAN SIGN LANGUAGE - 136 CLASS REAL-TIME V4")
    print("=" * 78)

    print(f"Model       : {MODEL_PATH}")
    print(f"Mapping     : {MAPPING_FILE}")
    print(f"Sequence    : {SEQUENCE_LENGTH} frames")
    print(f"Features    : {FEATURES}")
    print(f"Threshold   : {CONFIDENCE_THRESHOLD:.2f}")
    print(
        f"Min gesture : {MIN_GESTURE_FRAMES} frames"
    )
    print(
        f"Max gap     : {MAX_INTERPOLATION_GAP} frames"
    )
    print(
        f"End gap     : {GESTURE_END_GAP} frames"
    )
    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    print("Loading targeted 136-class model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    labels = load_labels()

    if model.output_shape[-1] != len(labels):
        raise RuntimeError(
            f"Model outputs {model.output_shape[-1]} classes, "
            f"but mapping contains {len(labels)} labels."
        )

    print(
        f"Model compatibility: PASS "
        f"({len(labels)} classes)"
    )

    # --------------------------------------------------------
    # Preprocessing objects
    # --------------------------------------------------------

    print("Initializing MediaPipe Holistic...")

    extractor = LegacyLandmarkExtractor()

    interpolator = HandLandmarkInterpolator(
        max_gap=MAX_INTERPOLATION_GAP
    )

    normalizer = LandmarkNormalizer()
    generator = SequenceGenerator()

    stabilizer = TemporalStabilizer(
        history_size=5,
        initial_min_votes=2,
        transition_min_votes=3,
    )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        extractor.close()
        raise RuntimeError(
            "Could not open webcam."
        )

    print()
    print("Webcam started.")
    print("Press Q to quit.")
    print()

    # ========================================================
    # Gesture state
    # ========================================================

    # IMPORTANT:
    # Unlike the old pipeline, this buffer stores EVERY frame
    # while the gesture is active, including frames where the
    # hand is temporarily not detected.
    gesture_buffer = []

    consecutive_no_hand = 0

    prediction_counter = 0

    raw_label = "Uncertain"
    raw_confidence = 0.0

    stable_label = "Uncertain"
    stable_confidence = 0.0

    last_inference_ms = 0.0

    gesture_number = 0

    gesture_state = "WAITING"

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    fps_start = time.perf_counter()
    fps_frames = 0
    fps = 0.0

    # --------------------------------------------------------
    # Window
    # --------------------------------------------------------

    window_name = (
        "Indian Sign Language - 136 Class V4"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            fps_frames += 1

            # =================================================
            # LANDMARK EXTRACTION
            # =================================================

            landmarks = extractor.extract(frame)

            if landmarks.shape != (FEATURES,):
                raise RuntimeError(
                    f"Unexpected landmark shape: "
                    f"{landmarks.shape}"
                )

            has_hand = hand_detected(
                landmarks
            )

            # =================================================
            # EVERY FRAME IS RETAINED
            # =================================================

            if gesture_state == "WAITING":

                if has_hand:

                    gesture_number += 1
                    gesture_state = "ACTIVE"

                    gesture_buffer = []
                    consecutive_no_hand = 0
                    prediction_counter = 0

                    raw_label = "Uncertain"
                    raw_confidence = 0.0

                    # Start the new gesture with THIS frame.
                    gesture_buffer.append(
                        landmarks.copy()
                    )

                else:

                    # Nothing to classify yet.
                    stable_label = "Uncertain"
                    stable_confidence = 0.0

            elif gesture_state == "ACTIVE":

                # CRITICAL:
                # Always append the frame.
                #
                # A missing hand is represented by the normal
                # zero landmark frame at its ORIGINAL position.
                gesture_buffer.append(
                    landmarks.copy()
                )

                if has_hand:

                    consecutive_no_hand = 0

                else:

                    consecutive_no_hand += 1

                prediction_counter += 1

                # =================================================
                # GESTURE FINALIZATION
                # =================================================

                if (
                    consecutive_no_hand
                    >= GESTURE_END_GAP
                ):

                    # Classify the completed gesture one final time
                    # before clearing it.
                    if len(gesture_buffer) >= MIN_GESTURE_FRAMES:

                        raw_data = np.asarray(
                            gesture_buffer,
                            dtype=np.float32,
                        )

                        try:

                            interpolated = (
                                interpolator.interpolate(
                                    raw_data
                                )
                            )

                            normalized = (
                                normalizer.normalize(
                                    interpolated
                                )
                            )

                            sequence = (
                                generator.generate(
                                    normalized
                                )
                            )

                            start = time.perf_counter()

                            probabilities = model.predict(
                                np.expand_dims(
                                    sequence,
                                    axis=0,
                                ),
                                verbose=0,
                            )[0]

                            last_inference_ms = (
                                time.perf_counter()
                                - start
                            ) * 1000.0

                            index = int(
                                np.argmax(
                                    probabilities
                                )
                            )

                            confidence = float(
                                probabilities[index]
                            )

                            label = (
                                labels[index]
                                if confidence
                                >= CONFIDENCE_THRESHOLD
                                else "Uncertain"
                            )

                            raw_label = label
                            raw_confidence = confidence

                            stable_label, stable_confidence = (
                                stabilizer.update(
                                    label,
                                    confidence,
                                    hand_detected=True,
                                )
                            )

                        except Exception as exc:

                            print(
                                f"\nGesture finalization "
                                f"warning: {exc}"
                            )

                    # Reset AFTER finalizing.
                    gesture_buffer = []
                    consecutive_no_hand = 0
                    prediction_counter = 0
                    gesture_state = "WAITING"

                # =================================================
                # LIVE PARTIAL GESTURE PREDICTION
                # =================================================

                elif (
                    has_hand
                    and
                    len(gesture_buffer)
                    >= MIN_GESTURE_FRAMES
                    and
                    prediction_counter
                    >= PREDICTION_INTERVAL
                ):

                    raw_data = np.asarray(
                        gesture_buffer,
                        dtype=np.float32,
                    )

                    # Prevent unlimited processing.
                    #
                    # If a gesture becomes very long, keep the
                    # most recent temporal span.
                    if (
                        len(raw_data)
                        > MAX_GESTURE_FRAMES
                    ):
                        raw_data = raw_data[
                            -MAX_GESTURE_FRAMES:
                        ]

                    try:

                        # SAME proven preprocessing as V4 test.
                        interpolated = (
                            interpolator.interpolate(
                                raw_data
                            )
                        )

                        normalized = (
                            normalizer.normalize(
                                interpolated
                            )
                        )

                        sequence = (
                            generator.generate(
                                normalized
                            )
                        )

                        start = time.perf_counter()

                        probabilities = model.predict(
                            np.expand_dims(
                                sequence,
                                axis=0,
                            ),
                            verbose=0,
                        )[0]

                        last_inference_ms = (
                            time.perf_counter()
                            - start
                        ) * 1000.0

                        index = int(
                            np.argmax(
                                probabilities
                            )
                        )

                        confidence = float(
                            probabilities[index]
                        )

                        label = (
                            labels[index]
                            if confidence
                            >= CONFIDENCE_THRESHOLD
                            else "Uncertain"
                        )

                        raw_label = label
                        raw_confidence = confidence

                        stable_label, stable_confidence = (
                            stabilizer.update(
                                label,
                                confidence,
                                hand_detected=True,
                            )
                        )

                    except Exception as exc:

                        print(
                            f"\nInference warning: {exc}"
                        )

                    prediction_counter = 0

            # =================================================
            # FRAME-BASED STATE
            # =================================================

            buffer_length = len(
                gesture_buffer
            )

            if gesture_state == "WAITING":

                status = "NO HAND / WAITING"

            elif consecutive_no_hand > 0:

                status = (
                    f"GAP {consecutive_no_hand}/"
                    f"{GESTURE_END_GAP}"
                )

            elif buffer_length < MIN_GESTURE_FRAMES:

                status = "COLLECTING"

            else:

                status = confidence_quality(
                    stable_confidence,
                    stable_label,
                )

            # =================================================
            # FPS
            # =================================================

            now = time.perf_counter()
            elapsed = now - fps_start

            if elapsed >= 1.0:

                fps = (
                    fps_frames / elapsed
                )

                fps_frames = 0
                fps_start = now

            # =================================================
            # DISPLAY
            # =================================================

            display = frame.copy()

            # Header
            cv2.putText(
                display,
                "ISL 136-CLASS V4",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Hand
            hand_text = (
                "HAND: YES"
                if has_hand
                else "HAND: NO"
            )

            cv2.putText(
                display,
                hand_text,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0)
                if has_hand
                else (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

            # Buffer
            cv2.putText(
                display,
                (
                    f"BUFFER: {buffer_length} "
                    f"frames"
                ),
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # State
            cv2.putText(
                display,
                f"STATE: {status}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Raw
            cv2.putText(
                display,
                (
                    f"RAW: {format_label(raw_label)} "
                    f"{raw_confidence:.1%}"
                ),
                (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Stable
            cv2.putText(
                display,
                (
                    f"STABLE: "
                    f"{format_label(stable_label)} "
                    f"{stable_confidence:.1%}"
                ),
                (20, 225),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 0)
                if stable_label != "Uncertain"
                else (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

            # Gesture
            cv2.putText(
                display,
                f"GESTURE #: {gesture_number}",
                (20, 265),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Performance
            cv2.putText(
                display,
                f"FPS: {fps:.1f}",
                (20, 305),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                f"INFERENCE: {last_inference_ms:.1f} ms",
                (20, 340),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Pipeline explanation
            cv2.putText(
                display,
                "258 features | 60-frame model",
                (20, 385),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                (
                    "Short gaps retained | "
                    "SequenceGenerator active"
                ),
                (20, 415),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                "Q = Quit",
                (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                window_name,
                display,
            )

            # =================================================
            # KEYBOARD
            # =================================================

            key = cv2.waitKey(1) & 0xFF

            if key in (
                ord("q"),
                ord("Q"),
            ):
                # Finalize the currently active gesture before quitting.
                # This is especially useful when the camera recording or
                # test ends during the GESTURE_END_GAP window.
                if (
                    gesture_state == "ACTIVE"
                    and len(gesture_buffer) >= MIN_GESTURE_FRAMES
                ):
                    raw_data = np.asarray(
                        gesture_buffer,
                        dtype=np.float32,
                    )

                    if len(raw_data) > MAX_GESTURE_FRAMES:
                        raw_data = raw_data[-MAX_GESTURE_FRAMES:]

                    try:
                        interpolated = interpolator.interpolate(raw_data)
                        normalized = normalizer.normalize(interpolated)
                        sequence = generator.generate(normalized)

                        start = time.perf_counter()

                        probabilities = model.predict(
                            np.expand_dims(sequence, axis=0),
                            verbose=0,
                        )[0]

                        final_inference_ms = (
                            time.perf_counter() - start
                        ) * 1000.0

                        index = int(np.argmax(probabilities))
                        confidence = float(probabilities[index])

                        final_label = (
                            labels[index]
                            if confidence >= CONFIDENCE_THRESHOLD
                            else "Uncertain"
                        )

                        print()
                        print("FINAL GESTURE BEFORE QUIT")
                        print(f"  Gesture #: {gesture_number}")
                        print(f"  Frames    : {len(gesture_buffer)}")
                        print(f"  Prediction: {final_label}")
                        print(f"  Confidence: {confidence:.2%}")
                        print(f"  Inference : {final_inference_ms:.1f} ms")

                    except Exception as exc:
                        print(
                            f"\nFinal gesture classification warning: {exc}"
                        )

                break

    finally:

        cap.release()

        cv2.destroyAllWindows()

        extractor.close()

    print()
    print("=" * 78)
    print("REAL-TIME V4 STOPPED")
    print("=" * 78)


if __name__ == "__main__":
    main()
