from pathlib import Path
import json
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

from src.training.hand_landmark_interpolator import HandLandmarkInterpolator
from src.training.landmark_normalizer import LandmarkNormalizer
from src.training.sequence_generator import SequenceGenerator


ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = (
    ROOT
    / "outputs"
    / "external_tests"
    / "isl_test_video.mp4"
)

MODEL_PATH = (
    ROOT
    / "saved_models"
    / "isl_lstm_generalized_114_legacy.keras"
)

MAPPING_PATH = (
    ROOT
    / "outputs"
    / "generalized_114_label_mapping.json"
)

LANDMARK_OUTPUT = (
    ROOT
    / "outputs"
    / "external_tests"
    / "isl_test_video_landmarks.npy"
)

RESULT_OUTPUT = (
    ROOT
    / "outputs"
    / "external_tests"
    / "isl_test_video_predictions.txt"
)


# ------------------------------------------------------------
# MediaPipe Holistic configuration
# ------------------------------------------------------------

mp_holistic = mp.solutions.holistic


def extract_landmarks():

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{VIDEO_PATH}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print("\nVIDEO INFORMATION")
    print("-" * 60)
    print("Resolution:", width, "x", height)
    print("FPS:", fps)
    print("Frames reported:", frame_count)

    landmarks = []

    pose_detected = 0
    left_detected = 0
    right_detected = 0
    both_detected = 0

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        frame_index = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            result = holistic.process(
                rgb
            )

            # ------------------------------------------------
            # Pose: 33 × 4
            # ------------------------------------------------

            pose = []

            if result.pose_landmarks:

                pose_detected += 1

                for lm in result.pose_landmarks.landmark:

                    pose.extend([
                        lm.x,
                        lm.y,
                        lm.z,
                        lm.visibility,
                    ])

            else:

                pose = [0.0] * 132

            # ------------------------------------------------
            # Left hand: 21 × 3
            # ------------------------------------------------

            left_hand = []

            if result.left_hand_landmarks:

                left_detected += 1

                for lm in result.left_hand_landmarks.landmark:

                    left_hand.extend([
                        lm.x,
                        lm.y,
                        lm.z,
                    ])

            else:

                left_hand = [0.0] * 63

            # ------------------------------------------------
            # Right hand: 21 × 3
            # ------------------------------------------------

            right_hand = []

            if result.right_hand_landmarks:

                right_detected += 1

                for lm in result.right_hand_landmarks.landmark:

                    right_hand.extend([
                        lm.x,
                        lm.y,
                        lm.z,
                    ])

            else:

                right_hand = [0.0] * 63

            if (
                result.left_hand_landmarks
                and result.right_hand_landmarks
            ):
                both_detected += 1

            features = (
                pose
                + left_hand
                + right_hand
            )

            if len(features) != 258:

                raise RuntimeError(
                    f"Frame {frame_index}: "
                    f"expected 258 features, "
                    f"got {len(features)}"
                )

            landmarks.append(
                features
            )

            frame_index += 1

    cap.release()

    landmarks = np.asarray(
        landmarks,
        dtype=np.float32
    )

    np.save(
        LANDMARK_OUTPUT,
        landmarks
    )

    total = len(landmarks)

    print("\nLANDMARK EXTRACTION")
    print("-" * 60)

    print("Extracted frames:", total)
    print("Landmark shape:", landmarks.shape)

    if total > 0:

        print(
            f"Pose detected:       "
            f"{pose_detected}/{total} "
            f"({pose_detected / total * 100:.2f}%)"
        )

        print(
            f"Left hand detected:  "
            f"{left_detected}/{total} "
            f"({left_detected / total * 100:.2f}%)"
        )

        print(
            f"Right hand detected: "
            f"{right_detected}/{total} "
            f"({right_detected / total * 100:.2f}%)"
        )

        print(
            f"Both hands detected: "
            f"{both_detected}/{total} "
            f"({both_detected / total * 100:.2f}%)"
        )

    print(
        "\nSaved landmarks:"
    )
    print(LANDMARK_OUTPUT)

    return landmarks


def load_mapping():

    with open(
        MAPPING_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        mapping = json.load(f)

    if all(
        str(k).isdigit()
        for k in mapping.keys()
    ):

        return {
            int(k): v
            for k, v in mapping.items()
        }

    return {
        int(v): k
        for k, v in mapping.items()
    }


def preprocess(raw):

    interpolator = HandLandmarkInterpolator(
        max_gap=5
    )

    normalizer = LandmarkNormalizer()

    generator = SequenceGenerator()

    x = interpolator.interpolate(
        raw
    )

    x = normalizer.normalize(
        x
    )

    x = generator.generate(
        x
    )

    return x


def run_predictions(landmarks):

    print("\n")
    print("=" * 70)
    print("MODEL PREDICTION TEST")
    print("=" * 70)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    mapping = load_mapping()

    total_frames = len(landmarks)

    sequence_length = 60

    # --------------------------------------------------------
    # Create overlapping windows.
    #
    # Step = 15 frames.
    # --------------------------------------------------------

    windows = []

    if total_frames < sequence_length:

        windows.append(
            (
                0,
                total_frames,
                landmarks
            )
        )

    else:

        start = 0

        while (
            start + sequence_length
            <= total_frames
        ):

            end = start + sequence_length

            windows.append(
                (
                    start,
                    end,
                    landmarks[start:end]
                )
            )

            start += 15

        # Include final window if necessary
        if (
            len(windows) == 0
            or windows[-1][1] != total_frames
        ):

            start = max(
                0,
                total_frames - sequence_length
            )

            windows.append(
                (
                    start,
                    total_frames,
                    landmarks[
                        start:total_frames
                    ]
                )
            )

    all_results = []

    for window_number, (
        start,
        end,
        raw
    ) in enumerate(windows, start=1):

        try:

            sequence = preprocess(
                raw
            )

            prediction = model.predict(
                sequence[np.newaxis, ...],
                verbose=0
            )[0]

            top_indices = np.argsort(
                prediction
            )[::-1][:5]

            top_results = []

            for idx in top_indices:

                label = mapping.get(
                    int(idx),
                    f"Class {idx}"
                )

                confidence = float(
                    prediction[idx] * 100
                )

                top_results.append(
                    (
                        label,
                        confidence
                    )
                )

            all_results.append(
                (
                    window_number,
                    start,
                    end,
                    top_results
                )
            )

            print("\n" + "-" * 70)

            print(
                f"Window {window_number}: "
                f"frames {start}–{end - 1}"
            )

            for rank, (
                label,
                confidence
            ) in enumerate(
                top_results,
                start=1
            ):

                print(
                    f"{rank}. "
                    f"{label:30s}"
                    f"{confidence:7.2f}%"
                )

        except Exception as e:

            print(
                f"\nWindow {window_number} "
                f"failed: {e}"
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    with open(
        RESULT_OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "ISL EXTERNAL VIDEO PREDICTION RESULTS\n"
        )
        f.write("=" * 70 + "\n\n")

        for (
            window_number,
            start,
            end,
            top_results
        ) in all_results:

            f.write(
                f"Window {window_number}: "
                f"frames {start}-{end - 1}\n"
            )

            for rank, (
                label,
                confidence
            ) in enumerate(
                top_results,
                start=1
            ):

                f.write(
                    f"{rank}. "
                    f"{label:30s}"
                    f"{confidence:7.2f}%\n"
                )

            f.write("\n")

    print("\n")
    print("=" * 70)
    print("RESULT FILE")
    print("=" * 70)

    print(RESULT_OUTPUT)


def main():

    print("=" * 70)
    print("ISL EXTERNAL VIDEO TEST")
    print("=" * 70)

    if not VIDEO_PATH.exists():

        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

    landmarks = extract_landmarks()

    if len(landmarks) == 0:

        raise RuntimeError(
            "No frames were extracted."
        )

    run_predictions(
        landmarks
    )


if __name__ == "__main__":
    main()