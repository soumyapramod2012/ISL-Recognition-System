from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from src.training.config import (
    TEST_SIZE,
    RANDOM_STATE,
    LANDMARK_SIZE,
)

from src.training.dataset_loader import DatasetLoader
from src.training.label_encoder import LabelEncoder
from src.training.sequence_generator import SequenceGenerator
from src.training.landmark_normalizer import LandmarkNormalizer
from src.training.hand_landmark_interpolator import (
    HandLandmarkInterpolator,
)


DATASET_DIR = Path(
    "dataset/processed/landmarks"
)

MODEL_PATH = Path(
    "saved_models/isl_lstm.keras"
)

RANDOM_SEED = 42


# ============================================================
# Dataset
# ============================================================

def load_raw_samples():

    loader = DatasetLoader(
        DATASET_DIR
    )

    samples = loader.load()

    encoder = LabelEncoder()
    encoder.fit(samples)

    return samples, encoder


def build_test_samples():

    samples, encoder = load_raw_samples()

    test_samples = []

    # We reproduce the exact v1.5.0 split.
    indices = np.arange(len(samples))

    labels = np.array(
        [
            encoder.encode(
                sample["label"]
            )
            for sample in samples
        ],
        dtype=np.int32,
    )

    _, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    for index in test_indices:

        test_samples.append(
            samples[index]
        )

    return test_samples, encoder


# ============================================================
# Raw landmark validation
# ============================================================

def load_landmarks(path):

    landmarks = np.load(path)

    if landmarks.ndim != 2:
        raise ValueError(
            "Landmarks must be a 2D array."
        )

    if landmarks.shape[1] != LANDMARK_SIZE:
        raise ValueError(
            f"Expected {LANDMARK_SIZE} features, "
            f"found {landmarks.shape[1]}."
        )

    if landmarks.shape[0] == 0:
        raise ValueError(
            "Empty landmark sequence."
        )

    if np.isnan(landmarks).any():
        raise ValueError(
            "Landmarks contain NaN values."
        )

    if np.isinf(landmarks).any():
        raise ValueError(
            "Landmarks contain infinite values."
        )

    return landmarks.astype(
        np.float32,
        copy=True,
    )


# ============================================================
# Artificial hand degradation
# ============================================================

POSE_FEATURES = 33 * 4
HAND_FEATURES = 21 * 3

LEFT_HAND_START = POSE_FEATURES
RIGHT_HAND_START = (
    POSE_FEATURES + HAND_FEATURES
)

TOTAL_FEATURES = (
    POSE_FEATURES
    + HAND_FEATURES
    + HAND_FEATURES
)


def create_degradation_mask(
    frame_count,
    gap_length,
    rng,
):
    """
    Create deterministic missing-frame masks
    for left and right hands.

    True  = remove hand landmarks
    False = keep original landmarks

    The same masks are reused for both:
        1. No interpolation
        2. Interpolation
    """

    left_mask = np.zeros(
        frame_count,
        dtype=bool,
    )

    right_mask = np.zeros(
        frame_count,
        dtype=bool,
    )

    if gap_length <= 0:
        return (
            left_mask,
            right_mask,
        )

    if frame_count <= gap_length:
        return (
            left_mask,
            right_mask,
        )

    # Choose a random starting position
    # for one left-hand gap.
    left_start = int(
        rng.integers(
            0,
            frame_count - gap_length + 1,
        )
    )

    left_mask[
        left_start:
        left_start + gap_length
    ] = True

    # Choose an independent random starting
    # position for the right-hand gap.
    right_start = int(
        rng.integers(
            0,
            frame_count - gap_length + 1,
        )
    )

    right_mask[
        right_start:
        right_start + gap_length
    ] = True

    return (
        left_mask,
        right_mask,
    )


def apply_hand_degradation(
    landmarks,
    left_mask,
    right_mask,
):

    degraded = landmarks.copy()

    # Remove complete hand landmarks
    # for selected frames.

    degraded[
        left_mask,
        LEFT_HAND_START:
        RIGHT_HAND_START
    ] = 0.0

    degraded[
        right_mask,
        RIGHT_HAND_START:
    ] = 0.0

    return degraded


def create_degraded_dataset(
    samples,
    encoder,
    gap_length,
    seed,
):
    """
    Create one deterministic degraded dataset.

    Returns:
        X_raw
        y
        degradation_masks

    degradation_masks contains the artificial
    left/right hand corruption masks for every sample.

    """

    X_raw = []
    y = []
    degradation_masks = []
    original_landmarks = []

    rng = np.random.default_rng(seed)

    for sample in samples:

        landmarks = load_landmarks(
            sample["path"]
        )

        original_landmarks.append(
            landmarks.copy()
        )

        left_mask, right_mask = (
            create_degradation_mask(
                frame_count=landmarks.shape[0],
                gap_length=gap_length,
                rng=rng,
            )
        )        
        
        degraded = apply_hand_degradation(
            landmarks,
            left_mask,
            right_mask,
        )

        X_raw.append(degraded)

        degradation_masks.append(
            (
                left_mask.copy(),
                right_mask.copy(),
            )
        )

        y.append(
            encoder.encode(
                sample["label"]
            )
        )

    return(
        X_raw,
        np.asarray(y, dtype=np.int32),
        degradation_masks,
        original_landmarks,
    ) 
        

def preprocess_dataset(
    raw_samples,
    use_interpolation,
):
    """
    Apply preprocessing to an already-created
    degraded dataset.
    """

    X = []

    for landmarks in raw_samples:

        sequence = preprocess(
            landmarks,
            use_interpolation=use_interpolation,
        )

        X.append(sequence)

    return np.asarray(
        X,
        dtype=np.float32,
    )


# ============================================================
# Preprocessing
# ============================================================

def preprocess(
    landmarks,
    use_interpolation,
):

    data = landmarks.copy()

    if use_interpolation:

        interpolator = (
            HandLandmarkInterpolator(
                max_gap=5
            )
        )

        data = interpolator.interpolate(
            data
        )

    normalizer = LandmarkNormalizer()

    data = normalizer.normalize(
        data
    )

    generator = SequenceGenerator()

    sequence = generator.generate(
        data
    )

    return sequence


# ============================================================
# Model evaluation
# ============================================================


def evaluate_dataset(
    model,
    X,
    y,
):

    probabilities = model.predict(
        X,
        verbose=0,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    correct = (
        predictions == y
    )

    return {
        "accuracy": float(
            np.mean(correct)
        ),
        "correct": int(
            correct.sum()
        ),
        "total": int(
            len(y)
        ),
        "probabilities": probabilities,
        "predictions": predictions,
        "labels": y,
    }


# ============================================================
# Repair diagnostics
# ============================================================

def count_missing_hand_frames(landmarks):
    """
    Count frames where either hand is completely missing.
    """

    landmarks = np.asarray(
        landmarks,
        dtype=np.float32,
    )

    pose_features = 33 * 4
    hand_features = 21 * 3

    left_start = pose_features
    left_end = (
        left_start
        + hand_features
    )

    right_start = left_end
    right_end = (
        right_start
        + hand_features
    )

    left_hand = landmarks[
        :,
        left_start:left_end,
    ].reshape(
        landmarks.shape[0],
        21,
        3,
    )

    right_hand = landmarks[
        :,
        right_start:right_end,
    ].reshape(
        landmarks.shape[0],
        21,
        3,
    )

    left_missing = np.all(
        left_hand == 0,
        axis=(1, 2),
    )

    right_missing = np.all(
        right_hand == 0,
        axis=(1, 2),
    )

    return int(
        left_missing.sum()
        + right_missing.sum()
    )


def count_artificial_repair(
    original_landmarks,
    interpolated_landmarks,
    left_mask,
    right_mask,
):
    """
    Measure repair of artificially corrupted hand frames.

    Only frames where the hand was originally present are
    considered valid artificial corruption targets.

    Returns:
        artificial_missing:
            Number of artificially corrupted frames that
            originally contained a valid complete hand.

        artificial_repaired:
            Number of those frames whose hand is no longer
            completely missing after interpolation.
    """

    artificial_missing = 0
    artificial_repaired = 0

    for frame_index in range(
        original_landmarks.shape[0]
    ):

        # -------------------------------------------------
        # LEFT HAND
        # -------------------------------------------------

        if left_mask[frame_index]:

            original_left = (
                original_landmarks[
                    frame_index,
                    LEFT_HAND_START:
                    RIGHT_HAND_START
                ]
            )

            interpolated_left = (
                interpolated_landmarks[
                    frame_index,
                    LEFT_HAND_START:
                    RIGHT_HAND_START
                ]
            )

            # Only count this as artificial corruption
            # if the hand originally existed.
            if not np.allclose(
                original_left,
                0.0,
            ):

                artificial_missing += 1

                if not np.allclose(
                    interpolated_left,
                    0.0,
                ):
                    artificial_repaired += 1

        # -------------------------------------------------
        # RIGHT HAND
        # -------------------------------------------------

        if right_mask[frame_index]:

            original_right = (
                original_landmarks[
                    frame_index,
                    RIGHT_HAND_START:
                ]
            )

            interpolated_right = (
                interpolated_landmarks[
                    frame_index,
                    RIGHT_HAND_START:
                ]
            )

            # Only count this as artificial corruption
            # if the hand originally existed.
            if not np.allclose(
                original_right,
                0.0,
            ):

                artificial_missing += 1

                if not np.allclose(
                    interpolated_right,
                    0.0,
                ):
                    artificial_repaired += 1

    return {
        "artificial_missing":
            artificial_missing,

        "artificial_repaired":
            artificial_repaired,
    }


def analyze_repair_effect(
    raw_degraded,
    interpolated,
):
    """
    Compare hand landmark missingness before and
    after interpolation.
    """

    missing_before = (
        count_missing_hand_frames(
            raw_degraded
        )
    )

    missing_after = (
        count_missing_hand_frames(
            interpolated
        )
    )

    repaired = (
        missing_before
        - missing_after
    )

    return {
        "missing_before":
            missing_before,

        "missing_after":
            missing_after,

        "repaired":
            max(repaired, 0),
    }


# ============================================================
# Multi-trial robustness experiment
# ============================================================

def run_multi_trial_experiment(
    model,
    samples,
    encoder,
    gap_lengths,
    trial_seeds,
):
    """
    Run paired robustness experiments across multiple
    deterministic trials.

    For every trial and gap length, the exact same degraded
    landmarks are used for both:

        1. Without interpolation
        2. With interpolation
    """

    results = []

    for gap_length in gap_lengths:

        print()
        print("=" * 80)
        print(
            f"GAP LENGTH: {gap_length}"
        )
        print("=" * 80)

        for trial_number, seed in enumerate(
            trial_seeds,
            start=1,
        ):

            # ------------------------------------------------
            # Create degradation ONCE
            # ------------------------------------------------

            raw_degraded, y, degradation_masks, original_landmarks = (
                create_degraded_dataset(
                    samples=samples,
                    encoder=encoder,
                    gap_length=gap_length,
                    seed=seed,
                )
            )

            # ------------------------------------------------
            # Repair diagnostics
            # ------------------------------------------------

            '''raw_missing = (
                count_missing_hand_frames(
                    raw_degraded
                )
            )'''

            # ------------------------------------------------
            # Without interpolation
            # ------------------------------------------------

            X_without = preprocess_dataset(
                raw_degraded,
                use_interpolation=False,
            )

            without_result = (
                evaluate_dataset(
                    model=model,
                    X=X_without,
                    y=y,
                )
            )

            # ------------------------------------------------
            # With interpolation
            # ------------------------------------------------

            X_with = preprocess_dataset(
                raw_degraded,
                use_interpolation=True,
            )
                        
            # ------------------------------------------------
            # Repair diagnostics
            # ------------------------------------------------  

            interpolator = HandLandmarkInterpolator(
                max_gap=5
            )

            missing_before = 0
            missing_after = 0

            artificial_missing = 0
            artificial_repaired = 0
            
            for index, landmarks in enumerate(
                raw_degraded
            ):

                missing_before += (
                    count_missing_hand_frames(
                        landmarks
                    )
                )
                
                repaired_landmarks = (
                    interpolator.interpolate(
                        landmarks
                    )
                )

                missing_after += (
                    count_missing_hand_frames(
                        repaired_landmarks
                    )
                )

                original = original_landmarks[index]
                
                left_mask, right_mask = (
                    degradation_masks[index]
                )

                artificial_stats = (
                    count_artificial_repair(
                        original_landmarks=original,
                        interpolated_landmarks=repaired_landmarks,
                        left_mask=left_mask,
                        right_mask=right_mask,
                    )
                )

                artificial_missing += (
                    artificial_stats[
                        "artificial_missing"
                    ]
                )

                artificial_repaired += (
                    artificial_stats[
                        "artificial_repaired"
                    ]
                )
                        
            repair_stats = {
                "missing_before": missing_before,
                "missing_after": missing_after,
                "repaired": max(
                    missing_before - missing_after,
                    0,
                ),

                "artificial_missing": artificial_missing,
                "artificial_repaired": artificial_repaired,            

            }
            
            with_result = (
                evaluate_dataset(
                    model=model,
                    X=X_with,
                    y=y,
                )
            )

            improvement = (
                with_result["accuracy"]
                - without_result["accuracy"]
            )

            result = {
                "gap_length": gap_length,
                "trial": trial_number,
                "seed": seed,

                "missing_before":
                    repair_stats["missing_before"],

                "missing_after":
                    repair_stats["missing_after"],

                "repaired_frames":
                    repair_stats["repaired"],

                "artificial_missing":
                    repair_stats[
                        "artificial_missing"
                    ],

                "artificial_repaired":
                    repair_stats[
                        "artificial_repaired"
                    ],
                
                "without_accuracy":
                    without_result["accuracy"],

                "with_accuracy":
                    with_result["accuracy"],

                "improvement":
                    improvement,

                "without_correct":
                    without_result["correct"],

                "with_correct":
                    with_result["correct"],

                "total":
                    without_result["total"],
            }

            results.append(result)

            print(
                f"Trial {trial_number:2d} "
                f"(seed={seed})  "
                f"Without="
                f"{without_result['accuracy']:.4f} "
                f"({without_result['correct']}/"
                f"{without_result['total']})  "
                f"With="
                f"{with_result['accuracy']:.4f} "
                f"({with_result['correct']}/"
                f"{with_result['total']})  "
                f"Improvement="
                f"{improvement:+.4f}"
                f"Missing="
                f"{repair_stats['missing_before']}→"
                f"{repair_stats['missing_after']}  "
                f"Repaired="
                f"{repair_stats['repaired']}"
                f"Artificial="
                f"{repair_stats['artificial_missing']}  "
                f"Artificial Repaired="
                f"{repair_stats['artificial_repaired']}"
            )

    return results


def summarize_multi_trial_results(
    results,
    gap_lengths,
):
    """
    Calculate mean, standard deviation and trial-level
    improvement statistics for each degradation level.
    """

    print()
    print("=" * 80)
    print("MULTI-TRIAL ROBUSTNESS SUMMARY")
    print("=" * 80)

    print()
    print(
        f"{'Gap':<8}"
        f"{'Without Mean':<16}"
        f"{'Without Std':<14}"
        f"{'With Mean':<16}"
        f"{'With Std':<14}"
        f"{'Improvement':<14}"
        f"{'Improved':<10}"
        f"{'Equal':<8}"
        f"{'Worse':<8}"
    )

    print("-" * 110)

    for gap_length in gap_lengths:

        gap_results = [
            result
            for result in results
            if result["gap_length"]
            == gap_length
        ]

        without_values = np.array(
            [
                result[
                    "without_accuracy"
                ]
                for result in gap_results
            ],
            dtype=np.float64,
        )

        with_values = np.array(
            [
                result[
                    "with_accuracy"
                ]
                for result in gap_results
            ],
            dtype=np.float64,
        )

        improvements = (
            with_values
            - without_values
        )

        mean_without = np.mean(
            without_values
        )

        std_without = np.std(
            without_values,
            ddof=1,
        )

        mean_with = np.mean(
            with_values
        )

        std_with = np.std(
            with_values,
            ddof=1,
        )

        mean_improvement = np.mean(
            improvements
        )

        improved = int(
            np.sum(
                improvements > 0
            )
        )

        equal = int(
            np.sum(
                improvements == 0
            )
        )

        worse = int(
            np.sum(
                improvements < 0
            )
        )

        print(
            f"{gap_length:<8}"
            f"{mean_without:.4f}"
            f"{'':<10}"
            f"{std_without:.4f}"
            f"{'':<8}"
            f"{mean_with:.4f}"
            f"{'':<10}"
            f"{std_with:.4f}"
            f"{'':<8}"
            f"{mean_improvement:+.4f}"
            f"{'':<7}"
            f"{improved:<10}"
            f"{equal:<8}"
            f"{worse:<8}"
        )


# ============================================================
# Per-sample robustness analysis
# ============================================================

def run_per_sample_analysis(
    model,
    samples,
    encoder,
    gap_lengths,
    seed=42,
):
    """
    Analyze individual test samples to determine which
    predictions are recovered, unchanged, or harmed by
    landmark interpolation.

    Uses the same deterministic degradation seed for all
    gap lengths.
    """

    print()
    print("=" * 80)
    print("STEP 6 - PER-SAMPLE ROBUSTNESS ANALYSIS")
    print("=" * 80)

    for gap_length in gap_lengths:

        print()
        print("=" * 80)
        print(f"GAP LENGTH: {gap_length}  (seed={seed})")
        print("=" * 80)

        # ----------------------------------------------------
        # Create exactly one degraded dataset
        # ----------------------------------------------------

        (
            raw_degraded,
            y,
            degradation_masks,
            original_landmarks,
        ) = create_degraded_dataset(
            samples=samples,
            encoder=encoder,
            gap_length=gap_length,
            seed=seed,
        )

        # ----------------------------------------------------
        # WITHOUT interpolation
        # ----------------------------------------------------

        X_without = preprocess_dataset(
            raw_degraded,
            use_interpolation=False,
        )

        without_result = evaluate_dataset(
            model=model,
            X=X_without,
            y=y,
        )

        # ----------------------------------------------------
        # WITH interpolation
        # ----------------------------------------------------

        X_with = preprocess_dataset(
            raw_degraded,
            use_interpolation=True,
        )

        with_result = evaluate_dataset(
            model=model,
            X=X_with,
            y=y,
        )

        # ----------------------------------------------------
        # Compare individual predictions
        # ----------------------------------------------------

        without_predictions = (
            without_result["predictions"]
        )

        with_predictions = (
            with_result["predictions"]
        )

        probabilities_without = (
            without_result["probabilities"]
        )

        probabilities_with = (
            with_result["probabilities"]
        )

        changed_count = 0
        recovered_count = 0
        harmed_count = 0

        for index, sample in enumerate(samples):

            true_label = y[index]

            prediction_without = (
                without_predictions[index]
            )

            prediction_with = (
                with_predictions[index]
            )

            correct_without = (
                prediction_without == true_label
            )

            correct_with = (
                prediction_with == true_label
            )

            # Only print samples where the prediction
            # changed OR where interpolation affected
            # correctness.
            if prediction_without != prediction_with:

                changed_count += 1

                if (
                    not correct_without
                    and correct_with
                ):
                    status = "RECOVERED"
                    recovered_count += 1

                elif (
                    correct_without
                    and not correct_with
                ):
                    status = "HARMED"
                    harmed_count += 1

                else:
                    status = "CHANGED"

                true_name = (
                    encoder.classes[true_label]
                )

                without_name = (
                    encoder.classes[prediction_without]
                )

                with_name = (
                    encoder.classes[prediction_with]
                )

                confidence_without = float(
                    probabilities_without[
                        index,
                        prediction_without,
                    ]
                )

                confidence_with = float(
                    probabilities_with[
                        index,
                        prediction_with,
                    ]
                )

                print()
                print(
                    f"Sample {index + 1}"
                )

                print(
                    f"File       : "
                    f"{sample['path']}"
                )

                print(
                    f"True       : "
                    f"{true_name}"
                )

                print(
                    f"Without    : "
                    f"{without_name} "
                    f"({confidence_without:.4f})"
                )

                print(
                    f"With       : "
                    f"{with_name} "
                    f"({confidence_with:.4f})"
                )

                print(
                    f"Status     : {status}"
                )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print()
        print("-" * 80)
        print(
            f"Changed predictions : "
            f"{changed_count}/{len(samples)}"
        )

        print(
            f"Recovered           : "
            f"{recovered_count}"
        )

        print(
            f"Harmed              : "
            f"{harmed_count}"
        )

        print(
            f"Without accuracy    : "
            f"{without_result['accuracy']:.4f}"
        )

        print(
            f"With accuracy       : "
            f"{with_result['accuracy']:.4f}"
        )

        print(
            f"Improvement         : "
            f"{with_result['accuracy'] - without_result['accuracy']:+.4f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 80)
    print("v1.6.0 LANDMARK ROBUSTNESS ANALYSIS")
    print("=" * 80)

    samples, encoder = (
        build_test_samples()
    )

    print()
    print(
        f"Test samples : {len(samples)}"
    )

    print(
        "Classes      : "
        f"{len(encoder.classes)}"
    )

    print()
    print("Loading frozen v1.5.0 model:")
    print(MODEL_PATH)

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Clean baseline
    # --------------------------------------------------------
  
    clean_raw, clean_y, _, _= (
        create_degraded_dataset(
            samples=samples,
            encoder=encoder,
            gap_length=0,
            seed=RANDOM_SEED,
        )
    )

    clean_X = preprocess_dataset(
        clean_raw,
        use_interpolation=True,
    )

    clean = evaluate_dataset(
        model=model,
        X=clean_X,
        y=clean_y,
    )

    print()
    print("=" * 80)
    print("CLEAN BASELINE")
    print("=" * 80)

    print(
        f"Accuracy : "
        f"{clean['accuracy']:.4f}"
    )

    print(
        f"Correct  : "
        f"{clean['correct']}/{clean['total']}"
    )


    # --------------------------------------------------------
    # Multi-level robustness experiment
    # --------------------------------------------------------

    gap_lengths = [
        2,
        5,
        10,
        20,
    ]

    trial_seeds = [
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
    ]

    print()
    print("=" * 80)
    print("ROBUSTNESS CURVE")
    print("=" * 80)

    print()
    print(
        f"{'Gap':<10}"
        f"{'Without Interp.':<20}"
        f"{'With Interp.':<20}"
        f"{'Improvement':<15}"
    )

    print("-" * 65)

    for gap_length in gap_lengths:

        # --------------------------------------------------------
        # Create degradation ONCE
        # --------------------------------------------------------

        raw_degraded, y, _, _ = (
            create_degraded_dataset(
                samples=samples,
                encoder=encoder,
                gap_length=gap_length,
                seed=RANDOM_SEED,
            )
        )

        # --------------------------------------------------------
        # Condition 1: WITHOUT interpolation
        # --------------------------------------------------------

        X_without = preprocess_dataset(
            raw_degraded,
            use_interpolation=False,
        )

        no_interpolation = evaluate_dataset(
            model=model,
            X=X_without,
            y=y,
        )

        # --------------------------------------------------------
        # Condition 2: WITH interpolation
        # --------------------------------------------------------

        X_with = preprocess_dataset(
            raw_degraded,
            use_interpolation=True,
        )

        with_interpolation = evaluate_dataset(
            model=model,
            X=X_with,
            y=y,
        )

        # --------------------------------------------------------
        # Improvement
        # --------------------------------------------------------

        improvement = (
            with_interpolation["accuracy"]
            - no_interpolation["accuracy"]
        )

        print(
            f"{gap_length:<10}"
            f"{no_interpolation['accuracy']:.4f}"
            f" ({no_interpolation['correct']}/"
            f"{no_interpolation['total']})"
            f"{'':<5}"
            f"{with_interpolation['accuracy']:.4f}"
            f" ({with_interpolation['correct']}/"
            f"{with_interpolation['total']})"
            f"{'':<5}"
            f"{improvement:+.4f}"
        )

    # --------------------------------------------------------
    # Multi-trial robustness experiment
    # --------------------------------------------------------

    multi_trial_results = (
        run_multi_trial_experiment(
            model=model,
            samples=samples,
            encoder=encoder,
            gap_lengths=gap_lengths,
            trial_seeds=trial_seeds,
        )
    )
 

    summarize_multi_trial_results(
        results=multi_trial_results,
        gap_lengths=gap_lengths,
    )

    # --------------------------------------------------------
    # Step 6 - Per-sample analysis
    # --------------------------------------------------------

    run_per_sample_analysis(
        model=model,
        samples=samples,
        encoder=encoder,
        gap_lengths=gap_lengths,
        seed=42,
    )   
    
    print()
    print("=" * 80)
    print("STEP 6 COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()