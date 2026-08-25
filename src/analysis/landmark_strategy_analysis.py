from pathlib import Path
import pandas as pd

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


def _adaptive_interpolate_hand(
    hand,
    short_gap=5,
    adaptive_gap=10,
    cosine_threshold=0.85,
    magnitude_ratio_min=0.5,
    magnitude_ratio_max=2.0,
):
    """
    v1.7 adaptive hand-landmark recovery.

    Strategy
    --------
    1. Gaps up to short_gap are repaired using
       standard linear interpolation.

    2. Gaps between short_gap + 1 and adaptive_gap
       are repaired only when the local motion before
       and after the gap is sufficiently consistent.

    3. Gaps longer than adaptive_gap are left unchanged.

    Motion consistency is estimated using:
        - cosine similarity of the velocity vectors
        - ratio of velocity magnitudes
    """

    hand = hand.copy()

    missing = np.all(
        hand == 0.0,
        axis=1,
    )

    frame_count = len(hand)

    index = 0

    while index < frame_count:

        if not missing[index]:
            index += 1
            continue

        start = index

        while (
            index < frame_count
            and missing[index]
        ):
            index += 1

        end = index - 1

        gap_length = (
            end - start + 1
        )

        previous_index = start - 1
        next_index = end + 1

        # ----------------------------------------------------
        # Need valid frames immediately around the gap.
        # ----------------------------------------------------

        if previous_index < 0:
            continue

        if next_index >= frame_count:
            continue

        if missing[previous_index]:
            continue

        if missing[next_index]:
            continue

        # ----------------------------------------------------
        # Long gaps are not repaired.
        # ----------------------------------------------------

        if gap_length > adaptive_gap:
            continue

        previous = hand[
            previous_index
        ]

        next_frame = hand[
            next_index
        ]

        # ----------------------------------------------------
        # Short gaps use the proven v1.6 linear strategy.
        # ----------------------------------------------------

        if gap_length <= short_gap:

            for frame_index in range(
                start,
                end + 1,
            ):

                alpha = (
                    frame_index
                    - previous_index
                ) / (
                    next_index
                    - previous_index
                )

                hand[frame_index] = (
                    previous
                    + alpha
                    * (
                        next_frame
                        - previous
                    )
                )

            continue

        # ----------------------------------------------------
        # v1.7 adaptive decision for longer gaps.
        # ----------------------------------------------------

        # Velocity immediately before the gap.
        velocity_before = (
            hand[previous_index]
            - hand[previous_index - 1]
            if previous_index >= 1
            else None
        )

        # Velocity immediately after the gap.
        velocity_after = (
            hand[next_index + 1]
            - hand[next_index]
            if next_index + 1 < frame_count
            else None
        )

        # Cannot estimate motion reliably.
        if (
            velocity_before is None
            or velocity_after is None
        ):
            continue

        norm_before = np.linalg.norm(
            velocity_before
        )

        norm_after = np.linalg.norm(
            velocity_after
        )

        # Both velocities essentially stationary.
        if (
            norm_before < 1e-8
            and norm_after < 1e-8
        ):

            should_interpolate = True

        elif (
            norm_before < 1e-8
            or norm_after < 1e-8
        ):

            should_interpolate = False

        else:

            cosine_similarity = (
                np.dot(
                    velocity_before,
                    velocity_after,
                )
                / (
                    norm_before
                    * norm_after
                )
            )

            cosine_similarity = float(
                np.clip(
                    cosine_similarity,
                    -1.0,
                    1.0,
                )
            )
            
            magnitude_ratio = (
                norm_after
                / norm_before
            )

            should_interpolate = (
                cosine_similarity
                >= cosine_threshold
                and magnitude_ratio
                >= magnitude_ratio_min
                and magnitude_ratio
                <= magnitude_ratio_max
            )

            '''print(
                f"[v1.7 diagnostic] "
                f"gap={gap_length} "
                f"cosine={cosine_similarity:.4f} "
                f"ratio={magnitude_ratio:.4f}"
                f"accepted={should_interpolate}"
            )'''

        if not should_interpolate:
            continue

        # ----------------------------------------------------
        # Motion-consistent gap:
        # use linear interpolation.
        # ----------------------------------------------------

        for frame_index in range(
            start,
            end + 1,
        ):

            alpha = (
                frame_index
                - previous_index
            ) / (
                next_index
                - previous_index
            )

            hand[frame_index] = (
                previous
                + alpha
                * (
                    next_frame
                    - previous
                )
            )

    return hand


def _windowed_adaptive_interpolate_hand(
    hand,
    short_gap=5,
    adaptive_gap=10,
    motion_window=3,
    cosine_threshold=0.75,
    magnitude_ratio_min=0.5,
    magnitude_ratio_max=2.0,
    landmark_consistency_threshold=0.60,
):
    """
    v1.7.1 windowed landmark-aware hand recovery.

    Strategy
    --------
    1. Gaps up to short_gap use the proven linear
       interpolation strategy.

    2. Gaps between short_gap + 1 and adaptive_gap
       are evaluated using motion consistency.

    3. Motion is estimated over a window of frames
       before and after the gap instead of using
       only one frame-to-frame velocity.

    4. Motion consistency is evaluated independently
       for each of the 21 hand landmarks.

    5. A gap is repaired when a sufficient proportion
       of landmarks satisfy the motion-consistency
       criteria.

    6. Gaps longer than adaptive_gap are not repaired.
    """

    hand = hand.copy()

    missing = np.all(
        hand == 0.0,
        axis=1,
    )

    frame_count = len(hand)

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if hand.ndim != 2:
        raise ValueError(
            "Expected hand landmarks with shape "
            "(frames, features)"
        )

    if hand.shape[1] != 63:
        raise ValueError(
            "Expected 63 hand landmark features "
            "(21 landmarks × 3 coordinates)."
        )

    if motion_window < 1:
        raise ValueError(
            "motion_window must be >= 1"
        )

    index = 0

    while index < frame_count:

        if not missing[index]:
            index += 1
            continue

        start = index

        while (
            index < frame_count
            and missing[index]
        ):
            index += 1

        end = index - 1

        gap_length = (
            end - start + 1
        )

        previous_index = start - 1
        next_index = end + 1

        # ----------------------------------------------------
        # Need valid frames immediately around the gap.
        # ----------------------------------------------------

        if previous_index < 0:
            continue

        if next_index >= frame_count:
            continue

        if missing[previous_index]:
            continue

        if missing[next_index]:
            continue

        # ----------------------------------------------------
        # Long gaps are not repaired.
        # ----------------------------------------------------

        if gap_length > adaptive_gap:
            continue

        previous = hand[
            previous_index
        ]

        next_frame = hand[
            next_index
        ]

        # ----------------------------------------------------
        # Short gaps retain the proven linear strategy.
        # ----------------------------------------------------

        if gap_length <= short_gap:

            for frame_index in range(
                start,
                end + 1,
            ):

                alpha = (
                    frame_index
                    - previous_index
                ) / (
                    next_index
                    - previous_index
                )

                hand[frame_index] = (
                    previous
                    + alpha
                    * (
                        next_frame
                        - previous
                    )
                )

            continue

        # ----------------------------------------------------
        # v1.7.1 windowed motion analysis
        # ----------------------------------------------------

        # We need enough valid frames on both sides
        # to calculate windowed velocities.
        #
        # Before:
        #   p-3 -> p-2 -> p-1 -> p
        #
        # After:
        #   n -> n+1 -> n+2 -> n+3
        #
        # where p = previous_index
        # and n = next_index.

        if (
            previous_index < motion_window
            or
            next_index + motion_window
            >= frame_count
        ):
            continue

        before_frames = hand[
            previous_index
            - motion_window:
            previous_index + 1
        ]

        after_frames = hand[
            next_index:
            next_index + motion_window + 1
        ]

        # ----------------------------------------------------
        # Calculate frame-to-frame velocities.
        # ----------------------------------------------------

        before_velocity = (
            np.diff(
                before_frames,
                axis=0,
            )
        )

        after_velocity = (
            np.diff(
                after_frames,
                axis=0,
            )
        )

        # ----------------------------------------------------
        # Reshape:
        #
        # (motion_window, 63)
        #
        # -> (motion_window, 21, 3)
        # ----------------------------------------------------

        before_velocity = (
            before_velocity.reshape(
                motion_window,
                21,
                3,
            )
        )

        after_velocity = (
            after_velocity.reshape(
                motion_window,
                21,
                3,
            )
        )

        # ----------------------------------------------------
        # Average velocity for each landmark.
        # ----------------------------------------------------

        before_mean = np.mean(
            before_velocity,
            axis=0,
        )

        after_mean = np.mean(
            after_velocity,
            axis=0,
        )

        # ----------------------------------------------------
        # Landmark-wise motion comparison.
        # ----------------------------------------------------

        consistent_landmarks = 0

        for landmark_index in range(21):

            velocity_before = (
                before_mean[
                    landmark_index
                ]
            )

            velocity_after = (
                after_mean[
                    landmark_index
                ]
            )

            norm_before = np.linalg.norm(
                velocity_before
            )

            norm_after = np.linalg.norm(
                velocity_after
            )

            # Both essentially stationary.
            if (
                norm_before < 1e-8
                and norm_after < 1e-8
            ):
                consistent_landmarks += 1
                continue

            # One stationary and one moving:
            # insufficient evidence.
            if (
                norm_before < 1e-8
                or norm_after < 1e-8
            ):
                continue

            cosine_similarity = (
                np.dot(
                    velocity_before,
                    velocity_after,
                )
                / (
                    norm_before
                    * norm_after
                )
            )

            cosine_similarity = float(
                np.clip(
                    cosine_similarity,
                    -1.0,
                    1.0,
                )
            )

            magnitude_ratio = (
                norm_after
                / norm_before
            )

            if (
                cosine_similarity
                >= cosine_threshold
                and
                magnitude_ratio
                >= magnitude_ratio_min
                and
                magnitude_ratio
                <= magnitude_ratio_max
            ):
                consistent_landmarks += 1

        # ----------------------------------------------------
        # Overall landmark consistency.
        # ----------------------------------------------------

        consistency_ratio = (
            consistent_landmarks / 21.0
        )

        should_interpolate = (
            consistency_ratio
            >= landmark_consistency_threshold
        )

        '''print(
            f"[v1.7.1 diagnostic] "
            f"gap={gap_length} "
            f"consistent="
            f"{consistent_landmarks}/21 "
            f"ratio="
            f"{consistency_ratio:.4f} "
            f"accepted="
            f"{should_interpolate}"
        )'''

        if not should_interpolate:
            continue

        # ----------------------------------------------------
        # Motion-consistent gap:
        # use linear interpolation.
        # ----------------------------------------------------

        for frame_index in range(
            start,
            end + 1,
        ):

            alpha = (
                frame_index
                - previous_index
            ) / (
                next_index
                - previous_index
            )

            hand[frame_index] = (
                previous
                + alpha
                * (
                    next_frame
                    - previous
                )
            )

    return hand


def adaptive_landmark_interpolation(
    landmarks,
):
    """
    Apply the v1.7 adaptive recovery strategy
    independently to the left and right hands.
    """

    landmarks = np.asarray(
        landmarks,
        dtype=np.float32,
    ).copy()

    expected_features = (
        33 * 4
        + 2 * 21 * 3
    )

    if landmarks.ndim != 2:
        raise ValueError(
            "Expected landmarks with shape "
            "(frames, features)"
        )

    if landmarks.shape[1] != expected_features:
        raise ValueError(
            f"Expected {expected_features} "
            f"features, found "
            f"{landmarks.shape[1]}"
        )

    pose_features = 33 * 4
    hand_landmarks = 21
    features_per_hand = 3
    hand_features = (
        hand_landmarks
        * features_per_hand
    )

    # --------------------------------------------------------
    # Left hand
    # --------------------------------------------------------

    left_start = pose_features
    left_end = (
        left_start
        + hand_features
    )

    left_hand = landmarks[
        :,
        left_start:left_end,
    ]

    left_hand = _adaptive_interpolate_hand(
        left_hand
    )

    landmarks[
        :,
        left_start:left_end,
    ] = left_hand

    # --------------------------------------------------------
    # Right hand
    # --------------------------------------------------------

    right_start = (
        pose_features
        + hand_features
    )

    right_end = (
        right_start
        + hand_features
    )

    right_hand = landmarks[
        :,
        right_start:right_end,
    ]

    right_hand = _adaptive_interpolate_hand(
        right_hand
    )

    landmarks[
        :,
        right_start:right_end,
    ] = right_hand

    return landmarks


def apply_recovery_strategy(
    landmarks,
    strategy,
):
    """
    Apply a landmark recovery strategy.

    Strategies
    ----------
    none:
        No landmark recovery.

    linear:
        Existing v1.5/v1.6 linear interpolation
        with maximum gap of 5 frames.

    v1.7:
        Adaptive landmark recovery.

        Gaps up to 5 frames are repaired using
        linear interpolation.

        Gaps from 6 to 10 frames are repaired
        only when local motion before and after
        the gap is sufficiently consistent.

        Gaps longer than 10 frames are left
        unrepaired.
    """

    if strategy == "none":

        return landmarks.copy()

    if strategy == "linear":

        interpolator = HandLandmarkInterpolator(
            max_gap=5
        )

        return interpolator.interpolate(
            landmarks
        )

    if strategy == "v1.7":

        return adaptive_landmark_interpolation(
            landmarks
        )

    if strategy == "v1.7.1":

        recovered = landmarks.copy()

        # --------------------------------------------------------
        # Left hand
        # --------------------------------------------------------

        left_start = (
            HandLandmarkInterpolator.POSE_FEATURES
        )

        left_end = (
            left_start
            + HandLandmarkInterpolator.HAND_FEATURES
        )

        left_hand = recovered[
            :,
            left_start:left_end,
        ]

        left_hand = (
            left_hand
            .reshape(
                -1,
                63,
            )
        )

        left_hand = (
            _windowed_adaptive_interpolate_hand(
                left_hand
            )
        )

        recovered[
            :,
            left_start:left_end,
        ] = left_hand

        # --------------------------------------------------------
        # Right hand
        # --------------------------------------------------------

        right_start = (
            HandLandmarkInterpolator.RIGHT_HAND_START
        )

        right_end = (
            right_start
            + HandLandmarkInterpolator.HAND_FEATURES
        )

        right_hand = recovered[
            :,
            right_start:right_end,
        ]

        right_hand = (
            right_hand
            .reshape(
                -1,
                63,
            )
        )

        right_hand = (
            _windowed_adaptive_interpolate_hand(
                right_hand
            )
        )

        recovered[
            :,
            right_start:right_end,
        ] = right_hand

        return recovered

    raise ValueError(
        f"Unknown recovery strategy: {strategy}"
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

        data = apply_recovery_strategy(
            data,
            strategy="linear",
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
    Run paired robustness experiments across four recovery strategies.

    Strategies:
        none
        linear
        v1.7
        v1.7.1

    The same degraded landmarks are used for every strategy
    within each trial.
    """

    strategies = [
        "none",
        "linear",
        "v1.7",
        "v1.7.1",
    ]

    results = []

    for gap_length in gap_lengths:

        print()
        print("=" * 80)
        print(f"GAP LENGTH: {gap_length}")
        print("=" * 80)

        for trial_number, seed in enumerate(
            trial_seeds,
            start=1,
        ):

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

            strategy_results = {}

            for strategy in strategies:

                X_strategy = []

                repaired_landmarks_list = []

                for landmarks in raw_degraded:

                    recovered = apply_recovery_strategy(
                        landmarks,
                        strategy=strategy,
                    )

                    repaired_landmarks_list.append(
                        recovered
                    )

                    sequence = preprocess(
                        recovered,
                        use_interpolation=False,
                    )

                    X_strategy.append(sequence)

                X_strategy = np.asarray(
                    X_strategy,
                    dtype=np.float32,
                )

                evaluation = evaluate_dataset(
                    model=model,
                    X=X_strategy,
                    y=y,
                )

                strategy_results[strategy] = {
                    "evaluation": evaluation,
                    "landmarks": repaired_landmarks_list,
                }

            # ------------------------------------------------
            # Evaluation results
            # ------------------------------------------------

            none_result = strategy_results[
                "none"
            ]["evaluation"]

            linear_result = strategy_results[
                "linear"
            ]["evaluation"]

            v17_result = strategy_results[
                "v1.7"
            ]["evaluation"]

            v171_result = strategy_results[
                "v1.7.1"
            ]["evaluation"]

            # ------------------------------------------------
            # Repair statistics
            # ------------------------------------------------

            linear_repaired = 0
            v17_repaired = 0
            v171_repaired = 0

            for sample_index in range(
                len(raw_degraded)
            ):

                original = original_landmarks[
                    sample_index
                ]

                linear_recovered = (
                    strategy_results["linear"][
                        "landmarks"
                    ][sample_index]
                )

                v17_recovered = (
                    strategy_results["v1.7"][
                        "landmarks"
                    ][sample_index]
                )

                v171_recovered = (
                    strategy_results["v1.7.1"][
                        "landmarks"
                    ][sample_index]
                )

                left_mask = degradation_masks[
                    sample_index
                ][0]

                right_mask = degradation_masks[
                    sample_index
                ][1]

                linear_repair_stats = count_artificial_repair(
                    original,
                    linear_recovered,
                    left_mask,
                    right_mask,
                )

                v17_repair_stats = count_artificial_repair(
                    original,
                    v17_recovered,
                    left_mask,
                    right_mask,
                )

                v171_repair_stats = count_artificial_repair(
                    original,
                    v171_recovered,
                    left_mask,
                    right_mask,
                )

                linear_repaired += (
                    linear_repair_stats["artificial_repaired"]
                )

                v17_repaired += (
                    v17_repair_stats["artificial_repaired"]
                )

                v171_repaired += (
                    v171_repair_stats["artificial_repaired"]
                )
                

            # ------------------------------------------------
            # Store results
            # ------------------------------------------------

            result = {
                "gap_length": gap_length,
                "trial": trial_number,
                "seed": seed,

                "none_accuracy":
                    none_result["accuracy"],

                "linear_accuracy":
                    linear_result["accuracy"],

                "v17_accuracy":
                    v17_result["accuracy"],

                "v171_accuracy":
                    v171_result["accuracy"],

                "linear_improvement":
                    linear_result["accuracy"]
                    - none_result["accuracy"],

                "v17_improvement":
                    v17_result["accuracy"]
                    - none_result["accuracy"],

                "v171_improvement":
                    v171_result["accuracy"]
                    - none_result["accuracy"],

                "v171_vs_linear":
                    v171_result["accuracy"]
                    - linear_result["accuracy"],

                "none_correct":
                    none_result["correct"],

                "linear_correct":
                    linear_result["correct"],

                "v17_correct":
                    v17_result["correct"],

                "v171_correct":
                    v171_result["correct"],

                "total":
                    none_result["total"],

                "linear_repaired":
                    linear_repaired,

                "v17_repaired":
                    v17_repaired,

                "v171_repaired":
                    v171_repaired,
            }

            results.append(result)

            print(
                f"Trial {trial_number:2d} "
                f"(seed={seed})  "
                f"None="
                f"{none_result['accuracy']:.4f} "
                f"({none_result['correct']}/"
                f"{none_result['total']})  "
                f"Linear="
                f"{linear_result['accuracy']:.4f} "
                f"({linear_result['correct']}/"
                f"{linear_result['total']})  "
                f"v1.7="
                f"{v17_result['accuracy']:.4f} "
                f"({v17_result['correct']}/"
                f"{v17_result['total']})  "
                f"v1.7.1="
                f"{v171_result['accuracy']:.4f} "
                f"({v171_result['correct']}/"
                f"{v171_result['total']})  "
                f"Linear Δ="
                f"{result['linear_improvement']:+.4f}  "
                f"v1.7 Δ="
                f"{result['v17_improvement']:+.4f}  "
                f"v1.7.1 Δ="
                f"{result['v171_improvement']:+.4f}  "
                f"v1.7.1 vs Linear="
                f"{result['v171_vs_linear']:+.4f}  "
                f"Linear Repaired="
                f"{linear_repaired}  "
                f"v1.7 Repaired="
                f"{v17_repaired}  "
                f"v1.7.1 Repaired="
                f"{v171_repaired}"
            )

    return results


def summarize_multi_trial_results(results):
    """
    Summarize multi-trial results for the three
    v1.7 recovery strategies.

    Strategies:
        none
        linear
        v1.7
    """

    gap_lengths = sorted(
        set(
            result["gap_length"]
            for result in results
        )
    )

    print()
    print("=" * 80)
    print("MULTI-TRIAL STRATEGY SUMMARY")
    print("=" * 80)

    print()
    print(
        f"{'Gap':<5}"
        f"{'None Mean':<15}"
        f"{'None Std':<15}"
        f"{'Linear Mean':<15}"
        f"{'Linear Std':<15}"
        f"{'v1.7 Mean':<15}"
        f"{'v1.7 Std':<15}"
        f"{'v1.7.1 Mean':<17}"
        f"{'v1.7.1 Std':<17}"
        f"{'Linear Repair':<16}"
        f"{'v1.7 Repair':<16}"
        f"{'v1.7.1 Repair':<18}"
    )

    print("-" * 125)

    for gap_length in gap_lengths:

        gap_results = [
            result
            for result in results
            if result["gap_length"]
            == gap_length
        ]

        none_values = np.asarray(
            [
                result["none_accuracy"]
                for result in gap_results
            ],
            dtype=np.float32,
        )

        linear_values = np.asarray(
            [
                result["linear_accuracy"]
                for result in gap_results
            ],
            dtype=np.float32,
        )

        v17_values = np.asarray(
            [
                result["v17_accuracy"]
                for result in gap_results
            ],
            dtype=np.float32,
        )

        v171_values = np.asarray(
            [
                result["v171_accuracy"]
                for result in gap_results
            ],
            dtype=np.float32,
        )

        none_mean = np.mean(
            none_values
        )

        none_std = np.std(
            none_values
        )

        linear_mean = np.mean(
            linear_values
        )

        linear_std = np.std(
            linear_values
        )

        v17_mean = np.mean(
            v17_values
        )

        v17_std = np.std(
            v17_values
        )

        v171_mean = np.mean(
            v171_values
        )

        v171_std = np.std(
            v171_values
        )

        linear_repaired_values = np.asarray(
            [
                result["linear_repaired"]
                for result in gap_results
            ],
            dtype=np.float32,
        )

        v17_repaired_values = np.asarray(
            [
                result["v17_repaired"]
                for result in gap_results
            ],
            dtype=np.float32,
        )

        v171_repaired_values = np.asarray(
            [
                result["v171_repaired"]
                for result in gap_results
            ],
            dtype=np.float32,
        )

        print(
            f"{gap_length:<5}"
            f"{np.mean(none_values):<15.4f}"
            f"{np.std(none_values):<15.4f}"
            f"{np.mean(linear_values):<15.4f}"
            f"{np.std(linear_values):<15.4f}"
            f"{np.mean(v17_values):<15.4f}"
            f"{np.std(v17_values):<15.4f}"
            f"{np.mean(v171_values):<17.4f}"
            f"{np.std(v171_values):<17.4f}"
            f"{np.mean(linear_repaired_values):<16.1f}"
            f"{np.mean(v17_repaired_values):<16.1f}"
            f"{np.mean(v171_repaired_values):<18.1f}"
        )

    print()
    print(
        "Mean improvement relative to None:"
    )

    print()

    for gap_length in gap_lengths:

        gap_results = [
            result
            for result in results
            if result["gap_length"]
            == gap_length
        ]

        none_values = np.asarray(
            [
                result["none_accuracy"]
                for result in gap_results
            ],
            dtype=np.float32,
        )

        linear_values = np.asarray(
            [
                result["linear_accuracy"]
                for result in gap_results
            ],
            dtype=np.float32,
        )

        v17_values = np.asarray(
            [
                result["v17_accuracy"]
                for result in gap_results
            ],
            dtype=np.float32,
        )

        v171_values = np.asarray(
            [
                result["v171_accuracy"]
                for result in gap_results
            ],
            dtype=np.float32,
        )

        linear_improvement = (
            np.mean(linear_values)
            - np.mean(none_values)
        )

        v17_improvement = (
            np.mean(v17_values)
            - np.mean(none_values)
        )

        v17_vs_linear = (
            np.mean(v17_values)
            - np.mean(linear_values)
        )

        v171_improvement = (
            np.mean(v171_values)
            - np.mean(none_values)
        )

        v171_vs_linear = (
            np.mean(v171_values)
            - np.mean(linear_values)
        )

        print(
            f"Gap {gap_length:2d}: "
            f"Linear={linear_improvement:+.4f}  "
            f"v1.7={v17_improvement:+.4f}  "
            f"v1.7 vs Linear={v17_vs_linear:+.4f}  "
            f"v1.7.1={v171_improvement:+.4f}  "
            f"v1.7.1 vs Linear={v171_vs_linear:+.4f}  "
        )

    print()


def export_strategy_summary(results):
    """
    Export multi-trial strategy results to CSV.

    The exported file contains one row per
    gap length with mean and standard deviation
    for each recovery strategy.
    """

    output_dir = Path(
        "docs/experiments/v1.7.0-landmark-strategy"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gap_lengths = sorted(
        set(
            result["gap_length"]
            for result in results
        )
    )

    rows = []

    for gap_length in gap_lengths:

        gap_results = [
            result
            for result in results
            if result["gap_length"] == gap_length
        ]

        none_values = np.asarray(
            [
                result["none_accuracy"]
                for result in gap_results
            ],
            dtype=np.float32,
        )

        linear_values = np.asarray(
            [
                result["linear_accuracy"]
                for result in gap_results
            ],
            dtype=np.float32,
        )

        v17_values = np.asarray(
            [
                result["v17_accuracy"]
                for result in gap_results
            ],
            dtype=np.float32,
        )

        v171_values = np.asarray(
            [
                result["v171_accuracy"]
                for result in gap_results
            ],
            dtype=np.float32,
        )

        rows.append(
            {
                "gap_length": gap_length,

                "none_mean": np.mean(
                    none_values
                ),
                "none_std": np.std(
                    none_values
                ),

                "linear_mean": np.mean(
                    linear_values
                ),
                "linear_std": np.std(
                    linear_values
                ),

                "v17_mean": np.mean(
                    v17_values
                ),
                "v17_std": np.std(
                    v17_values
                ),

                "v171_mean": np.mean(
                    v171_values
                ),
                "v171_std": np.std(
                    v171_values
                ),

                "linear_improvement":
                    np.mean(linear_values)
                    - np.mean(none_values),

                "v17_improvement":
                    np.mean(v17_values)
                    - np.mean(none_values),

                "v171_improvement":
                    np.mean(v171_values)
                    - np.mean(none_values),

                "v17_vs_linear":
                    np.mean(v17_values)
                    - np.mean(linear_values),

                "v171_vs_linear":
                    np.mean(v171_values)
                    - np.mean(linear_values),

                "linear_repair_mean":
                    np.mean(
                        [
                            result["linear_repaired"]
                            for result in gap_results
                        ]
                    ),

                "v17_repair_mean":
                    np.mean(
                        [
                            result["v17_repaired"]
                            for result in gap_results
                        ]
                    ),

                "v171_repair_mean":
                    np.mean(
                        [
                            result["v171_repaired"]
                            for result in gap_results
                        ]
                    ),
            }
        )

    output_path = (
        output_dir / "strategy_summary.csv"
    )

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Saved strategy summary: "
        f"{output_path}"
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
    Compare individual predictions across all v1.7
    landmark recovery strategies.

    Strategies:
        none
        linear
        v1.7
        v1.7.1
    """

    output_dir = Path(
        "docs/experiments/v1.7.0-landmark-strategy"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    print()
    print("=" * 80)
    print("STEP 6 - PER-SAMPLE STRATEGY ANALYSIS")
    print("=" * 80)

    strategies = [
        "none",
        "linear",
        "v1.7",
        "v1.7.1",
    ]

    for gap_length in gap_lengths:

        print()
        print("=" * 80)
        print(
            f"GAP LENGTH: {gap_length} "
            f"(seed={seed})"
        )
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

        strategy_results = {}

        # ----------------------------------------------------
        # Evaluate every strategy on SAME degradation
        # ----------------------------------------------------

        for strategy in strategies:

            predictions = []
            probabilities = []

            for landmarks in raw_degraded:

                recovered = apply_recovery_strategy(
                    landmarks,
                    strategy=strategy,
                )

                sequence = preprocess(
                    recovered,
                    use_interpolation=False,
                )

                sequence = np.asarray(
                    sequence,
                    dtype=np.float32,
                )

                prediction_result = evaluate_dataset(
                    model=model,
                    X=np.asarray(
                        [sequence],
                        dtype=np.float32,
                    ),
                    y=np.asarray(
                        [y[len(predictions)]],
                        dtype=np.int32,
                    ),
                )

                predictions.append(
                    prediction_result["predictions"][0]
                )

                probabilities.append(
                    prediction_result["probabilities"][0]
                )

            predictions = np.asarray(
                predictions
            )

            probabilities = np.asarray(
                probabilities
            )

            correct = (
                predictions == y
            )

            strategy_results[strategy] = {
                "predictions": predictions,
                "probabilities": probabilities,
                "correct": correct,
            }

        # ----------------------------------------------------
        # Compare strategies sample-by-sample
        # ----------------------------------------------------

        changed_count = 0
        recovered_count = 0
        harmed_count = 0

        for index, sample in enumerate(samples):

            true_label = y[index]

            rows.append(
                {
                    "gap_length": gap_length,
                    "seed": seed,
                    "sample_index": index + 1,
                    "file": sample["path"],
                    "true_label": encoder.classes[true_label],

                    "none_prediction":
                        encoder.classes[
                            strategy_results["none"]
                            ["predictions"][index]
                        ],

                    "linear_prediction":
                        encoder.classes[
                            strategy_results["linear"]
                            ["predictions"][index]
                        ],

                    "v17_prediction":
                        encoder.classes[
                            strategy_results["v1.7"]
                            ["predictions"][index]
                        ],

                    "v171_prediction":
                        encoder.classes[
                            strategy_results["v1.7.1"]
                            ["predictions"][index]
                        ],

                    "none_correct":
                        bool(
                            strategy_results["none"]
                            ["correct"][index]
                        ),

                    "linear_correct":
                        bool(
                            strategy_results["linear"]
                            ["correct"][index]
                        ),

                    "v17_correct":
                        bool(
                            strategy_results["v1.7"]
                            ["correct"][index]
                        ),

                    "v171_correct":
                        bool(
                            strategy_results["v1.7.1"]
                            ["correct"][index]
                        ),
                }
            )

            none_prediction = (
                strategy_results["none"]
                ["predictions"][index]
            )

            linear_prediction = (
                strategy_results["linear"]
                ["predictions"][index]
            )

            v17_prediction = (
                strategy_results["v1.7"]
                ["predictions"][index]
            )

            v171_prediction = (
                strategy_results["v1.7.1"]
                ["predictions"][index]
            )

            predictions = [
                none_prediction,
                linear_prediction,
                v17_prediction,
                v171_prediction,
            ]

            # Only show samples where strategies differ.
            if len(set(predictions)) == 1:
                continue

            changed_count += 1

            #true_label = y[index]

            none_correct = (
                none_prediction == true_label
            )

            v171_correct = (
                v171_prediction == true_label
            )

            if not none_correct and v171_correct:
                recovered_count += 1

            elif none_correct and not v171_correct:
                harmed_count += 1

            true_name = encoder.classes[
                true_label
            ]

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

            for strategy in strategies:

                prediction = (
                    strategy_results[strategy]
                    ["predictions"][index]
                )

                probability = (
                    strategy_results[strategy]
                    ["probabilities"][index]
                )

                prediction_name = (
                    encoder.classes[prediction]
                )

                confidence = float(
                    probability[prediction]
                )

                status = (
                    "CORRECT"
                    if prediction == true_label
                    else "WRONG"
                )

                print(
                    f"{strategy:<10}: "
                    f"{prediction_name:<15} "
                    f"({confidence:.4f}) "
                    f"{status}"
                )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print()
        print("-" * 80)

        for strategy in strategies:

            accuracy = np.mean(
                strategy_results[strategy]["correct"]
            )

            correct_count = np.sum(
                strategy_results[strategy]["correct"]
            )

            print(
                f"{strategy:<10}: "
                f"{accuracy:.4f} "
                f"({correct_count}/{len(samples)})"
            )

        print(
            f"Samples with different predictions : "
            f"{changed_count}/{len(samples)}"
        )

        print(
            f"None -> v1.7.1 recovered             : "
            f"{recovered_count}"
        )

        print(
            f"None -> v1.7.1 harmed                : "
            f"{harmed_count}"
        )


    # --------------------------------------------------------
    # Export per-sample analysis
    # --------------------------------------------------------

    output_path = (
        output_dir / "per_sample_analysis.csv"
    )

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Saved per-sample analysis: "
        f"{output_path}"
    )       


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 80)
    print("v1.7.0 LANDMARK STRATEGY ANALYSIS")
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
    # v1.7 Multi-strategy robustness experiment
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
    print("v1.7 LANDMARK STRATEGY EXPERIMENT")
    print("=" * 80)


    # --------------------------------------------------------
    # Multi-trial robustness experiment
    # --------------------------------------------------------

    results = (
        run_multi_trial_experiment(
            model=model,
            samples=samples,
            encoder=encoder,
            gap_lengths=gap_lengths,
            trial_seeds=trial_seeds,
        )
    )
 

    summarize_multi_trial_results(
        results
    )

    export_strategy_summary(
        results
    )


    # --------------------------------------------------------
    # Per-sample analysis
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("PER-SAMPLE LANDMARK STRATEGY ANALYSIS")
    print("=" * 80)

    run_per_sample_analysis(
        model=model,
        samples=samples,
        encoder=encoder,
        gap_lengths=gap_lengths,
        seed=RANDOM_SEED,
    )


    

if __name__ == "__main__":
    main()