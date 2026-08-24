from pathlib import Path
import numpy as np


DATASET_DIR = Path("dataset/processed/landmarks")


POSE_FEATURES = 33 * 4
HAND_FEATURES = 21 * 3

LEFT_HAND_START = POSE_FEATURES
RIGHT_HAND_START = POSE_FEATURES + HAND_FEATURES


def calculate_missing_segments(missing_frames):
    """
    Analyze consecutive missing-frame segments.

    Example:
        [False, True, True, False, True]

    gives:
        segments = 2
        maximum_gap = 2
        average_gap = 1.5
    """

    missing_frames = np.asarray(
        missing_frames,
        dtype=bool,
    )

    segment_lengths = []

    current_length = 0

    for missing in missing_frames:

        if missing:

            current_length += 1

        else:

            if current_length > 0:

                segment_lengths.append(
                    current_length
                )

                current_length = 0

    # Handle a missing segment that continues
    # until the final frame.

    if current_length > 0:

        segment_lengths.append(
            current_length
        )

    if not segment_lengths:

        return {
            "segments": 0,
            "maximum_gap": 0,
            "average_gap": 0.0,
        }

    return {
        "segments": len(segment_lengths),

        "maximum_gap":
            int(max(segment_lengths)),

        "average_gap":
            float(np.mean(segment_lengths)),
    }


def calculate_repairable_frames(
    missing_frames,
    max_gap=5,
):
    """
    Calculate how many missing frames belong to
    gaps that can be repaired by interpolation.

    Only gaps with length <= max_gap are counted.
    Gaps at the beginning or end are not repairable
    because interpolation requires valid frames on
    both sides.
    """

    missing_frames = np.asarray(
        missing_frames,
        dtype=bool,
    )

    frame_count = len(missing_frames)

    repairable_frames = 0
    repairable_segments = 0

    index = 0

    while index < frame_count:

        if not missing_frames[index]:

            index += 1
            continue

        start = index

        while (
            index < frame_count
            and missing_frames[index]
        ):
            index += 1

        end = index - 1

        gap_length = (
            end - start + 1
        )

        previous_index = start - 1
        next_index = end + 1

        # Gap must be short enough.

        if gap_length > max_gap:
            continue

        # A valid frame must exist before
        # and after the gap.

        if previous_index < 0:
            continue

        if next_index >= frame_count:
            continue

        if (
            missing_frames[previous_index]
            or missing_frames[next_index]
        ):
            continue

        repairable_segments += 1
        repairable_frames += gap_length

    return {
        "repairable_segments":
            repairable_segments,

        "repairable_frames":
            repairable_frames,
    }

def analyze_file(path):

    data = np.load(path)

    if data.ndim != 2:

        return None

    frame_count = data.shape[0]

    # -------------------------------------------------
    # Pose
    # -------------------------------------------------

    pose = data[:, :POSE_FEATURES]

    pose_xyz = pose.reshape(
        frame_count,
        33,
        4,
    )[:, :, :3]

    pose_missing = np.all(
        pose_xyz == 0,
        axis=2,
    ).all(axis=1)

    # -------------------------------------------------
    # Left hand
    # -------------------------------------------------

    left_hand = data[
        :,
        LEFT_HAND_START:
        RIGHT_HAND_START,
    ]

    left_hand = left_hand.reshape(
        frame_count,
        21,
        3,
    )

    left_missing = np.all(
        left_hand == 0,
        axis=(1, 2),
    )

    left_repair = calculate_repairable_frames(
        left_missing,
        max_gap=5,
    )

    left_gap_stats = calculate_missing_segments(
        left_missing
    )

    # -------------------------------------------------
    # Right hand
    # -------------------------------------------------

    right_hand = data[
        :,
        RIGHT_HAND_START:,
    ]

    right_hand = right_hand.reshape(
        frame_count,
        21,
        3,
    )

    right_missing = np.all(
        right_hand == 0,
        axis=(1, 2),
    )

    right_repair = calculate_repairable_frames(
        right_missing,
        max_gap=5,
    )

    right_gap_stats = calculate_missing_segments(
        right_missing
    )

    # -------------------------------------------------
    # Zero frames
    # -------------------------------------------------

    zero_frames = np.all(
        data == 0,
        axis=1,
    )

    return {

        "frames":
            frame_count,

        "pose_missing_frames":
            int(pose_missing.sum()),

        "left_hand_missing_frames":
            int(left_missing.sum()),

        "left_hand_missing_segments":
            left_gap_stats["segments"],

        "left_hand_max_gap":
            left_gap_stats["maximum_gap"],

        "left_hand_average_gap":
            left_gap_stats["average_gap"],

        "left_hand_repairable_segments":
            left_repair["repairable_segments"],

        "left_hand_repairable_frames":
            left_repair["repairable_frames"],

        "right_hand_missing_frames":
            int(right_missing.sum()),

        "right_hand_missing_segments":
            right_gap_stats["segments"],

        "right_hand_max_gap":
            right_gap_stats["maximum_gap"],

        "right_hand_average_gap":
            right_gap_stats["average_gap"],

        "right_hand_repairable_segments":
            right_repair["repairable_segments"],

        "right_hand_repairable_frames":
            right_repair["repairable_frames"],

        "zero_frames":
            int(zero_frames.sum()),

        "data_min":
            float(np.min(data)),

        "data_max":
            float(np.max(data)),

        "data_mean":
            float(np.mean(data)),

        "data_std":
            float(np.std(data)),
    }


def analyze_class(class_dir):

    results = []

    for path in sorted(
        class_dir.glob("*.npy")
    ):

        result = analyze_file(path)

        if result is not None:

            results.append(result)

    if not results:

        return None

    frames = np.array(
        [
            r["frames"]
            for r in results
        ]
    )

    short_sequences = np.sum(
        frames < 60
    )

    exact_sequences = np.sum(
        frames == 60
    )

    long_sequences = np.sum(
        frames > 60
    )

    pose_missing = np.array(
        [
            r["pose_missing_frames"]
            for r in results
        ]
    )

    left_missing = np.array(
        [
            r["left_hand_missing_frames"]
            for r in results
        ]
    )

    right_missing = np.array(
        [
            r["right_hand_missing_frames"]
            for r in results
        ]
    )

    left_segments = np.array(
        [
            r["left_hand_missing_segments"]
            for r in results
        ]
    )

    right_segments = np.array(
        [
            r["right_hand_missing_segments"]
            for r in results
        ]
    )

    left_max_gaps = np.array(
        [
            r["left_hand_max_gap"]
            for r in results
        ]
    )

    right_max_gaps = np.array(
        [
            r["right_hand_max_gap"]
            for r in results
        ]
    )

    left_repairable_segments = np.array(
        [
            r["left_hand_repairable_segments"]
            for r in results
        ]
    )

    left_repairable_frames = np.array(
        [
            r["left_hand_repairable_frames"]
            for r in results
        ]
    )

    right_repairable_segments = np.array(
        [
            r["right_hand_repairable_segments"]
            for r in results
        ]
    )

    right_repairable_frames = np.array(
        [
            r["right_hand_repairable_frames"]
            for r in results
        ]
    )

    zero_frames = np.array(
        [
            r["zero_frames"]
            for r in results
        ]
    )

    return {

        "samples":
            len(results),

        "frames_min":
            int(np.min(frames)),

        "frames_max":
            int(np.max(frames)),

        "frames_mean":
            float(np.mean(frames)),

        "frames_median":
            float(np.median(frames)),

        "frames_below_60":
            int(short_sequences),

        "frames_equal_60":
            int(exact_sequences),

        "frames_above_60":
            int(long_sequences),

        "pose_missing_total":
            int(np.sum(pose_missing)),

        # ---------------- Left hand ----------------

        "left_hand_missing_total":
            int(np.sum(left_missing)),

        "left_hand_missing_segments":
            int(np.sum(left_segments)),

        "left_hand_max_gap":
            int(np.max(left_max_gaps)),

        "left_hand_average_gap":
            float(
                np.mean(
                    [
                        r["left_hand_average_gap"]
                        for r in results
                        if r["left_hand_missing_segments"] > 0
                    ]
                )
            )
            if np.any(left_segments > 0)
            else 0.0,

        "left_hand_repairable_segments":
            int(np.sum(left_repairable_segments)),

        "left_hand_repairable_frames":
            int(np.sum(left_repairable_frames)),

        # ---------------- Right hand ----------------

        "right_hand_missing_total":
            int(np.sum(right_missing)),

        "right_hand_missing_segments":
            int(np.sum(right_segments)),

        "right_hand_max_gap":
            int(np.max(right_max_gaps)),

        "right_hand_average_gap":
            float(
                np.mean(
                    [
                        r["right_hand_average_gap"]
                        for r in results
                        if r["right_hand_missing_segments"] > 0
                    ]
                )
            )
            if np.any(right_segments > 0)
            else 0.0,

        "right_hand_repairable_segments":
            int(np.sum(right_repairable_segments)),

        "right_hand_repairable_frames":
            int(np.sum(right_repairable_frames)),

        "zero_frames_total":
            int(np.sum(zero_frames)),
    }


def main():

    if not DATASET_DIR.exists():

        print(
            f"Dataset directory not found: "
            f"{DATASET_DIR}"
        )

        return

    print()
    print("=" * 75)
    print("v1.5.0 LANDMARK QUALITY ANALYSIS")
    print("=" * 75)

    class_dirs = sorted(
        [
            path
            for path in DATASET_DIR.iterdir()
            if path.is_dir()
        ]
    )

    for class_dir in class_dirs:

        result = analyze_class(
            class_dir
        )

        if result is None:

            continue

        print()
        print(
            f"CLASS: {class_dir.name}"
        )

        print("-" * 75)

        print(
            f"Samples                  : "
            f"{result['samples']}"
        )

        print(
            f"Frames Min               : "
            f"{result['frames_min']}"
        )

        print(
            f"Frames Max               : "
            f"{result['frames_max']}"
        )

        print(
            f"Frames Mean              : "
            f"{result['frames_mean']:.2f}"
        )

        print(
            f"Frames Median            : "
            f"{result['frames_median']:.2f}"
        )

        print(
            f"Frames < 60              : "
            f"{result['frames_below_60']}"
        )

        print(
            f"Frames = 60              : "
            f"{result['frames_equal_60']}"
        )

        print(
            f"Frames > 60              : "
            f"{result['frames_above_60']}"
        )

        print(
            f"Pose Missing Frames      : "
            f"{result['pose_missing_total']}"
        )

        print()
        print("LEFT HAND")
        print("-" * 40)

        print(
            f"Missing Frames           : "
            f"{result['left_hand_missing_total']}"
        )

        print(
            f"Missing Segments         : "
            f"{result['left_hand_missing_segments']}"
        )

        print(
            f"Maximum Gap              : "
            f"{result['left_hand_max_gap']}"
        )

        print(
            f"Average Gap              : "
            f"{result['left_hand_average_gap']:.2f}"
        )

        print(
            f"Repairable Segments <=5 : "
            f"{result['left_hand_repairable_segments']}"
        )

        print(
            f"Repairable Frames <=5   : "
            f"{result['left_hand_repairable_frames']}"
        )

        print()
        print("RIGHT HAND")
        print("-" * 40)

        print(
            f"Missing Frames           : "
            f"{result['right_hand_missing_total']}"
        )

        print(
            f"Missing Segments         : "
            f"{result['right_hand_missing_segments']}"
        )

        print(
            f"Maximum Gap              : "
            f"{result['right_hand_max_gap']}"
        )

        print(
            f"Average Gap              : "
            f"{result['right_hand_average_gap']:.2f}"
        )

        print(
            f"Repairable Segments <=5 : "
            f"{result['right_hand_repairable_segments']}"
        )

        print(
            f"Repairable Frames <=5   : "
            f"{result['right_hand_repairable_frames']}"
        )

        print()
        print(
            f"Zero Frames              : "
            f"{result['zero_frames_total']}"
        )

    print()
    print("=" * 75)
    print("ANALYSIS COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()