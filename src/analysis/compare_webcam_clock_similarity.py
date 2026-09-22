from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

WEBCAM_PATH = (
    ROOT
    / "outputs"
    / "webcam_test"
    / "clock_webcam_20260920_222015.npy"
)

DATASET_DIR = (
    ROOT
    / "dataset"
    / "processed"
    / "generalized_landmarks_legacy"
)

SEQ_LEN = 60


def prepare_sequence(sequence):

    sequence = np.asarray(sequence, dtype=np.float32)

    if len(sequence) > SEQ_LEN:

        indices = np.linspace(
            0,
            len(sequence) - 1,
            SEQ_LEN
        ).astype(int)

        sequence = sequence[indices]

    elif len(sequence) < SEQ_LEN:

        padding = np.zeros(
            (SEQ_LEN - len(sequence), sequence.shape[1]),
            dtype=np.float32
        )

        sequence = np.vstack([
            sequence,
            padding
        ])

    return sequence


def normalize(sequence):

    sequence = sequence.copy()

    # Pose landmarks occupy first 132 features.
    pose = sequence[:, :132].reshape(
        -1, 33, 4
    )

    # Left hand
    left_hand = sequence[:, 132:195].reshape(
        -1, 21, 3
    )

    # Right hand
    right_hand = sequence[:, 195:258].reshape(
        -1, 21, 3
    )

    # -------------------------------------------------
    # Calculate shoulder center
    # -------------------------------------------------

    left_shoulder = pose[:, 11, :3]
    right_shoulder = pose[:, 12, :3]

    center = (
        left_shoulder + right_shoulder
    ) / 2.0

    # Shoulder distance
    scale = np.linalg.norm(
        left_shoulder - right_shoulder,
        axis=1
    )

    scale = np.maximum(
        scale,
        1e-6
    )

    # Normalize pose
    pose[:, :, :3] -= center[:, None, :]
    pose[:, :, :3] /= scale[:, None, None]

    # Normalize hands using same transformation
    left_hand -= center[:, None, :]
    left_hand /= scale[:, None, None]

    right_hand -= center[:, None, :]
    right_hand /= scale[:, None, None]

    sequence[:, :132] = pose.reshape(
        -1, 132
    )

    sequence[:, 132:195] = left_hand.reshape(
        -1, 63
    )

    sequence[:, 195:258] = right_hand.reshape(
        -1, 63
    )

    return sequence


def sequence_distance(a, b):

    a = normalize(prepare_sequence(a))
    b = normalize(prepare_sequence(b))

    return np.mean(
        np.linalg.norm(
            a - b,
            axis=1
        )
    )


# -------------------------------------------------
# Load webcam
# -------------------------------------------------

webcam = np.load(
    WEBCAM_PATH
)

print("Webcam:")
print(webcam.shape)


# -------------------------------------------------
# Find Clock samples
# -------------------------------------------------

clock_files = list(
    (DATASET_DIR / "51. Clock").glob("*.npy")
)

print(
    f"\nClock samples found: {len(clock_files)}"
)


results = []

for path in clock_files:

    try:

        sample = np.load(path)

        distance = sequence_distance(
            webcam,
            sample
        )

        results.append(
            (
                distance,
                path.name,
                str(path)
            )
        )

    except Exception as e:

        print(
            f"Skipping {path}: {e}"
        )


# -------------------------------------------------
# Sort
# -------------------------------------------------

results.sort(
    key=lambda x: x[0]
)


print("\n")
print("=" * 75)
print("CLOSEST CLOCK LANDMARK SEQUENCES")
print("=" * 75)

for distance, name, path in results[:20]:

    print(
        f"{distance:10.5f}   {name}"
    )


# -------------------------------------------------
# Compare Boat samples too
# -------------------------------------------------

boat_files = list(
    (DATASET_DIR / "15. Boat").glob("*.npy")
)

print(
    f"\nBoat samples found: {len(boat_files)}"
)

boat_results = []

for path in boat_files:

    try:

        sample = np.load(path)

        distance = sequence_distance(
            webcam,
            sample
        )

        boat_results.append(
            (
                distance,
                path.name
            )
        )

    except Exception:
        pass


boat_results.sort(
    key=lambda x: x[0]
)


print("\n")
print("=" * 75)
print("CLOSEST BOAT LANDMARK SEQUENCES")
print("=" * 75)

for distance, name in boat_results[:20]:

    print(
        f"{distance:10.5f}   {name}"
    )


# -------------------------------------------------
# Statistics
# -------------------------------------------------

if results and boat_results:

    best_clock = results[0][0]
    best_boat = boat_results[0][0]

    avg_clock = np.mean(
        [x[0] for x in results]
    )

    avg_boat = np.mean(
        [x[0] for x in boat_results]
    )

    print("\n")
    print("=" * 75)
    print("DISTANCE SUMMARY")
    print("=" * 75)

    print(
        f"Best Clock distance : {best_clock:.5f}"
    )

    print(
        f"Best Boat distance  : {best_boat:.5f}"
    )

    print(
        f"Average Clock       : {avg_clock:.5f}"
    )

    print(
        f"Average Boat        : {avg_boat:.5f}"
    )

    if best_clock < best_boat:

        print(
            "\nWebcam landmarks are geometrically "
            "closer to Clock than Boat."
        )

    else:

        print(
            "\nWebcam landmarks are geometrically "
            "closer to Boat than Clock."
        )