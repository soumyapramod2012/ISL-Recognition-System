import numpy as np
from pathlib import Path


DATASET_PATH = Path(
    "dataset/processed/landmarks"
)

LANDMARK_SIZE = 258


def analyze_dataset():

    files = sorted(
        DATASET_PATH.rglob("*.npy")
    )

    if not files:

        print("No .npy files found.")
        return

    frame_counts = []

    pose_x = []
    pose_y = []
    pose_z = []

    hand_x = []
    hand_y = []
    hand_z = []

    invalid_files = []

    class_counts = {}

    for file in files:

        try:

            landmarks = np.load(file)

            # ---------------- Validation ----------------

            if landmarks.ndim != 2:

                invalid_files.append(
                    (file, "Not a 2D array")
                )

                continue

            if landmarks.shape[1] != LANDMARK_SIZE:

                invalid_files.append(
                    (
                        file,
                        f"Expected {LANDMARK_SIZE} features, "
                        f"found {landmarks.shape[1]}",
                    )
                )

                continue

            if landmarks.shape[0] == 0:

                invalid_files.append(
                    (file, "Empty sequence")
                )

                continue

            if np.isnan(landmarks).any():

                invalid_files.append(
                    (file, "Contains NaN")
                )

                continue

            if np.isinf(landmarks).any():

                invalid_files.append(
                    (file, "Contains Inf")
                )

                continue

            # ---------------- Frame count ----------------

            frame_counts.append(
                landmarks.shape[0]
            )

            # ---------------- Class ----------------

            class_name = file.parent.name

            class_counts[class_name] = (
                class_counts.get(class_name, 0) + 1
            )

            # ---------------- Pose ----------------

            # Pose occupies:
            # 33 landmarks × 4 values = 132 features

            pose = landmarks[:, :132]

            pose_x.extend(
                pose[:, 0::4].flatten()
            )

            pose_y.extend(
                pose[:, 1::4].flatten()
            )

            pose_z.extend(
                pose[:, 2::4].flatten()
            )

            # ---------------- Hands ----------------

            # Left hand:
            # 63 features

            left_hand = landmarks[:, 132:195]

            # Right hand:
            # 63 features

            right_hand = landmarks[:, 195:258]

            hands = np.concatenate(
                [
                    left_hand,
                    right_hand,
                ],
                axis=1,
            )

            hand_x.extend(
                hands[:, 0::3].flatten()
            )

            hand_y.extend(
                hands[:, 1::3].flatten()
            )

            hand_z.extend(
                hands[:, 2::3].flatten()
            )

        except Exception as error:

            invalid_files.append(
                (file, str(error))
            )

    # ==================================================
    # RESULTS
    # ==================================================

    frame_counts = np.array(
        frame_counts
    )

    print()
    print("=" * 60)
    print("LANDMARK DATASET DIAGNOSTIC")
    print("=" * 60)

    print()
    print("DATASET")
    print("-" * 60)

    print(
        f"Videos Found       : {len(files)}"
    )

    print(
        f"Valid Videos       : {len(frame_counts)}"
    )

    print(
        f"Invalid Videos     : {len(invalid_files)}"
    )

    print()
    print("FRAME LENGTH")
    print("-" * 60)

    print(
        f"Minimum Frames     : {frame_counts.min()}"
    )

    print(
        f"Maximum Frames     : {frame_counts.max()}"
    )

    print(
        f"Average Frames     : {frame_counts.mean():.2f}"
    )

    print(
        f"Median Frames      : {np.median(frame_counts):.2f}"
    )

    print(
        f"Frames < 60        : "
        f"{np.sum(frame_counts < 60)}"
    )

    print(
        f"Frames = 60        : "
        f"{np.sum(frame_counts == 60)}"
    )

    print(
        f"Frames > 60        : "
        f"{np.sum(frame_counts > 60)}"
    )

    print()
    print("CLASS DISTRIBUTION")
    print("-" * 60)

    for class_name in sorted(class_counts):

        print(
            f"{class_name:<20} : "
            f"{class_counts[class_name]}"
        )

    print()
    print("POSE COORDINATES")
    print("-" * 60)

    print(
        f"X Min              : {min(pose_x):.6f}"
    )

    print(
        f"X Max              : {max(pose_x):.6f}"
    )

    print(
        f"X Mean             : {np.mean(pose_x):.6f}"
    )

    print(
        f"Y Min              : {min(pose_y):.6f}"
    )

    print(
        f"Y Max              : {max(pose_y):.6f}"
    )

    print(
        f"Y Mean             : {np.mean(pose_y):.6f}"
    )

    print(
        f"Z Min              : {min(pose_z):.6f}"
    )

    print(
        f"Z Max              : {max(pose_z):.6f}"
    )

    print(
        f"Z Mean             : {np.mean(pose_z):.6f}"
    )

    print()
    print("HAND COORDINATES")
    print("-" * 60)

    print(
        f"X Min              : {min(hand_x):.6f}"
    )

    print(
        f"X Max              : {max(hand_x):.6f}"
    )

    print(
        f"X Mean             : {np.mean(hand_x):.6f}"
    )

    print(
        f"Y Min              : {min(hand_y):.6f}"
    )

    print(
        f"Y Max              : {max(hand_y):.6f}"
    )

    print(
        f"Y Mean             : {np.mean(hand_y):.6f}"
    )

    print(
        f"Z Min              : {min(hand_z):.6f}"
    )

    print(
        f"Z Max              : {max(hand_z):.6f}"
    )

    print(
        f"Z Mean             : {np.mean(hand_z):.6f}"
    )

    # ---------------- Invalid files ----------------

    if invalid_files:

        print()
        print("INVALID FILES")
        print("-" * 60)

        for file, reason in invalid_files:

            print(file)
            print("Reason :", reason)
            print()

    print()
    print("=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":

    analyze_dataset()