import numpy as np
from pathlib import Path


WINTER_DIR = Path(
    r"dataset\processed\landmarks\63. Winter"
)


def analyse(file_path):
    data = np.load(file_path)

    left = data[:, 132:195]
    right = data[:, 195:258]

    left_detected = np.any(left != 0, axis=1)
    right_detected = np.any(right != 0, axis=1)

    # Count detected frames
    left_count = np.sum(left_detected)
    right_count = np.sum(right_detected)

    # ---------------------------------------------------------
    # Hand activity
    # ---------------------------------------------------------
    #
    # Calculate the average frame-to-frame movement of each hand.
    # Only compare consecutive frames where that hand is present
    # in both frames.
    #

    left_activity = []
    right_activity = []

    for i in range(1, len(data)):

        if left_detected[i - 1] and left_detected[i]:
            movement = np.mean(
                np.abs(left[i] - left[i - 1])
            )
            left_activity.append(movement)

        if right_detected[i - 1] and right_detected[i]:
            movement = np.mean(
                np.abs(right[i] - right[i - 1])
            )
            right_activity.append(movement)

    left_activity_mean = (
        np.mean(left_activity)
        if left_activity else 0.0
    )

    right_activity_mean = (
        np.mean(right_activity)
        if right_activity else 0.0
    )

    return {
        "frames": len(data),
        "left_pct": left_count / len(data) * 100,
        "right_pct": right_count / len(data) * 100,
        "none_pct": (
            np.sum(~left_detected & ~right_detected)
            / len(data) * 100
        ),
        "left_activity": left_activity_mean,
        "right_activity": right_activity_mean,
    }


def main():

    print("=" * 95)
    print("WINTER TRAINING DATA — HAND ACTIVITY ANALYSIS")
    print("=" * 95)

    print()

    print(
        f"{'Sample':<12}"
        f"{'Frames':>8}"
        f"{'Left %':>10}"
        f"{'Right %':>10}"
        f"{'None %':>10}"
        f"{'Left Move':>14}"
        f"{'Right Move':>15}"
    )

    print("-" * 95)

    files = sorted(WINTER_DIR.glob("*.npy"))

    all_results = []

    for file in files:

        result = analyse(file)

        all_results.append(result)

        print(
            f"{file.stem:<12}"
            f"{result['frames']:>8}"
            f"{result['left_pct']:>10.2f}"
            f"{result['right_pct']:>10.2f}"
            f"{result['none_pct']:>10.2f}"
            f"{result['left_activity']:>14.6f}"
            f"{result['right_activity']:>15.6f}"
        )

    print("-" * 95)

    if all_results:

        print(
            f"{'AVERAGE':<12}"
            f"{np.mean([r['frames'] for r in all_results]):>8.2f}"
            f"{np.mean([r['left_pct'] for r in all_results]):>10.2f}"
            f"{np.mean([r['right_pct'] for r in all_results]):>10.2f}"
            f"{np.mean([r['none_pct'] for r in all_results]):>10.2f}"
            f"{np.mean([r['left_activity'] for r in all_results]):>14.6f}"
            f"{np.mean([r['right_activity'] for r in all_results]):>15.6f}"
        )

    print()
    print("=" * 95)


if __name__ == "__main__":
    main()