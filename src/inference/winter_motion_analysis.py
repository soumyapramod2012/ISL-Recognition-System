import numpy as np
from pathlib import Path


WINTER_DIR = Path(
    r"dataset\processed\landmarks\63. Winter"
)


# ---------------------------------------------------------
# Extract the center of a hand
# ---------------------------------------------------------

def hand_center(hand_data):
    """
    hand_data shape:
        (63,) = 21 landmarks × 3 coordinates

    Returns:
        [x, y] center of the detected hand
        None if the hand is missing
    """

    points = hand_data.reshape(21, 3)

    # A completely zero hand means no detection
    if np.all(points == 0):
        return None

    # Use mean of the 21 landmark positions
    return np.mean(points[:, :2], axis=0)


# ---------------------------------------------------------
# Extract hand trajectory
# ---------------------------------------------------------

def get_trajectory(data, start, end):
    trajectory = []

    for frame in data:
        center = hand_center(frame[start:end])

        if center is not None:
            trajectory.append(center)

    if not trajectory:
        return np.empty((0, 2))

    return np.array(trajectory)


# ---------------------------------------------------------
# Calculate motion statistics
# ---------------------------------------------------------

def analyse_motion(data):

    left = get_trajectory(data, 132, 195)
    right = get_trajectory(data, 195, 258)

    result = {}

    for name, trajectory in [
        ("left", left),
        ("right", right)
    ]:

        if len(trajectory) < 3:

            result[name] = {
                "frames": len(trajectory),
                "x_range": 0.0,
                "y_range": 0.0,
                "total_motion": 0.0,
                "direction_changes_x": 0,
                "direction_changes_y": 0,
            }

            continue

        x = trajectory[:, 0]
        y = trajectory[:, 1]

        dx = np.diff(x)
        dy = np.diff(y)

        # Remove tiny changes/noise
        threshold = 0.002

        dx_sign = np.sign(
            np.where(np.abs(dx) < threshold, 0, dx)
        )

        dy_sign = np.sign(
            np.where(np.abs(dy) < threshold, 0, dy)
        )

        # Remove zeros before detecting direction changes
        dx_sign = dx_sign[dx_sign != 0]
        dy_sign = dy_sign[dy_sign != 0]

        direction_changes_x = np.sum(
            dx_sign[1:] != dx_sign[:-1]
        )

        direction_changes_y = np.sum(
            dy_sign[1:] != dy_sign[:-1]
        )

        motion = np.sqrt(
            dx ** 2 + dy ** 2
        )

        result[name] = {
            "frames": len(trajectory),
            "x_range": np.ptp(x),
            "y_range": np.ptp(y),
            "total_motion": np.sum(motion),
            "direction_changes_x": int(direction_changes_x),
            "direction_changes_y": int(direction_changes_y),
        }

    return result


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 110)
    print("WINTER TRAINING DATA — HAND MOTION / OSCILLATION ANALYSIS")
    print("=" * 110)

    print()

    print(
        f"{'Sample':<12}"
        f"{'L Frames':>10}"
        f"{'L X Range':>12}"
        f"{'L Y Range':>12}"
        f"{'L Motion':>12}"
        f"{'L X Rev':>9}"
        f"{'L Y Rev':>9}"
        f"{'R Frames':>10}"
        f"{'R X Range':>12}"
        f"{'R Y Range':>12}"
        f"{'R Motion':>12}"
        f"{'R X Rev':>9}"
        f"{'R Y Rev':>9}"
    )

    print("-" * 110)

    files = sorted(WINTER_DIR.glob("*.npy"))

    for file in files:

        data = np.load(file)

        result = analyse_motion(data)

        L = result["left"]
        R = result["right"]

        print(
            f"{file.stem:<12}"
            f"{L['frames']:>10}"
            f"{L['x_range']:>12.5f}"
            f"{L['y_range']:>12.5f}"
            f"{L['total_motion']:>12.5f}"
            f"{L['direction_changes_x']:>9}"
            f"{L['direction_changes_y']:>9}"
            f"{R['frames']:>10}"
            f"{R['x_range']:>12.5f}"
            f"{R['y_range']:>12.5f}"
            f"{R['total_motion']:>12.5f}"
            f"{R['direction_changes_x']:>9}"
            f"{R['direction_changes_y']:>9}"
        )

    print()
    print("=" * 110)


if __name__ == "__main__":
    main()