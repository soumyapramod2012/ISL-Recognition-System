import numpy as np


USER_FILE = r"outputs\webcam_recordings\winter_legacy_mp_0.10.21_landmarks.npy"


def hand_center(hand_data):
    points = hand_data.reshape(21, 3)

    if np.all(points == 0):
        return None

    return np.mean(points[:, :2], axis=0)


def get_trajectory(data, start, end):
    trajectory = []

    for frame in data:
        center = hand_center(frame[start:end])

        if center is not None:
            trajectory.append(center)

    if not trajectory:
        return np.empty((0, 2))

    return np.array(trajectory)


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

        threshold = 0.002

        dx_sign = np.sign(
            np.where(np.abs(dx) < threshold, 0, dx)
        )

        dy_sign = np.sign(
            np.where(np.abs(dy) < threshold, 0, dy)
        )

        dx_sign = dx_sign[dx_sign != 0]
        dy_sign = dy_sign[dy_sign != 0]

        direction_changes_x = np.sum(
            dx_sign[1:] != dx_sign[:-1]
        )

        direction_changes_y = np.sum(
            dy_sign[1:] != dy_sign[:-1]
        )

        motion = np.sqrt(dx ** 2 + dy ** 2)

        result[name] = {
            "frames": len(trajectory),
            "x_range": np.ptp(x),
            "y_range": np.ptp(y),
            "total_motion": np.sum(motion),
            "direction_changes_x": int(direction_changes_x),
            "direction_changes_y": int(direction_changes_y),
        }

    return result


data = np.load(USER_FILE)

result = analyse_motion(data)

L = result["left"]
R = result["right"]

print()
print("=" * 90)
print("NEW WINTER RECORDING — MEDIA PIPE 0.10.21 HAND MOTION")
print("=" * 90)

print()

print("Video landmark frames:", len(data))

print()

print(
    f"{'Hand':<10}"
    f"{'Frames':>10}"
    f"{'X Range':>14}"
    f"{'Y Range':>14}"
    f"{'Motion':>14}"
    f"{'X Reversals':>14}"
    f"{'Y Reversals':>14}"
)

print("-" * 90)

for name, H in [("Left", L), ("Right", R)]:

    print(
        f"{name:<10}"
        f"{H['frames']:>10}"
        f"{H['x_range']:>14.5f}"
        f"{H['y_range']:>14.5f}"
        f"{H['total_motion']:>14.5f}"
        f"{H['direction_changes_x']:>14}"
        f"{H['direction_changes_y']:>14}"
    )

print()
print("=" * 90)