import numpy as np
from pathlib import Path


WINTER_DIR = Path(
    r"dataset\processed\landmarks\63. Winter"
)

USER_FILE = Path(
    r"outputs\webcam_recordings\winter_legacy_mp_0.10.21_landmarks.npy"
)


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

    if len(trajectory) < 2:
        return np.empty((0, 2))

    return np.array(trajectory)


def analyse_hand(data, start, end):

    trajectory = get_trajectory(data, start, end)

    if len(trajectory) < 2:
        return {
            "frames": len(trajectory),
            "avg_motion": 0.0,
            "avg_x_motion": 0.0,
            "avg_y_motion": 0.0,
            "x_range": 0.0,
            "y_range": 0.0,
        }

    x = trajectory[:, 0]
    y = trajectory[:, 1]

    dx = np.diff(x)
    dy = np.diff(y)

    motion = np.sqrt(dx ** 2 + dy ** 2)

    return {
        "frames": len(trajectory),
        "avg_motion": np.mean(motion),
        "avg_x_motion": np.mean(np.abs(dx)),
        "avg_y_motion": np.mean(np.abs(dy)),
        "x_range": np.ptp(x),
        "y_range": np.ptp(y),
    }


def analyse_file(path):

    data = np.load(path)

    return {
        "left": analyse_hand(data, 132, 195),
        "right": analyse_hand(data, 195, 258),
    }


print()
print("=" * 105)
print("NORMALIZED WINTER HAND-MOTION COMPARISON")
print("=" * 105)

print()

print(
    f"{'Sample':<14}"
    f"{'L Frames':>10}"
    f"{'L AvgMove':>12}"
    f"{'L XMove':>12}"
    f"{'L YMove':>12}"
    f"{'R Frames':>10}"
    f"{'R AvgMove':>12}"
    f"{'R XMove':>12}"
    f"{'R YMove':>12}"
)

print("-" * 105)


# Winter training samples

for file in sorted(WINTER_DIR.glob("*.npy")):

    result = analyse_file(file)

    L = result["left"]
    R = result["right"]

    print(
        f"{file.stem:<14}"
        f"{L['frames']:>10}"
        f"{L['avg_motion']:>12.5f}"
        f"{L['avg_x_motion']:>12.5f}"
        f"{L['avg_y_motion']:>12.5f}"
        f"{R['frames']:>10}"
        f"{R['avg_motion']:>12.5f}"
        f"{R['avg_x_motion']:>12.5f}"
        f"{R['avg_y_motion']:>12.5f}"
    )


# User recording

result = analyse_file(USER_FILE)

L = result["left"]
R = result["right"]

print()
print("-" * 105)
print("USER RECORDING — MediaPipe 0.10.21")
print("-" * 105)

print(
    f"{'USER':<14}"
    f"{L['frames']:>10}"
    f"{L['avg_motion']:>12.5f}"
    f"{L['avg_x_motion']:>12.5f}"
    f"{L['avg_y_motion']:>12.5f}"
    f"{R['frames']:>10}"
    f"{R['avg_motion']:>12.5f}"
    f"{R['avg_x_motion']:>12.5f}"
    f"{R['avg_y_motion']:>12.5f}"
)

print()
print("=" * 105)