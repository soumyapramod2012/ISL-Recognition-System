import numpy as np
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

WINTER_DIR = Path(
    r"dataset\processed\landmarks\63. Winter"
)

WEBCAM_FILE = Path(
    r"outputs\webcam_recordings\winter_test_20260908_110507_landmarks.npy"
)


# ---------------------------------------------------------
# Analyse one landmark sequence
# ---------------------------------------------------------

def analyse_sequence(data):
    """
    Landmark format:
        Pose       = 0:132
        Left hand  = 132:195
        Right hand = 195:258
    """

    left = data[:, 132:195]
    right = data[:, 195:258]

    left_detected = np.any(left != 0, axis=1)
    right_detected = np.any(right != 0, axis=1)

    both = left_detected & right_detected
    left_only = left_detected & ~right_detected
    right_only = right_detected & ~left_detected
    none = ~left_detected & ~right_detected

    return {
        "frames": len(data),
        "both": int(np.sum(both)),
        "left_only": int(np.sum(left_only)),
        "right_only": int(np.sum(right_only)),
        "none": int(np.sum(none)),
    }


# ---------------------------------------------------------
# Print result
# ---------------------------------------------------------

def print_result(name, result):

    frames = result["frames"]

    print(
        f"{name:<18}"
        f"{frames:>7}"
        f"{result['both']:>8}"
        f"{result['left_only']:>9}"
        f"{result['right_only']:>9}"
        f"{result['none']:>8}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("WINTER TRAINING DATA — TWO-HAND ANALYSIS")
    print("=" * 70)

    if not WINTER_DIR.exists():
        print(f"\nERROR: Winter directory not found:")
        print(WINTER_DIR)
        return

    files = sorted(WINTER_DIR.glob("*.npy"))

    if not files:
        print("\nERROR: No .npy files found.")
        return

    print()
    print(
        f"{'Sample':<18}"
        f"{'Frames':>7}"
        f"{'Both':>8}"
        f"{'Left':>9}"
        f"{'Right':>9}"
        f"{'None':>8}"
    )
    print("-" * 70)

    totals = {
        "frames": 0,
        "both": 0,
        "left_only": 0,
        "right_only": 0,
        "none": 0,
    }

    for file in files:

        data = np.load(file)

        result = analyse_sequence(data)

        print_result(file.stem, result)

        for key in totals:
            totals[key] += result[key]

    print("-" * 70)

    print_result("TOTAL", totals)

    print()
    print("Training-set percentages:")

    if totals["frames"] > 0:

        for key, label in [
            ("both", "Both hands"),
            ("left_only", "Left only"),
            ("right_only", "Right only"),
            ("none", "No hands"),
        ]:

            percentage = totals[key] / totals["frames"] * 100

            print(f"{label:<15}: {percentage:6.2f}%")

    # -----------------------------------------------------
    # Webcam
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("WEBCAM RECORDING")
    print("=" * 70)

    if not WEBCAM_FILE.exists():

        print()
        print("Landmark file not found:")
        print(WEBCAM_FILE)

        print()
        print(
            "If your webcam landmarks have a different filename, "
            "tell me the filename."
        )

        return

    webcam = np.load(WEBCAM_FILE)

    result = analyse_sequence(webcam)

    print()
    print(
        f"{'Webcam':<18}"
        f"{result['frames']:>7}"
        f"{result['both']:>8}"
        f"{result['left_only']:>9}"
        f"{result['right_only']:>9}"
        f"{result['none']:>8}"
    )

    print()
    print("Webcam percentages:")

    frames = result["frames"]

    if frames > 0:

        print(
            f"Both hands     : "
            f"{result['both'] / frames * 100:.2f}%"
        )

        print(
            f"Left only      : "
            f"{result['left_only'] / frames * 100:.2f}%"
        )

        print(
            f"Right only     : "
            f"{result['right_only'] / frames * 100:.2f}%"
        )

        print(
            f"No hands       : "
            f"{result['none'] / frames * 100:.2f}%"
        )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()