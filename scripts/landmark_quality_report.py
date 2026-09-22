import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANDMARK_ROOT = PROJECT_ROOT / "dataset" / "processed" / "landmarks"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "landmark_quality_report.csv"


def analyse_file(path):
    data = np.load(path)

    total_frames = data.shape[0]

    # Landmark layout:
    # Pose       = 33 x 4 = 132
    # Left hand  = 21 x 3 = 63
    # Right hand = 21 x 3 = 63
    #
    # Total = 258

    pose = data[:, :132]
    left_hand = data[:, 132:195]
    right_hand = data[:, 195:258]

    # Hand landmark coordinates are zero when the hand is not detected.
    left_hand = left_hand.reshape(total_frames, 21, 3)
    right_hand = right_hand.reshape(total_frames, 21, 3)

    left_detected = np.any(np.abs(left_hand) > 1e-6, axis=(1, 2))
    right_detected = np.any(np.abs(right_hand) > 1e-6, axis=(1, 2))

    both_detected = left_detected & right_detected
    either_detected = left_detected | right_detected
    no_hand = ~either_detected

    return {
        "file": str(path.relative_to(PROJECT_ROOT)),
        "class": path.parent.name,
        "frames": total_frames,

        "left_hand_frames": int(np.sum(left_detected)),
        "right_hand_frames": int(np.sum(right_detected)),
        "both_hand_frames": int(np.sum(both_detected)),
        "no_hand_frames": int(np.sum(no_hand)),

        "left_hand_pct": float(np.mean(left_detected) * 100),
        "right_hand_pct": float(np.mean(right_detected) * 100),
        "both_hand_pct": float(np.mean(both_detected) * 100),
        "no_hand_pct": float(np.mean(no_hand) * 100),

        "nan_values": int(np.isnan(data).sum()),
        "inf_values": int(np.isinf(data).sum()),
    }


def main():
    files = sorted(LANDMARK_ROOT.rglob("*.npy"))

    print("=" * 80)
    print("LANDMARK QUALITY DIAGNOSTIC")
    print("=" * 80)

    print(f"Landmark files found: {len(files)}")

    if not files:
        print("No landmark files found.")
        return

    results = []

    for i, path in enumerate(files, start=1):
        try:
            result = analyse_file(path)
            results.append(result)

        except Exception as e:
            print(f"ERROR: {path}")
            print(f"       {e}")

        if i % 25 == 0 or i == len(files):
            print(f"Processed {i}/{len(files)}")

    # ---------------------------------------------------------
    # Save detailed CSV
    # ---------------------------------------------------------

    fieldnames = [
        "file",
        "class",
        "frames",
        "left_hand_frames",
        "right_hand_frames",
        "both_hand_frames",
        "no_hand_frames",
        "left_hand_pct",
        "right_hand_pct",
        "both_hand_pct",
        "no_hand_pct",
        "nan_values",
        "inf_values",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ---------------------------------------------------------
    # Overall statistics
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)

    def average(key):
        return np.mean([r[key] for r in results])

    print(f"Average left-hand detection  : {average('left_hand_pct'):.2f}%")
    print(f"Average right-hand detection : {average('right_hand_pct'):.2f}%")
    print(f"Average both-hand detection  : {average('both_hand_pct'):.2f}%")
    print(f"Average no-hand frames       : {average('no_hand_pct'):.2f}%")

    # ---------------------------------------------------------
    # Worst samples by both-hand detection
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("10 SAMPLES WITH LOWEST BOTH-HAND DETECTION")
    print("=" * 80)

    worst = sorted(
        results,
        key=lambda r: r["both_hand_pct"]
    )[:10]

    for r in worst:
        print(
            f"{r['both_hand_pct']:6.2f}% | "
            f"{r['class']:15s} | "
            f"{r['file']}"
        )

    # ---------------------------------------------------------
    # Samples with highest no-hand percentage
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("10 SAMPLES WITH HIGHEST NO-HAND PERCENTAGE")
    print("=" * 80)

    worst_no_hand = sorted(
        results,
        key=lambda r: r["no_hand_pct"],
        reverse=True
    )[:10]

    for r in worst_no_hand:
        print(
            f"{r['no_hand_pct']:6.2f}% | "
            f"{r['class']:15s} | "
            f"{r['file']}"
        )

    # ---------------------------------------------------------
    # Class-level statistics
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("CLASS-LEVEL STATISTICS")
    print("=" * 80)

    classes = sorted(set(r["class"] for r in results))

    print(
        f"{'Class':20s} "
        f"{'Samples':>7s} "
        f"{'Both %':>10s} "
        f"{'No-hand %':>12s}"
    )

    print("-" * 55)

    for cls in classes:
        class_results = [
            r for r in results
            if r["class"] == cls
        ]

        both = np.mean(
            [r["both_hand_pct"] for r in class_results]
        )

        no_hand = np.mean(
            [r["no_hand_pct"] for r in class_results]
        )

        print(
            f"{cls:20s} "
            f"{len(class_results):7d} "
            f"{both:10.2f} "
            f"{no_hand:12.2f}"
        )

    print()
    print("=" * 80)
    print(f"Detailed report saved to:")
    print(OUTPUT_CSV)
    print("=" * 80)


if __name__ == "__main__":
    main()