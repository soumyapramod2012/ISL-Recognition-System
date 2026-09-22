import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_FILE = PROJECT_ROOT / "outputs" / "test_set_predictions.csv"
QUALITY_FILE = PROJECT_ROOT / "outputs" / "landmark_quality_report.csv"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "test_predictions_with_quality.csv"


def normalize_path(path):
    return str(Path(path)).replace("/", "\\").lower()


# ---------------------------------------------------------
# Load prediction results
# ---------------------------------------------------------

with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
    predictions = list(csv.DictReader(f))


# ---------------------------------------------------------
# Load landmark quality results
# ---------------------------------------------------------

with open(QUALITY_FILE, "r", encoding="utf-8") as f:
    quality = list(csv.DictReader(f))


quality_by_file = {
    normalize_path(row["file"]): row
    for row in quality
}


# ---------------------------------------------------------
# Combine
# ---------------------------------------------------------

combined = []

for row in predictions:
    key = normalize_path(row["file"])

    if key not in quality_by_file:
        print(f"WARNING: Quality data not found for {row['file']}")
        continue

    q = quality_by_file[key]

    combined.append({
        "actual": row["actual"],
        "predicted": row["predicted"],
        "confidence": row["confidence"],
        "correct": row["correct"],
        "left_hand_pct": q["left_hand_pct"],
        "right_hand_pct": q["right_hand_pct"],
        "no_hand_pct": q["no_hand_pct"],
        "frames": q["frames"],
        "file": row["file"],
    })


# ---------------------------------------------------------
# Save combined CSV
# ---------------------------------------------------------

fieldnames = [
    "actual",
    "predicted",
    "confidence",
    "correct",
    "left_hand_pct",
    "right_hand_pct",
    "no_hand_pct",
    "frames",
    "file",
]

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(combined)


# ---------------------------------------------------------
# Convert numeric fields
# ---------------------------------------------------------

for row in combined:
    row["confidence"] = float(row["confidence"])
    row["left_hand_pct"] = float(row["left_hand_pct"])
    row["right_hand_pct"] = float(row["right_hand_pct"])
    row["no_hand_pct"] = float(row["no_hand_pct"])


correct_rows = [
    r for r in combined
    if r["correct"].lower() == "true"
]

wrong_rows = [
    r for r in combined
    if r["correct"].lower() == "false"
]


def average(rows, field):
    if not rows:
        return 0.0
    return sum(r[field] for r in rows) / len(rows)


# ---------------------------------------------------------
# Overall comparison
# ---------------------------------------------------------

print()
print("=" * 80)
print("TEST PREDICTION vs LANDMARK QUALITY")
print("=" * 80)

print()
print(f"Total test samples : {len(combined)}")
print(f"Correct            : {len(correct_rows)}")
print(f"Wrong              : {len(wrong_rows)}")

print()
print("AVERAGE LANDMARK QUALITY")
print("-" * 80)

print(
    f"{'Metric':25s}"
    f"{'Correct':>15s}"
    f"{'Wrong':>15s}"
)

print("-" * 80)

for field, name in [
    ("left_hand_pct", "Left-hand detection %"),
    ("right_hand_pct", "Right-hand detection %"),
    ("no_hand_pct", "No-hand frames %"),
]:
    print(
        f"{name:25s}"
        f"{average(correct_rows, field):15.2f}"
        f"{average(wrong_rows, field):15.2f}"
    )


# ---------------------------------------------------------
# Individual wrong predictions
# ---------------------------------------------------------

print()
print("=" * 80)
print("WRONG PREDICTIONS WITH LANDMARK QUALITY")
print("=" * 80)

for r in wrong_rows:
    print(
        f"Actual: {r['actual']:15s} | "
        f"Predicted: {r['predicted']:15s} | "
        f"Conf: {r['confidence']:.4f} | "
        f"Left: {r['left_hand_pct']:6.2f}% | "
        f"Right: {r['right_hand_pct']:6.2f}% | "
        f"No-hand: {r['no_hand_pct']:6.2f}%"
    )
    print(f"    {r['file']}")


# ---------------------------------------------------------
# Highest no-hand among wrong predictions
# ---------------------------------------------------------

print()
print("=" * 80)
print("WRONG PREDICTIONS — HIGHEST NO-HAND %")
print("=" * 80)

for r in sorted(
    wrong_rows,
    key=lambda x: x["no_hand_pct"],
    reverse=True,
)[:10]:

    print(
        f"{r['no_hand_pct']:6.2f}% | "
        f"{r['actual']:15s} → "
        f"{r['predicted']:15s} | "
        f"{r['file']}"
    )


# ---------------------------------------------------------
# Highest no-hand among correct predictions
# ---------------------------------------------------------

print()
print("=" * 80)
print("CORRECT PREDICTIONS — HIGHEST NO-HAND %")
print("=" * 80)

for r in sorted(
    correct_rows,
    key=lambda x: x["no_hand_pct"],
    reverse=True,
)[:10]:

    print(
        f"{r['no_hand_pct']:6.2f}% | "
        f"{r['actual']:15s} | "
        f"{r['file']}"
    )


print()
print("=" * 80)
print(f"Combined report saved to:")
print(OUTPUT_FILE)
print("=" * 80)