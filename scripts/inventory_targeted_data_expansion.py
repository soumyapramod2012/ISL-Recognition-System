import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = PROJECT_ROOT / "dataset" / "raw" / "INCLUDE"
LANDMARK_ROOT = PROJECT_ROOT / "dataset" / "processed" / "landmarks_legacy"

TARGET_CLASSES = {
    "1. Dog",
    "2. Cat",
    "3. Fish",
    "4. Bird",
    "7. Horse",
    "8. Animal",
    "58. Camera",
    "60. Radio",
    "63. Winter",
}

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv"}
LANDMARK_EXTENSIONS = {".npy"}


def stem_set(folder, extensions):
    return {
        p.stem
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    }


def main():
    if not RAW_ROOT.exists():
        print(f"Raw dataset not found: {RAW_ROOT}")
        return

    print("=" * 70)
    print("TARGETED DATA-EXPANSION INVENTORY")
    print("=" * 70)
    print(f"Raw root: {RAW_ROOT}")
    print(f"Legacy landmarks: {LANDMARK_ROOT}")
    print()

    total_videos = 0
    total_processed = 0
    total_missing = 0
    rows = []

    for class_name in sorted(TARGET_CLASSES):
        raw_class = RAW_ROOT / _category_for_class(class_name) / class_name

        # Fallback: locate the class directory anywhere below INCLUDE.
        if not raw_class.exists():
            matches = [
                p for p in RAW_ROOT.rglob(class_name)
                if p.is_dir()
            ]
            raw_class = matches[0] if matches else None

        if raw_class is None:
            rows.append((class_name, 0, 0, 0, "NOT FOUND"))
            continue

        landmark_class = LANDMARK_ROOT / class_name

        raw_stems = stem_set(raw_class, VIDEO_EXTENSIONS)
        processed_stems = (
            stem_set(landmark_class, LANDMARK_EXTENSIONS)
            if landmark_class.exists()
            else set()
        )

        missing = sorted(raw_stems - processed_stems)

        total_videos += len(raw_stems)
        total_processed += len(raw_stems & processed_stems)
        total_missing += len(missing)

        status = "EXPAND" if missing else "COMPLETE"
        rows.append(
            (
                class_name,
                len(raw_stems),
                len(raw_stems & processed_stems),
                len(missing),
                status,
            )
        )

        print(
            f"{class_name:15s} "
            f"raw={len(raw_stems):3d} "
            f"processed={len(raw_stems & processed_stems):3d} "
            f"missing={len(missing):3d} "
            f"{status}"
        )

        if missing:
            print("   Missing:")
            for stem in missing:
                print(f"      {stem}")

    print()
    print("=" * 70)
    print(
        f"TARGET TOTALS: raw={total_videos}, "
        f"processed={total_processed}, missing={total_missing}"
    )
    print("=" * 70)

    output = PROJECT_ROOT / "outputs" / "targeted_data_expansion_inventory.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "target_classes": sorted(TARGET_CLASSES),
        "raw_video_total": total_videos,
        "processed_total": total_processed,
        "missing_total": total_missing,
        "rows": [
            {
                "class": name,
                "raw": raw,
                "processed": processed,
                "missing": missing,
                "status": status,
            }
            for name, raw, processed, missing, status in rows
        ],
    }

    output.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(f"Saved: {output}")


def _category_for_class(class_name):
    # Known current project category mapping.
    if class_name in {
        "1. Dog", "2. Cat", "3. Fish", "4. Bird",
        "7. Horse", "8. Animal",
    }:
        return "Animals"

    if class_name in {
        "58. Camera", "60. Radio",
    }:
        return "Electronics"

    if class_name == "63. Winter":
        return "Seasons"

    return ""


if __name__ == "__main__":
    main()
