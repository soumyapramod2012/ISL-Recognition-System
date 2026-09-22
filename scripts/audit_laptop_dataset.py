from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LAPTOP_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "INCLUDE"
    / "Electronics"
    / "56. Laptop"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "laptop_audit"
)

VIDEO_EXTENSIONS = {
    ".mov",
    ".mp4",
    ".avi",
    ".mkv",
}


def make_holistic():
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def inspect_video(video_path, holistic):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {
            "file": video_path.name,
            "frames": 0,
            "fps": 0.0,
            "duration": 0.0,
            "hand_frames": 0,
            "both_hand_frames": 0,
            "hand_ratio": 0.0,
            "error": "Cannot open video",
        }

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 0.0

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    left_count = 0
    right_count = 0
    both_count = 0
    hand_count = 0

    sample_frames = []
    frame_index = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        results = holistic.process(rgb)

        left = (
            results.left_hand_landmarks
            is not None
        )

        right = (
            results.right_hand_landmarks
            is not None
        )

        if left:
            left_count += 1

        if right:
            right_count += 1

        if left or right:
            hand_count += 1

        if left and right:
            both_count += 1

        # Save 5 representative frames.
        if (
            total_frames > 0
            and frame_index
            in {
                0,
                total_frames // 4,
                total_frames // 2,
                (3 * total_frames) // 4,
                total_frames - 1,
            }
        ):
            sample_frames.append(
                frame.copy()
            )

        frame_index += 1

    cap.release()

    duration = (
        frame_index / fps
        if fps > 0
        else 0.0
    )

    hand_ratio = (
        hand_count / frame_index
        if frame_index > 0
        else 0.0
    )

    return {
        "file": video_path.name,
        "frames": frame_index,
        "fps": fps,
        "duration": duration,
        "hand_frames": hand_count,
        "both_hand_frames": both_count,
        "left_frames": left_count,
        "right_frames": right_count,
        "hand_ratio": hand_ratio,
        "error": "",
        "samples": sample_frames,
    }


def create_contact_sheet(results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheets = []

    for result in results:

        samples = result.get(
            "samples",
            [],
        )

        if not samples:
            continue

        for i, image in enumerate(samples):

            resized = cv2.resize(
                image,
                (240, 180),
            )

            label = (
                f"{result['file']} "
                f"#{i + 1}"
            )

            cv2.putText(
                resized,
                label,
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )

            sheets.append(resized)

    if not sheets:
        return

    columns = 5
    rows = (
        len(sheets) + columns - 1
    ) // columns

    blank = np.zeros_like(
        sheets[0]
    )

    while len(sheets) < rows * columns:
        sheets.append(blank.copy())

    rows_data = []

    for r in range(rows):

        row = np.hstack(
            sheets[
                r * columns:
                (r + 1) * columns
            ]
        )

        rows_data.append(row)

    sheet = np.vstack(rows_data)

    output = (
        OUTPUT_DIR
        / "laptop_contact_sheet.jpg"
    )

    cv2.imwrite(
        str(output),
        sheet,
    )

    print()
    print(
        "Contact sheet:",
        output,
    )


def main():

    print("=" * 80)
    print("LAPTOP DATASET AUDIT")
    print("=" * 80)

    if not LAPTOP_DIR.exists():

        raise FileNotFoundError(
            f"Laptop folder not found:\n{LAPTOP_DIR}"
        )

    videos = sorted(
        p for p in LAPTOP_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in VIDEO_EXTENSIONS
    )

    print(
        f"Videos found: {len(videos)}"
    )

    if not videos:
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    with make_holistic() as holistic:

        for i, video in enumerate(
            videos,
            start=1,
        ):

            print(
                f"[{i}/{len(videos)}] "
                f"{video.name}"
            )

            result = inspect_video(
                video,
                holistic,
            )

            print(
                f"   frames={result['frames']} "
                f"duration={result['duration']:.2f}s "
                f"hands={result['hand_ratio']:.1%} "
                f"both={result['both_hand_frames']}"
            )

            results.append(result)

    # CSV-style text report
    report = (
        OUTPUT_DIR
        / "laptop_audit.txt"
    )

    with open(
        report,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "LAPTOP DATASET AUDIT\n"
        )
        f.write(
            "=" * 80 + "\n"
        )

        for r in results:

            f.write(
                f"{r['file']}\t"
                f"frames={r['frames']}\t"
                f"fps={r['fps']:.2f}\t"
                f"duration={r['duration']:.2f}\t"
                f"hand_ratio={r['hand_ratio']:.4f}\t"
                f"both={r['both_hand_frames']}\t"
                f"left={r.get('left_frames', 0)}\t"
                f"right={r.get('right_frames', 0)}\n"
            )

    print()
    print(
        "Report:",
        report,
    )

    create_contact_sheet(results)

    print()
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()