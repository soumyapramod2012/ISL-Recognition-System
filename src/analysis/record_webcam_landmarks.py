"""
Experimental webcam landmark recorder for the generalized ISL project.

Purpose:
    Record exactly the 258-feature landmark vectors produced by the
    same LegacyLandmarkExtractor used to generate the training dataset.

This script is diagnostic only.

It does NOT:
    - load or modify the trained model
    - modify the dataset
    - modify src/inference/realtime.py

Controls:
    R = start recording
    S = stop and save recording
    C = clear current recording
    Q = quit

Recommended test:
    1. Open the webcam.
    2. Press R.
    3. Perform the SAME Clock sign that produced "Truck".
    4. Continue for about 3-5 seconds.
    5. Press S.
    6. Press Q.
"""

from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from src.preprocessing.legacy_landmark_extractor import (
    LegacyLandmarkExtractor
)


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 0

OUTPUT_DIR = Path(
    "outputs/webcam_test"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("WEBCAM 258-FEATURE LANDMARK RECORDER")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\nOutput directory:",
        OUTPUT_DIR
    )

    print("\nControls:")
    print("  R = start recording")
    print("  S = stop and save")
    print("  C = clear recording")
    print("  Q = quit")

    # --------------------------------------------------------
    # MediaPipe extractor
    # --------------------------------------------------------

    print(
        "\nInitializing MediaPipe Holistic..."
    )

    extractor = LegacyLandmarkExtractor()

    # --------------------------------------------------------
    # Webcam
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not cap.isOpened():

        extractor.close()

        raise RuntimeError(
            f"Could not open webcam "
            f"(camera index {CAMERA_INDEX})"
        )

    print("Webcam opened.")

    recording = False
    frames = []

    saved_files = []

    try:

        while True:

            success, frame = cap.read()

            if not success:

                print(
                    "ERROR: Could not read webcam frame."
                )

                break

            # ------------------------------------------------
            # Extract exactly 258 legacy features
            # ------------------------------------------------

            try:

                landmarks = extractor.extract(
                    frame
                )

            except Exception as exc:

                print(
                    f"Landmark extraction error: {exc}"
                )

                continue

            # ------------------------------------------------
            # Record
            # ------------------------------------------------

            if recording:

                frames.append(
                    landmarks.copy()
                )

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            display = frame.copy()

            height, width = display.shape[:2]

            # Status background
            cv2.rectangle(
                display,
                (0, 0),
                (width, 125),
                (0, 0, 0),
                -1
            )

            if recording:

                status = "RECORDING"

                status_color = (
                    0,
                    0,
                    255
                )

            else:

                status = "READY"

                status_color = (
                    0,
                    255,
                    0
                )

            cv2.putText(
                display,
                status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                status_color,
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                display,
                f"Recorded frames: {len(frames)}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                display,
                "R: Record   S: Save   C: Clear   Q: Quit",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            cv2.imshow(
                "ISL Webcam Landmark Recorder",
                display
            )

            # ------------------------------------------------
            # Keyboard
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            # Start recording
            if key == ord("r"):

                frames = []

                recording = True

                print(
                    "\nRecording started..."
                )

            # Stop + save
            elif key == ord("s"):

                if not frames:

                    print(
                        "\nNo frames recorded."
                    )

                    continue

                recording = False

                data = np.asarray(
                    frames,
                    dtype=np.float32
                )

                # Validate shape
                print(
                    "\nStopping recording..."
                )

                print(
                    "Shape:",
                    data.shape
                )

                if data.ndim != 2:

                    print(
                        "ERROR: Invalid recorded data."
                    )

                    frames = []

                    continue

                if data.shape[1] != 258:

                    print(
                        "ERROR: Expected 258 features, "
                        f"found {data.shape[1]}"
                    )

                    frames = []

                    continue

                if np.isnan(data).any():

                    print(
                        "ERROR: NaN values detected."
                    )

                    frames = []

                    continue

                if np.isinf(data).any():

                    print(
                        "ERROR: Infinite values detected."
                    )

                    frames = []

                    continue

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )

                filename = (
                    f"clock_webcam_{timestamp}.npy"
                )

                output_path = (
                    OUTPUT_DIR / filename
                )

                np.save(
                    output_path,
                    data
                )

                saved_files.append(
                    output_path
                )

                print(
                    "\nSaved:"
                )

                print(
                    output_path
                )

                print(
                    "Frames:",
                    data.shape[0]
                )

                print(
                    "Features:",
                    data.shape[1]
                )

                # Keep data available for another save
                # unless user presses C.
                print(
                    "\nPress C to clear or R to record again."
                )

            # Clear
            elif key == ord("c"):

                frames = []

                recording = False

                print(
                    "\nRecording cleared."
                )

            # Quit
            elif key == ord("q"):

                break

    finally:

        cap.release()

        cv2.destroyAllWindows()

        extractor.close()

        print(
            "\nWebcam recorder stopped."
        )

        if saved_files:

            print(
                "\nSaved files:"
            )

            for path in saved_files:

                print(
                    " ",
                    path
                )


if __name__ == "__main__":
    main()
