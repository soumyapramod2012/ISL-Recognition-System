"""
Video Reader
"""

import cv2


def open_video(video_path):
    """
    Opens a video file.
    """

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")

    return video


def read_first_frame(video):
    """
    Reads the first frame from a video.
    """

    success, frame = video.read()

    if not success:
        raise RuntimeError("Cannot read video frame.")

    return frame