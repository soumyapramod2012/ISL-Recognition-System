import cv2
import numpy as np

from src.preprocessing.dataset_processor import DatasetProcessor
from src.preprocessing.landmark_extractor import LandmarkExtractor
from src.preprocessing.save_landmarks import LandmarkSaver


DATASET = "dataset/raw/INCLUDE"

OUTPUT = "dataset/processed/landmarks"


processor = DatasetProcessor(DATASET)

extractor = LandmarkExtractor()

saver = LandmarkSaver(OUTPUT)

videos = processor.get_video_files()

'''videos = videos[:1]'''

print(f"Videos Found : {len(videos)}")

for index, video in enumerate(videos, start=1):

    print(f"[{index}/{len(videos)}] {video['path']}")

    cap = cv2.VideoCapture(str(video["path"]))

    frames = []

    while True:

        success, frame = cap.read()

        if not success:
            break

        vector = extractor.extract(frame)

        frames.append(vector)

    cap.release()

    frames = np.array(frames, dtype=np.float32)

    saver.save(
        frames,
        video["label"],
        video["path"].stem,
    )

extractor.close()

print()

print("Finished.")