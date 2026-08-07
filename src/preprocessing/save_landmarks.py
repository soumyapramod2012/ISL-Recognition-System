from pathlib import Path
import numpy as np


class LandmarkSaver:

    def __init__(self, output_dir):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(self, landmarks, label, video_name):

        label_dir = self.output_dir / label

        label_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = label_dir / f"{video_name}.npy"

        np.save(output_file, landmarks)

        return output_file