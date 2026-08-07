from pathlib import Path


class DatasetProcessor:

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

    def get_video_files(self):

        videos = []

        for file in self.dataset_path.rglob("*"):

            if file.suffix.lower() in [".mov", ".mp4", ".avi"]:

                label = file.parent.name

                videos.append(
                    {
                        "label": label,
                        "path": file,
                    }
                )

        return videos