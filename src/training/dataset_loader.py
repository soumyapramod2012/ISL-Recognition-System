from pathlib import Path


class DatasetLoader:

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

    def load(self):

        samples = []

        for file in self.dataset_path.rglob("*.npy"):

            samples.append(
                {
                    "label": file.parent.name,
                    "path": file,
                }
            )

        return samples