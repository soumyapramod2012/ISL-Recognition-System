import json
from pathlib import Path


class LabelEncoder:

    def __init__(self):

        self.classes = []

        self.label_to_index = {}
        self.index_to_label = {}

    def fit(self, samples):

        labels = sorted({sample["label"] for sample in samples})

        self.classes = list(labels)

        self.label_to_index = {
            label: index
            for index, label in enumerate(labels)
        }

        self.index_to_label = {
            index: label
            for label, index in self.label_to_index.items()
        }

    def encode(self, label):

        return self.label_to_index[label]

    def decode(self, index):

        return self.index_to_label[index]

    def save(self, path):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(path, "w") as file:

            json.dump(
                self.label_to_index,
                file,
                indent=4,
            )