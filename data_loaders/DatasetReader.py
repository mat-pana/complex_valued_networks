from os import PathLike
from os.path import join
import numpy as np
import pandas as pd
from typing import Literal, Tuple


class DatasetReader:

    def __init__(
        self, dataset: Literal["FordA", "FordB", "FaultDetectionA"] = "FordA"
    ) -> None:
        self.dataset = dataset
        self.path = lambda part: join("datasets", dataset, f"{dataset}_{part}.ts")

    def load(self) -> Tuple[np.array, ...]:
        out = []
        train_test = ["TRAIN", "TEST"]
        for part in train_test:
            path = self.path(part)
            series, labels = self.load_ts_file(path)
            if self.dataset == "FaultDetectionA":
                series, labels = self.select_subset(series, labels)
                series = self.sparsify_dataset(series)
            labels = self.encode_labels(labels)
            out += [series, labels]
        return out

    def select_subset(
        self, series: np.array, labels: np.array
    ) -> Tuple[np.array, np.array]:
        # Removes "undamaged" class.
        subset_mask = (labels != 0).flatten()
        series = series[subset_mask]
        labels = labels[subset_mask]

        return series, labels

    def encode_labels(self, labels: np.array) -> np.array:

        # Unifies labeling.
        if self.dataset == "FaultDetectionA":
            labels = np.where(labels == 2, 0, 1)
        else:
            labels = np.where(labels == -1, 0, 1)

        return labels

    def sparsify_dataset(self, series: np.array) -> np.array:
        # Downsampling.
        return series[:, ::10]

    def load_ts_file(self, path: PathLike) -> Tuple[np.array, np.array]:
        x = []
        y = []
        reading_data = False

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # detect start of data
                if line.startswith("@"):
                    if line.lower() == "@data":
                        reading_data = True
                    continue

                if reading_data:
                    # split into values and label
                    parts = [p.strip() for p in line.split(":")]

                    values_str = parts[0]
                    label = int(parts[1])

                    ts = np.array([float(x) for x in values_str.split(",") if x != ""])
                    x.append(ts)

                    y.append(label)

        x = np.array(x, dtype=float)
        y = np.array(y).reshape(-1, 1)

        return x, y
