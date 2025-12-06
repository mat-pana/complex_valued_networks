import csv
from os import PathLike, makedirs
from os.path import join, exists
import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from typing import Literal, Tuple

from sklearn.model_selection import train_test_split
from data_loaders.DatasetReader import DatasetReader
from utils.torch_utils import tensorise


def load_data(
    letter: Literal["A", "B"],
    input_type: Literal["single_real", "single_recurrent", "single_complex", "dual"],
    device: str = "cuda:0",
    seed: int = 42,
) -> Tuple[Tensor, ...]:
    loader = DatasetReader(dataset=letter)
    x_train, y_train, x_test, y_test = loader.load()

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
    )

    x_time_train, x_freq_train = norm_fft(x_train)
    x_time_val, x_freq_val = norm_fft(x_val)
    x_time_test, x_freq_test = norm_fft(x_test)

    (
        x_time_train,
        x_freq_train,
        x_time_val,
        x_freq_val,
        x_time_test,
        x_freq_test,
        y_train,
        y_val,
        y_test,
    ) = tensorise(
        [
            x_time_train,
            x_freq_train,
            x_time_val,
            x_freq_val,
            x_time_test,
            x_freq_test,
            y_train,
            y_val,
            y_test,
        ],
        device=device,
    )

    if input_type in ["single_recurrent", "dual"]:
        x_time_train = x_time_train.unsqueeze(-1)
        x_time_val = x_time_val.unsqueeze(-1)
        x_time_test = x_time_test.unsqueeze(-1)

    if input_type in ["single_real", "single_recurrent"]:
        x_train = x_time_train
        x_val = x_time_val
        x_test = x_time_test

    elif input_type == "single_complex":
        x_train = x_freq_train
        x_val = x_freq_val
        x_test = x_freq_test

    elif input_type == "dual":
        x_train = x_time_train, x_freq_train
        x_val = x_time_val, x_freq_val
        x_test = x_time_test, x_freq_test

    batch_size = 256

    if isinstance(x_train, (tuple, list)):
        train_ds = TensorDataset(*x_train, y_train)
    else:
        train_ds = TensorDataset(x_train, y_train)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=False,
        num_workers=0,
    )

    return (
        train_loader,
        x_val,
        x_test,
        y_val,
        y_test,
    )


def norm_fft(series: np.array) -> Tuple[np.array, np.array]:
    mu = series.mean(1).reshape(-1, 1)
    sigma = series.std(1).reshape(-1, 1)
    normalised = (series - mu) / sigma
    ffts = np.fft.rfft(normalised, norm="ortho")
    return normalised, ffts


def write_log(
    res_dict: dict,
    experiment_name: str,
    dataset: Literal["FordA", "FordB", "FaultDetectionA"],
    log_path: PathLike = "results",
) -> None:
    results_to_csv(res_dict, experiment_name, dataset, log_path)
    log_time(res_dict, experiment_name, dataset, log_path)


def results_to_csv(
    res_dict: dict,
    experiment_name: str,
    dataset: Literal["FordA", "FordB", "FaultDetectionA"],
    log_path: PathLike = "results",
) -> None:
    needed_keys = ["train", "val", "test"]
    sub_dict = {k: res_dict[k] for k in needed_keys}

    df = pd.DataFrame(sub_dict)
    df.index.name = "epoch"
    full_dir = join(log_path, dataset)
    makedirs(full_dir, exist_ok=True)
    full_path = join(full_dir, f"{experiment_name}.csv")
    df.to_csv(full_path)


def log_time(
    res_dict: dict,
    experiment_name: str,
    dataset: Literal["FordA", "FordB", "FaultDetectionA"],
    log_path: PathLike = "results",
) -> None:
    full_dir = join(log_path, dataset)
    makedirs(full_dir, exist_ok=True)
    full_path = join(full_dir, "times.csv")

    file_exists = exists(full_path)

    with open(full_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["experiment", "duration"])

        writer.writerow([experiment_name, res_dict["experiment_time"]])