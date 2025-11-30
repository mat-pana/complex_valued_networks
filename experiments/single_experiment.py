from os import PathLike
import torch
import torch.nn as nn
from typing import Literal

from experiments.experiment_utils import load_data, results_to_csv
from experiments.training_loop import training_loop
from utils.torch_utils import set_seed


def single_experiment(
    model: nn.Module,
    learning_rate: float,
    n_epochs: int,
    experiment_name: str,
    dataset: Literal["FordA", "FordB", "FaultDetectionA"],
    device: str = "cuda:0",
    log_path: PathLike = "results",
    seed: int = 42,
) -> None:
    set_seed(seed)

    input_type = model.input_type

    (
        train_loader,
        x_val,
        x_test,
        y_val,
        y_test,
    ) = load_data(dataset, input_type, device, seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6
    )

    res_dict = training_loop(
        model,
        n_epochs,
        optimizer,
        train_loader,
        x_val,
        x_test,
        y_val,
        y_test,
        scheduler,
    )

    results_to_csv(res_dict, experiment_name, dataset, log_path)
