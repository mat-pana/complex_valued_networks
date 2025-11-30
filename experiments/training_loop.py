import torch
from torch.nn import Module, BCEWithLogitsLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from typing import Optional

from utils.torch_utils import accuracy_from_tensors


def training_loop(
    model: Module,
    n_epochs: int,
    optimizer: Optimizer,
    train_loader: DataLoader,
    x_val: torch.Tensor,
    x_test: torch.Tensor,
    y_val: torch.Tensor,
    y_test: torch.Tensor,
    scheduler: Optional[_LRScheduler] = None,
    verbose: bool = True,
) -> dict:

    criterion = BCEWithLogitsLoss()

    train_accs = []
    val_accs = []
    test_accs = []

    for epoch in range(n_epochs):
        model.train()

        train_logits = []
        train_targets = []

        for batch in train_loader:
            *xs, y = batch
            xs = [x for x in xs]

            optimizer.zero_grad()
            out = model(*xs)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_logits.append(out.detach())
            train_targets.append(y)

        train_outputs = torch.cat(train_logits, dim=0)
        train_labels = torch.cat(train_targets, dim=0)
        train_accuracy = (
            (torch.where(train_outputs > 0, 1, 0) == train_labels).cpu().numpy().mean()
        )

        model.eval()
        with torch.no_grad():
            if isinstance(x_val, (tuple, list)):
                val_outputs = model(*x_val)
            else:
                val_outputs = model(x_val)
            val_accuracy = accuracy_from_tensors(val_outputs, y_val)

            if isinstance(x_test, (tuple, list)):
                test_outputs = model(*x_test)
            else:
                test_outputs = model(x_test)
            test_accuracy = accuracy_from_tensors(test_outputs, y_test)

        train_accs += [train_accuracy.item()]
        val_accs += [val_accuracy.item()]
        test_accs += [test_accuracy.item()]

        if verbose:
            print(epoch, train_accuracy, val_accuracy, test_accuracy)

    return {"train": train_accs, "val": val_accs, "test": test_accs}
