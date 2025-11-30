import numpy as np
import random
import torch
from typing import List


def tensorise(values: List[np.array], device: str = "cuda:0") -> List[torch.tensor]:
    og_dtypes = [i.dtype for i in values]
    # complex64 is equivalent to conventional float32
    # with float32 for real and imaginary part
    get_dtype = lambda og_dtype: (
        torch.complex64
        if (og_dtype == "complex128" or og_dtype == "complex64")
        else torch.float32
    )
    target_dtypes = [get_dtype(i) for i in og_dtypes]
    return [
        torch.tensor(v, device=device).to(dt) for v, dt in zip(values, target_dtypes)
    ]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def accuracy_from_tensors(outputs: torch.Tensor, labels: torch.Tensor):
    return (torch.where(outputs > 0, 1, 0) == labels).cpu().numpy().mean()
