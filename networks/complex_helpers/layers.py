import torch
import torch.nn as nn
from typing import Literal

from networks.complex_helpers.initialisation_schemes import *

class ComplexLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: Literal["he", "xavier"] = "he",
        gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(in_features, out_features, dtype=torch.complex64)
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_features, dtype=torch.complex64))
            if bias
            else None
        )

        if init == "he":
            complex_kaiming_normal(self.weight, fan_in=in_features, gain=gain)
        elif init == "xavier":
            complex_xavier_normal(
                self.weight, fan_in=in_features, fan_out=out_features, gain=gain
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


class ComplexDropout(nn.Module):

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = torch.bernoulli(torch.full_like(x.real, keep)) / keep
        return torch.complex(x.real * mask, x.imag * mask)
