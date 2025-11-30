import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Literal

from networks.complex_helpers.activation_functions import *
from networks.complex_helpers.initialisation_schemes import *
from networks.complex_helpers.layers import *
from networks.complex_helpers.utils import complex_gain


class ComplexFeedForward(nn.Module):

    input_type = "single_complex"

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        activation: Literal[
            "modrelu",
            "cardioid",
            "splitgelu",
            "zrelu",
            "crelu",
            "cleakyrelu",
            "celu",
            "softsign",
        ] = "modrelu",
        dropout_val: float = 0.0,
        no_top: bool = False,
        init: Literal["he", "xavier"] = "he",
        device: str = "cuda:0",
    ) -> None:
        super().__init__()
        self.device = device
        self.no_top = no_top
        layers = []
        acts = []

        prev = input_size
        for h in hidden_layers:
            gain = complex_gain(
                activation,
            )
            layers.append(ComplexLinear(prev, h, bias=True, init=init, gain=gain))

            if activation == "modrelu":
                acts.append(ComplexModReLU(h))
            elif activation == "cardioid":
                acts.append(ComplexCardioid())
            elif activation == "splitgelu":
                acts.append(ComplexSplitGELU())
            elif activation == "zrelu":
                acts.append(ComplexZReLU())
            elif activation == "crelu":
                acts.append(ComplexCReLU())
            elif activation == "cleakyrelu":
                acts.append(ComplexCLeakyReLU())
            elif activation == "celu":
                acts.append(ComplexCELU())
            elif activation == "softsign":
                acts.append(ComplexSoftsign())
            else:
                raise ValueError("Unsupported activation")
            prev = h

        self.hidden = nn.ModuleList(layers)
        self.acts = nn.ModuleList(acts)
        self.cdrop = ComplexDropout(dropout_val)

        self.output = nn.Linear(2 * prev, 1)

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lin, act in zip(self.hidden, self.acts):
            x = self.cdrop(act(lin(x)))

        xr = torch.view_as_real(x)
        xr = xr.movedim(-2, -1).reshape(x.shape[:-1] + (2 * x.shape[-1],))

        # no_top variant for integration with hybrid
        if not self.no_top:
            xr = self.output(xr)

        return xr
