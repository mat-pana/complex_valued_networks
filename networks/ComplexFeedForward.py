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
            "cgelu",
            "zrelu",
            "crelu",
            "cleakyrelu",
            "cxelu",
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
            # In each block we have linear layer initialised with given gain.
            gain = complex_gain(
                activation,
            )
            layers.append(ComplexLinear(prev, h, bias=True, init=init, gain=gain))

            # Followed by activation function.
            activation_layer = self.get_activation_layer(activation, h)
            acts.append(activation_layer)

            prev = h

        self.hidden = nn.ModuleList(layers)
        self.acts = nn.ModuleList(acts)

        # Just a single dropout layer as it does not need to be separate for all layers.
        self.cdrop = ComplexDropout(dropout_val)

        # Final dimensionality reduction.
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

    def get_activation_layer(self, activation: str, h: int) -> nn.Module:
        if activation == "modrelu":
            activation_layer = ModReLU(h)
        elif activation == "cardioid":
            activation_layer = Cardioid()
        elif activation == "cgelu":
            activation_layer = CGELU()
        elif activation == "zrelu":
            activation_layer = zReLU()
        elif activation == "crelu":
            activation_layer = CReLU()
        elif activation == "cleakyrelu":
            activation_layer = CLeakyReLU()
        elif activation == "cxelu":
            activation_layer = CxELU()
        elif activation == "softsign":
            activation_layer = CSoftsign()
        else:
            raise ValueError("Unsupported activation")
        
        return activation_layer
    