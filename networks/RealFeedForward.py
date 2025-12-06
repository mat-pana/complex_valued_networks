import torch
import torch.nn as nn
from typing import List


class RealFeedForward(nn.Module):

    input_type = "single_real"

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        dropout_val: float = 0,
        device: str = "cuda:0",
    ) -> None:
        super(RealFeedForward, self).__init__()
        self.output_size = output_size

        self.device = device

        self.hidden_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        prev_size = input_size
        for layer_size in hidden_layers:
            # Conventional stack of dropout followed by linear layer.
            self.dropouts.append(nn.Dropout(dropout_val))
            self.hidden_layers.append(nn.Linear(prev_size, layer_size))
            prev_size = layer_size

        self.output_layer = nn.Linear(prev_size, output_size)
        self.activation = nn.GELU()

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for hidden_layer, dropout in zip(self.hidden_layers, self.dropouts):
            x = dropout(x)
            x = self.activation(hidden_layer(x))
        return self.output_layer(x)
