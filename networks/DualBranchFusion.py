import torch
import torch.nn as nn
from typing import List, Literal


class FusionHead(nn.Module):

    def __init__(
        self,
        hidden_layers: List[int] = [20, 10],
        dropout: float = 0.0,
        activation: Literal["relu", "gelu", "silu"] = "gelu",
        out_features: int = 1,
    ) -> None:
        super().__init__()
        acts = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        layers: List[nn.Module] = []

        # Lazy construction added to accommodate different different output size
        # of preceding preceding layers.
        prev_is_lazy = True
        for h in hidden_layers:
            if prev_is_lazy:
                layers.append(nn.LazyLinear(h))
                prev_is_lazy = False
            else:
                layers.append(nn.Linear(prev_h, h))
            layers.append(acts[activation])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_h = h

        if prev_is_lazy:
            layers.append(nn.LazyLinear(out_features))
        else:
            layers.append(nn.Linear(prev_h, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualBranchFusion(nn.Module):

    input_type = "dual"

    def __init__(
        self,
        rnn_branch: nn.Module,
        cv_branch: nn.Module,
        head_hidden: List[int] = [20, 10],
        head_dropout: float = 0.1,
        head_activation: Literal["relu", "gelu", "silu"] = "gelu",
        device: str = "cuda:0",
    ) -> None:
        super().__init__()
        self.rnn_branch = rnn_branch
        self.cv_branch = cv_branch

        self.head = FusionHead(
            hidden_layers=head_hidden,
            dropout=head_dropout,
            activation=head_activation,
            out_features=1,
        )
        self.device = device
        self.to(self.device)

    def forward(
        self,
        x_time: torch.Tensor,
        x_freq: torch.Tensor,
    ) -> torch.Tensor:
        e_time = self.rnn_branch(x_time)
        e_freq = self.cv_branch(x_freq)

        fused = torch.cat([e_time, e_freq], dim=-1)
        logits = self.head(fused)
        return logits
