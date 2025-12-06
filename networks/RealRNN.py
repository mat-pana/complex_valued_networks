import torch
import torch.nn as nn
from typing import List, Optional


class RealRNN(nn.Module):

    input_type = "single_recurrent"

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        cell: str = "gru",
        dropout_val: float = 0.0,
        bidirectional: bool = False,
        no_top: bool = False,
        pooling: str = "last",
        head: Optional[List[int]] = None,
        device: str = "cuda:0",
    ) -> None:
        super().__init__()

        self.output_size = output_size
        self.pooling = pooling
        self.bidirectional = bidirectional
        self.no_top = no_top
        self.device = device

        self.rnn_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        prev_size = input_size
        self.cell_type = cell.lower()

        for h in hidden_layers:
            # Stack of recurrent layers followed by dropout.
            self.rnn_layers.append(
                self._make_rnn(prev_size, h, self.cell_type, bidirectional)
            )
            self.dropouts.append(nn.Dropout(dropout_val))
            prev_size = h * (2 if bidirectional else 1)

        last_size = prev_size * 2 if bidirectional else prev_size

        self.head = None
        if head is not None and len(head) > 0:
            layers = []
            in_dim = last_size
            for h_dim in head:
                layers.append(nn.Linear(in_dim, h_dim))
                layers.append(nn.ReLU())
                in_dim = h_dim
            self.head = nn.Sequential(*layers)
            last_size = in_dim

        self.output_layer = nn.Linear(last_size, output_size)

        self.to(self.device)

    def _make_rnn(
        self, in_size: int, hidden_size: int, cell: str, bidirectional: bool
    ) -> nn.Module:
        kwargs = dict(
            input_size=in_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
        )
        if cell == "gru":
            return nn.GRU(**kwargs)
        elif cell == "lstm":
            return nn.LSTM(**kwargs)
        elif cell == "rnn":
            return nn.RNN(nonlinearity="tanh", **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x

        for rnn, do in zip(self.rnn_layers, self.dropouts):
            out, _ = rnn(out)
            out = do(out)

        # Only last hidden state for unidierctional and first and last for
        # bidirectional layers.
        if not self.bidirectional:
            out = out[:, -1, :]
        else:
            out = out[:, [0, -1], :].flatten(1)

        # no_top variant for integration with hybrid
        if not self.no_top:
            if self.head is not None:
                out = self.head(out)
            out = self.output_layer(out)

        return out
