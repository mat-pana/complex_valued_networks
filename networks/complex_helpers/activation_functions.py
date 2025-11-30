import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexModReLU(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        b_init = 0.0
        self.bias = nn.Parameter(
            torch.full((num_features,), b_init, dtype=torch.float32)
        )
        self.eps = 1e-8

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        r = torch.abs(z)
        scale = F.relu(r + self.bias) / (r + self.eps)
        return z * scale


class ComplexCardioid(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        ang = torch.angle(z)
        gain = 0.5 * (1.0 + torch.cos(ang))
        return z * gain


class ComplexSplitGELU(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.complex(F.gelu(z.real), F.gelu(z.imag))


class ComplexZReLU(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mask = (z.real > 0) & (z.imag > 0)
        return z * mask


class ComplexCReLU(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.complex(F.relu(z.real), F.relu(z.imag))


class ComplexCLeakyReLU(nn.Module):
    def __init__(self):
        super().__init__()
        negative_slope = 0.01
        self.negative_slope = negative_slope

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.complex(
            F.leaky_relu(z.real, self.negative_slope),
            F.leaky_relu(z.imag, self.negative_slope),
        )


class ComplexCELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 1.0

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.complex(F.elu(z.real, self.alpha), F.elu(z.imag, self.alpha))


class ComplexSoftsign(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z / (1 + torch.abs(z))
