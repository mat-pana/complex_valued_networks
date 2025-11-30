import math
import torch


def complex_kaiming_normal(t: torch.Tensor, fan_in: int, gain: float = 1.0) -> None:
    std_part = math.sqrt(1.0 / fan_in) / max(gain, 1e-12)
    wr = torch.empty_like(t.real).normal_(mean=0.0, std=std_part)
    wi = torch.empty_like(t.imag).normal_(mean=0.0, std=std_part)
    with torch.no_grad():
        t.copy_(torch.complex(wr, wi))


def complex_xavier_normal(
    t: torch.Tensor, fan_in: int, fan_out: int, gain: float = 1.0
) -> None:
    std_part = math.sqrt(1.0 / (fan_in + fan_out)) / max(gain, 1e-12)
    wr = torch.empty_like(t.real).normal_(mean=0.0, std=std_part)
    wi = torch.empty_like(t.imag).normal_(mean=0.0, std=std_part)
    with torch.no_grad():
        t.copy_(torch.complex(wr, wi))
