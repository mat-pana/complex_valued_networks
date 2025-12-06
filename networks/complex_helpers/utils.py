import math


def complex_gain(
    activation: str,
) -> float:
    act = activation.lower()
    if act == "cardioid":
        return math.sqrt(3.0 / 8.0)
    if act == "zrelu":
        return 0.5
    if act in {"crelu", "cxelu", "cgelu", "softsign"}:
        return 1.0 / math.sqrt(2.0)
    if act == "cleakyrelu":
        return math.sqrt((1.0 + 0.01**2) / 2.0)

    # Defaults.
    return 1.0
