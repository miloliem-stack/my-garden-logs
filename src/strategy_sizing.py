"""Sizing utilities: fractional Kelly placeholder."""


def fractional_kelly(p: float, q: float, k: float = 0.1) -> float:
    if p <= q:
        return 0.0
    f_star = (p - q) / (1 - q)
    return max(0.0, k * f_star)
