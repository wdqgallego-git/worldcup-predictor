"""Poisson score-model utilities."""

import math


def poisson_pmf(k: int, rate: float) -> float:
    """Return the Poisson probability of k goals at the given rate."""
    if k < 0:
        return 0.0
    if rate < 0:
        raise ValueError("rate must be non-negative")
    return math.exp(-rate) * rate**k / math.factorial(k)

