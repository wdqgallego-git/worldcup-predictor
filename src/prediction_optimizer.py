"""Optimize prediction outputs for expected points."""


def choose_highest_ev(options, value_key: str = "expected_points"):
    """Return the option with the highest expected-points value."""
    if not options:
        raise ValueError("options must not be empty")
    return max(options, key=lambda option: option[value_key])

