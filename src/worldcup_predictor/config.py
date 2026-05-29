"""Project constants and runtime settings."""

from dataclasses import dataclass


AWARD_POINTS = {
    "champion": 25,
    "runner_up": 15,
    "top_scorer": 35,
    "mvp": 10,
    "golden_glove": 10,
}

DEV_SIMULATIONS = 1_000
FINAL_SIMULATIONS = 20_000
STRATEGIC_EV_FLOOR = 0.85
STRATEGIC_EV_TARGET = 0.90


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration shared by development and final simulation runs."""

    n_simulations: int = DEV_SIMULATIONS
    random_seed: int = 2026
    use_limited_calibration: bool = True

    def validate(self) -> None:
        if self.n_simulations < 1:
            raise ValueError("n_simulations must be positive.")
        if self.n_simulations > FINAL_SIMULATIONS:
            raise ValueError(f"n_simulations must be <= {FINAL_SIMULATIONS}.")

