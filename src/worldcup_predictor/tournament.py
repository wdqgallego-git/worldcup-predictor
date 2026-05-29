"""World Cup 2026 tournament simulation skeleton."""

from .config import SimulationConfig
from .format_2026 import TournamentFormat, default_format


def simulate_tournament(
    config: SimulationConfig | None = None,
    tournament_format: TournamentFormat | None = None,
):
    """Simulate the 48-team World Cup 2026 format."""
    config = config or SimulationConfig()
    tournament_format = tournament_format or default_format()
    config.validate()
    tournament_format.validate()
    raise NotImplementedError("Tournament simulation will be implemented in a later step.")

