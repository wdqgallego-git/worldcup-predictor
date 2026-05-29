"""World Cup 2026 tournament format helpers."""

from dataclasses import dataclass


N_TEAMS = 48
N_GROUPS = 12
TEAMS_PER_GROUP = 4
GROUP_LABELS = tuple(chr(ord("A") + idx) for idx in range(N_GROUPS))
GROUP_QUALIFIERS_PER_GROUP = 2
BEST_THIRD_PLACE_QUALIFIERS = 8
ROUND_OF_32_TEAMS = 32
FINALIST_MATCHES = 8


@dataclass(frozen=True)
class TournamentFormat:
    """Static description of the 2026 tournament format."""

    n_teams: int = N_TEAMS
    n_groups: int = N_GROUPS
    teams_per_group: int = TEAMS_PER_GROUP
    group_qualifiers_per_group: int = GROUP_QUALIFIERS_PER_GROUP
    best_third_place_qualifiers: int = BEST_THIRD_PLACE_QUALIFIERS
    round_of_32_teams: int = ROUND_OF_32_TEAMS
    finalist_matches: int = FINALIST_MATCHES

    @property
    def total_direct_group_qualifiers(self) -> int:
        return self.n_groups * self.group_qualifiers_per_group

    @property
    def total_knockout_qualifiers(self) -> int:
        return self.total_direct_group_qualifiers + self.best_third_place_qualifiers

    def validate(self) -> None:
        if self.n_teams != self.n_groups * self.teams_per_group:
            raise ValueError("2026 format must have 12 groups of 4 teams.")
        if self.total_knockout_qualifiers != self.round_of_32_teams:
            raise ValueError("2026 format must produce 32 knockout teams.")


def default_format() -> TournamentFormat:
    tournament_format = TournamentFormat()
    tournament_format.validate()
    return tournament_format

