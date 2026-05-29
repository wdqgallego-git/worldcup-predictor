"""Shared data schemas for the project skeleton."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Team:
    team_id: str
    name: str
    group: str | None = None


@dataclass(frozen=True)
class Match:
    match_id: str
    team_a: str
    team_b: str
    stage: str
    group: str | None = None


@dataclass(frozen=True)
class ScorePrediction:
    match_id: str
    team_a: str
    team_b: str
    goals_a: float
    goals_b: float

    @property
    def goal_difference(self) -> float:
        return self.goals_a - self.goals_b

    @property
    def result(self) -> str:
        if self.goals_a > self.goals_b:
            return "team_a_win"
        if self.goals_b > self.goals_a:
            return "team_b_win"
        return "draw"

