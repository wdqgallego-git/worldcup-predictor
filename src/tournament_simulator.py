"""Monte Carlo simulator for the 48-team World Cup 2026 format."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


N_TEAMS = 48
N_GROUPS = 12
TEAMS_PER_GROUP = 4
DIRECT_QUALIFIERS_PER_GROUP = 2
BEST_THIRD_PLACE_QUALIFIERS = 8
KNOCKOUT_TEAMS = 32
OFFICIAL_THIRD_PLACE_SCENARIOS = 495
DEFAULT_EXPECTED_GOALS = 1.2
ROUND_OF_32 = "Round of 32"
ROUND_OF_16 = "Round of 16"
QUARTER_FINAL = "Quarter-final"
SEMI_FINAL = "Semi-final"
THIRD_PLACE = "Match for third place"
FINAL = "Final"


ROUND_OF_32_TEMPLATE = [
    (73, "2A", "2B"),
    (74, "1E", "3A/B/C/D/F"),
    (75, "1F", "2C"),
    (76, "1C", "2F"),
    (77, "1I", "3C/D/F/G/H"),
    (78, "2E", "2I"),
    (79, "1A", "3C/E/F/H/I"),
    (80, "1L", "3E/H/I/J/K"),
    (81, "1D", "3B/E/F/I/J"),
    (82, "1G", "3A/E/H/I/J"),
    (83, "2K", "2L"),
    (84, "1H", "2J"),
    (85, "1B", "3E/F/G/I/J"),
    (86, "1J", "2H"),
    (87, "1K", "3D/E/I/J/L"),
    (88, "2D", "2G"),
]

KNOCKOUT_TEMPLATE = [
    (89, ROUND_OF_16, "W74", "W77"),
    (90, ROUND_OF_16, "W73", "W75"),
    (91, ROUND_OF_16, "W76", "W78"),
    (92, ROUND_OF_16, "W79", "W80"),
    (93, ROUND_OF_16, "W83", "W84"),
    (94, ROUND_OF_16, "W81", "W82"),
    (95, ROUND_OF_16, "W86", "W88"),
    (96, ROUND_OF_16, "W85", "W87"),
    (97, QUARTER_FINAL, "W89", "W90"),
    (98, QUARTER_FINAL, "W93", "W94"),
    (99, QUARTER_FINAL, "W91", "W92"),
    (100, QUARTER_FINAL, "W95", "W96"),
    (101, SEMI_FINAL, "W97", "W98"),
    (102, SEMI_FINAL, "W99", "W100"),
    (103, THIRD_PLACE, "L101", "L102"),
    (104, FINAL, "W101", "W102"),
]


def validate_2026_format() -> None:
    """Validate the 48-team World Cup 2026 format assumptions."""
    if N_TEAMS != N_GROUPS * TEAMS_PER_GROUP:
        raise ValueError("World Cup 2026 must use 48 teams in 12 groups of 4.")
    qualifiers = N_GROUPS * DIRECT_QUALIFIERS_PER_GROUP + BEST_THIRD_PLACE_QUALIFIERS
    if qualifiers != KNOCKOUT_TEAMS:
        raise ValueError("World Cup 2026 must produce 32 knockout teams.")


def simulate_group_match(team_a: str, team_b: str, lambda_a: float, lambda_b: float, rng: np.random.Generator) -> dict[str, object]:
    """Simulate one regulation-time group match."""
    return {
        "team_a": team_a,
        "team_b": team_b,
        "goals_a": int(rng.poisson(max(0.0, lambda_a))),
        "goals_b": int(rng.poisson(max(0.0, lambda_b))),
    }


def empty_group_row(team: str) -> dict[str, object]:
    """Return an empty group-table row."""
    return {"team": team, "played": 0, "points": 0, "goals_for": 0, "goals_against": 0, "goal_difference": 0}


def update_group_table(group_table: dict[str, dict[str, object]], result: dict[str, object]) -> None:
    """Apply one simulated group result to a group table."""
    team_a, team_b = result["team_a"], result["team_b"]
    goals_a, goals_b = result["goals_a"], result["goals_b"]
    for team in (team_a, team_b):
        group_table.setdefault(team, empty_group_row(team))
        group_table[team]["played"] += 1
    group_table[team_a]["goals_for"] += goals_a
    group_table[team_a]["goals_against"] += goals_b
    group_table[team_b]["goals_for"] += goals_b
    group_table[team_b]["goals_against"] += goals_a
    if goals_a > goals_b:
        group_table[team_a]["points"] += 3
    elif goals_b > goals_a:
        group_table[team_b]["points"] += 3
    else:
        group_table[team_a]["points"] += 1
        group_table[team_b]["points"] += 1
    for team in (team_a, team_b):
        group_table[team]["goal_difference"] = group_table[team]["goals_for"] - group_table[team]["goals_against"]


def rank_group_table(group_table, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Rank a group by points, goal difference, goals scored, then random fallback."""
    rng = rng or np.random.default_rng()
    table = pd.DataFrame(list(group_table.values()) if isinstance(group_table, dict) else group_table).copy()
    table["_fallback"] = rng.random(len(table))
    return table.sort_values(
        ["points", "goal_difference", "goals_for", "_fallback"],
        ascending=[False, False, False, False],
        kind="stable",
    ).drop(columns="_fallback").reset_index(drop=True)


def rank_third_placed_teams(third_place_rows, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Rank third-place teams by points, goal difference, goals scored, then random fallback."""
    rng = rng or np.random.default_rng()
    rows = pd.DataFrame(third_place_rows).copy()
    rows["_fallback"] = rng.random(len(rows))
    return rows.sort_values(
        ["points", "goal_difference", "goals_for", "_fallback"],
        ascending=[False, False, False, False],
        kind="stable",
    ).drop(columns="_fallback").reset_index(drop=True)


def build_prediction_lookup(match_predictions) -> Callable[[str, str], tuple[float, float]]:
    """Return a transparent expected-goals resolver for arbitrary matchups."""
    if callable(match_predictions):
        return match_predictions
    predictions = pd.DataFrame(match_predictions).copy()
    required = {"team_a", "team_b", "expected_goals_a", "expected_goals_b"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"match_predictions is missing required columns: {missing}")

    direct = {}
    team_values: dict[str, list[float]] = defaultdict(list)
    for row in predictions.dropna(subset=["team_a", "team_b"]).itertuples(index=False):
        lambda_a = float(row.expected_goals_a)
        lambda_b = float(row.expected_goals_b)
        direct[(row.team_a, row.team_b)] = (lambda_a, lambda_b)
        team_values[row.team_a].append(lambda_a)
        team_values[row.team_b].append(lambda_b)
    team_average = {team: float(np.mean(values)) for team, values in team_values.items()}

    def resolve(team_a: str, team_b: str) -> tuple[float, float]:
        if (team_a, team_b) in direct:
            return direct[(team_a, team_b)]
        if (team_b, team_a) in direct:
            reverse_b, reverse_a = direct[(team_b, team_a)]
            return reverse_a, reverse_b
        return team_average.get(team_a, DEFAULT_EXPECTED_GOALS), team_average.get(team_b, DEFAULT_EXPECTED_GOALS)

    return resolve


def simulate_group_stage(fixtures: pd.DataFrame, match_predictions, rng: np.random.Generator) -> dict[str, object]:
    """Simulate all 72 group matches and return ranked group tables."""
    group_fixtures = fixtures[fixtures["stage"].eq("group_stage")].copy()
    if len(group_fixtures) != 72:
        raise ValueError(f"World Cup 2026 group stage must contain 72 matches. Found {len(group_fixtures)}.")
    resolve_xg = build_prediction_lookup(match_predictions)
    tables: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    results = []
    for match in group_fixtures.itertuples(index=False):
        lambda_a, lambda_b = resolve_xg(match.team_a, match.team_b)
        result = simulate_group_match(match.team_a, match.team_b, lambda_a, lambda_b, rng)
        result.update({"match_id": match.match_id, "group": match.group, "round": match.round})
        update_group_table(tables[match.group], result)
        results.append(result)
    ranked_tables = {group: rank_group_table(table, rng) for group, table in tables.items()}
    return {"group_tables": ranked_tables, "match_results": results}


def select_round_of_32_teams(group_tables, rng: np.random.Generator | None = None) -> dict[str, object]:
    """Select 24 direct qualifiers and the 8 best third-place teams."""
    rng = rng or np.random.default_rng()
    direct_qualifiers = {}
    third_place_rows = []
    for group, table in sorted(group_tables.items()):
        ranked = rank_group_table(table, rng) if not isinstance(table, pd.DataFrame) else table.reset_index(drop=True)
        if len(ranked) != TEAMS_PER_GROUP:
            raise ValueError(f"Group {group} must contain {TEAMS_PER_GROUP} teams.")
        group_letter = str(group).replace("Group ", "")
        direct_qualifiers[f"1{group_letter}"] = ranked.iloc[0]["team"]
        direct_qualifiers[f"2{group_letter}"] = ranked.iloc[1]["team"]
        third = ranked.iloc[2].to_dict()
        third["group"] = group_letter
        third_place_rows.append(third)
    ranked_thirds = rank_third_placed_teams(third_place_rows, rng)
    qualified_thirds = ranked_thirds.head(BEST_THIRD_PLACE_QUALIFIERS).reset_index(drop=True)
    return {"direct_qualifiers": direct_qualifiers, "qualified_third_teams": qualified_thirds}


def validate_third_place_assignment_matrix(third_place_assignment_matrix) -> pd.DataFrame:
    """Validate the complete official 495-scenario Annexe C assignment matrix."""
    matrix = pd.DataFrame(third_place_assignment_matrix).copy()
    required = {"qualified_third_groups", *[f"slot_{slot}" for slot in range(1, 9)]}
    missing = sorted(required - set(matrix.columns))
    if missing:
        raise ValueError(f"Third-place assignment matrix is missing columns: {missing}")
    if len(matrix) != OFFICIAL_THIRD_PLACE_SCENARIOS:
        raise ValueError(
            f"Official third-place assignment matrix must contain {OFFICIAL_THIRD_PLACE_SCENARIOS} scenarios. "
            f"Found {len(matrix)}."
        )
    expected_scenarios = {"-".join(groups) for groups in combinations("ABCDEFGHIJKL", 8)}
    matrix["qualified_third_groups"] = matrix["qualified_third_groups"].astype(str).str.replace("/", "-")
    actual_scenarios = set(matrix["qualified_third_groups"])
    if actual_scenarios != expected_scenarios:
        raise ValueError("Third-place assignment matrix does not contain every official eight-group scenario exactly once.")
    if matrix["qualified_third_groups"].duplicated().any():
        raise ValueError("Third-place assignment matrix contains duplicate scenarios.")
    for row in matrix.itertuples(index=False):
        scenario_groups = set(row.qualified_third_groups.split("-"))
        assigned_groups = {str(getattr(row, f"slot_{slot}")) for slot in range(1, 9)}
        if assigned_groups != scenario_groups:
            raise ValueError(f"Third-place matrix scenario {row.qualified_third_groups} has invalid slot assignments.")
    return matrix


def assign_third_placed_teams_to_bracket(
    qualified_third_teams,
    third_place_assignment_matrix=None,
    allow_development_fallback: bool = False,
) -> dict[str, str]:
    """Assign eight qualified third-place teams using the official matrix or a development fallback."""
    qualified = pd.DataFrame(qualified_third_teams).copy()
    if len(qualified) != BEST_THIRD_PLACE_QUALIFIERS:
        raise ValueError("Exactly 8 qualified third-place teams are required.")
    team_by_group = dict(zip(qualified["group"].astype(str), qualified["team"]))
    matrix = pd.DataFrame() if third_place_assignment_matrix is None else pd.DataFrame(third_place_assignment_matrix).copy()
    scenario = "-".join(sorted(team_by_group))
    if not matrix.empty:
        matrix = validate_third_place_assignment_matrix(matrix)
        matching = matrix[matrix["qualified_third_groups"].astype(str).str.replace("/", "-") == scenario]
        if not matching.empty:
            assignments = {}
            for slot_number in range(1, 9):
                group = str(matching.iloc[0][f"slot_{slot_number}"])
                assignments[f"slot_{slot_number}"] = team_by_group[group]
            return assignments

    if not allow_development_fallback:
        raise ValueError(
            f"Official third-place assignment is unavailable for scenario {scenario}. "
            "Development fallback is disabled."
        )

    # Development-only deterministic fallback: fill each official eligible slot with
    # the best still-unassigned qualified third-place team allowed by that slot.
    third_slots = [slot for _, _, slot in ROUND_OF_32_TEMPLATE if str(slot).startswith("3")]
    ranked_groups = list(qualified["group"].astype(str))

    def find_assignment(slot_index: int, remaining_groups: list[str]) -> list[str] | None:
        if slot_index == len(third_slots):
            return []
        eligible_groups = str(third_slots[slot_index])[1:].split("/")
        for group in ranked_groups:
            if group not in remaining_groups or group not in eligible_groups:
                continue
            remaining = [candidate for candidate in remaining_groups if candidate != group]
            later_assignments = find_assignment(slot_index + 1, remaining)
            if later_assignments is not None:
                return [group, *later_assignments]
        return None

    assigned_groups = find_assignment(0, ranked_groups)
    if assigned_groups is None:
        raise ValueError("Development fallback could not find a valid third-place bracket assignment.")
    return {
        f"slot_{slot_number}": team_by_group[group]
        for slot_number, group in enumerate(assigned_groups, start=1)
    }


def build_round_of_32_bracket(
    group_results,
    third_place_assignment_matrix=None,
    allow_development_fallback: bool = False,
) -> list[dict[str, object]]:
    """Build the official 16-match Round-of-32 bracket."""
    selection = select_round_of_32_teams(group_results)
    direct = selection["direct_qualifiers"]
    third_assignments = assign_third_placed_teams_to_bracket(
        selection["qualified_third_teams"],
        third_place_assignment_matrix,
        allow_development_fallback=allow_development_fallback,
    )
    third_index = 0
    bracket = []
    for match_id, team_a_token, team_b_token in ROUND_OF_32_TEMPLATE:
        team_a = direct.get(team_a_token, team_a_token)
        if str(team_b_token).startswith("3"):
            third_index += 1
            team_b = third_assignments[f"slot_{third_index}"]
        else:
            team_b = direct.get(team_b_token, team_b_token)
        bracket.append({"match_id": match_id, "round": ROUND_OF_32, "team_a": team_a, "team_b": team_b})
    return bracket


def resolve_knockout_tie_stochastically(team_a: str, team_b: str, lambda_a: float, lambda_b: float, rng: np.random.Generator) -> str:
    """Resolve a knockout tie stochastically with a small expected-goals edge."""
    probability_team_a = float(np.clip(0.50 + 0.04 * (lambda_a - lambda_b), 0.42, 0.58))
    return team_a if rng.random() < probability_team_a else team_b


def simulate_knockout_match(team_a: str, team_b: str, lambda_a: float, lambda_b: float, rng: np.random.Generator) -> dict[str, object]:
    """Simulate a knockout match and resolve regulation ties stochastically."""
    result = simulate_group_match(team_a, team_b, lambda_a, lambda_b, rng)
    if result["goals_a"] > result["goals_b"]:
        winner, loser = team_a, team_b
    elif result["goals_b"] > result["goals_a"]:
        winner, loser = team_b, team_a
    else:
        winner = resolve_knockout_tie_stochastically(team_a, team_b, lambda_a, lambda_b, rng)
        loser = team_b if winner == team_a else team_a
    return {**result, "winner": winner, "loser": loser}


def update_team_stats(team_stats, result: dict[str, object]) -> None:
    """Update cumulative tournament stats for one simulated match."""
    for team, goals_for, goals_against in (
        (result["team_a"], result["goals_a"], result["goals_b"]),
        (result["team_b"], result["goals_b"], result["goals_a"]),
    ):
        team_stats[team]["matches_played"] += 1
        team_stats[team]["goals_for"] += goals_for
        team_stats[team]["goals_against"] += goals_against
        team_stats[team]["clean_sheets"] += int(goals_against == 0)


def run_tournament_simulation(
    fixtures,
    match_predictions,
    third_place_assignment_matrix=None,
    rng=None,
    allow_development_fallback: bool = False,
    knockout_lambda_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Run one complete World Cup 2026 simulation and return team paths."""
    validate_2026_format()
    rng = rng or np.random.default_rng()
    fixtures = pd.DataFrame(fixtures).copy()
    resolve_xg = build_prediction_lookup(match_predictions)
    team_stats = defaultdict(lambda: {
        "matches_played": 0, "goals_for": 0, "goals_against": 0, "clean_sheets": 0,
        "reached_knockouts": False, "reached_round_of_16": False, "reached_quarterfinal": False,
        "reached_semifinal": False, "reached_final": False, "champion": False, "runner_up": False,
    })

    group_stage = simulate_group_stage(fixtures, match_predictions, rng)
    for result in group_stage["match_results"]:
        update_team_stats(team_stats, result)
    bracket = build_round_of_32_bracket(
        group_stage["group_tables"],
        third_place_assignment_matrix,
        allow_development_fallback=allow_development_fallback,
    )
    knockout_results = {}
    for match in bracket:
        team_stats[match["team_a"]]["reached_knockouts"] = True
        team_stats[match["team_b"]]["reached_knockouts"] = True
    all_knockout_matches = bracket + [
        {"match_id": match_id, "round": round_name, "team_a": team_a, "team_b": team_b}
        for match_id, round_name, team_a, team_b in KNOCKOUT_TEMPLATE
    ]

    stage_flags = {
        ROUND_OF_16: "reached_round_of_16",
        QUARTER_FINAL: "reached_quarterfinal",
        SEMI_FINAL: "reached_semifinal",
        FINAL: "reached_final",
    }
    for match in all_knockout_matches:
        def resolve_token(token):
            if str(token).startswith("W"):
                return knockout_results[int(str(token)[1:])]["winner"]
            if str(token).startswith("L"):
                return knockout_results[int(str(token)[1:])]["loser"]
            return token

        team_a, team_b = resolve_token(match["team_a"]), resolve_token(match["team_b"])
        if match["round"] in stage_flags:
            team_stats[team_a][stage_flags[match["round"]]] = True
            team_stats[team_b][stage_flags[match["round"]]] = True
        lambda_a, lambda_b = resolve_xg(team_a, team_b)
        lambda_a *= knockout_lambda_multiplier
        lambda_b *= knockout_lambda_multiplier
        result = simulate_knockout_match(team_a, team_b, lambda_a, lambda_b, rng)
        result.update({"match_id": match["match_id"], "round": match["round"]})
        knockout_results[match["match_id"]] = result
        update_team_stats(team_stats, result)

    champion = knockout_results[104]["winner"]
    runner_up = knockout_results[104]["loser"]
    third_place = knockout_results[103]["winner"]
    fourth_place = knockout_results[103]["loser"]
    team_stats[champion]["champion"] = True
    team_stats[runner_up]["runner_up"] = True
    finish_rank = {champion: 1, runner_up: 2, third_place: 3, fourth_place: 4}
    rows = []
    for team, stats in team_stats.items():
        rows.append({"team": team, **stats, "finish_rank": finish_rank.get(team, 0)})
    return pd.DataFrame(rows)[
        [
            "team", "matches_played", "goals_for", "goals_against", "clean_sheets", "finish_rank",
            "reached_knockouts", "reached_round_of_16", "reached_quarterfinal", "reached_semifinal",
            "reached_final", "champion", "runner_up",
        ]
    ]


def run_monte_carlo_tournament(
    fixtures,
    match_predictions,
    n_simulations: int = 1000,
    third_place_assignment_matrix=None,
    random_seed: int = 2026,
    output_dir: str | Path = "outputs",
    allow_development_fallback: bool = False,
    knockout_lambda_multiplier: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """Run Monte Carlo simulations and write team-path probability outputs."""
    if n_simulations < 1:
        raise ValueError("n_simulations must be positive.")
    rng = np.random.default_rng(random_seed)
    paths = []
    for simulation_id in range(1, n_simulations + 1):
        simulation = run_tournament_simulation(
            fixtures,
            match_predictions,
            third_place_assignment_matrix,
            rng,
            allow_development_fallback=allow_development_fallback,
            knockout_lambda_multiplier=knockout_lambda_multiplier,
        )
        simulation.insert(0, "simulation_id", simulation_id)
        paths.append(simulation)
    team_paths = pd.concat(paths, ignore_index=True)
    probability_columns = [
        "reached_knockouts", "reached_round_of_16", "reached_quarterfinal",
        "reached_semifinal", "reached_final", "champion", "runner_up",
    ]
    probabilities = team_paths.groupby("team", as_index=False)[probability_columns].mean()
    probabilities = probabilities.rename(columns={column: f"{column}_probability" for column in probability_columns})
    champion_probabilities = probabilities[["team", "champion_probability"]].sort_values("champion_probability", ascending=False)
    runner_up_probabilities = probabilities[["team", "runner_up_probability"]].sort_values("runner_up_probability", ascending=False)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    team_paths.to_csv(output_path / "team_path_simulations.csv", index=False)
    champion_probabilities.to_csv(output_path / "champion_probabilities.csv", index=False)
    runner_up_probabilities.to_csv(output_path / "runner_up_probabilities.csv", index=False)
    probabilities.to_csv(output_path / "team_tournament_probabilities.csv", index=False)
    return {
        "team_path_simulations": team_paths,
        "champion_probabilities": champion_probabilities,
        "runner_up_probabilities": runner_up_probabilities,
        "team_tournament_probabilities": probabilities,
    }
