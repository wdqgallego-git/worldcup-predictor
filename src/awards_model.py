"""Transparent direct player simulations for World Cup awards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TOP_SCORER_DISPERSION = 2.0
DEFAULT_PENALTY_RATE_PER_MATCH = 0.12
DEFAULT_STAR_PRIOR_TEMPERATURE = 1.0
DEFAULT_TEAM_SUCCESS_CAP = 1.0
DEFAULT_TEAM_ATTACK_STRENGTH = 1.35

ROLE_FACTORS = {"FW": 1.15, "MF": 0.90, "DF": 0.45, "GK": 0.02}
POSITION_BONUSES = {"FW": 0.10, "MF": 0.08, "DF": 0.04, "GK": 0.00}
TEAM_FINISH_SCORES = {0: 0.00, 4: 0.65, 3: 0.75, 2: 0.90, 1: 1.00}


def validate_columns(df: pd.DataFrame, required_columns, dataset_name: str) -> None:
    """Raise a clear error if an awards input is missing columns."""
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )


def numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Return a numeric series with a stable default."""
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def negative_binomial_draw(mean: float, dispersion: float, rng: np.random.Generator) -> int:
    """Draw from a Negative Binomial distribution parameterized by mean."""
    mean = max(0.0, float(mean))
    if mean == 0:
        return 0
    if dispersion <= 0:
        raise ValueError("top_scorer_dispersion must be positive.")
    probability = dispersion / (dispersion + mean)
    return int(rng.negative_binomial(dispersion, probability))


def simulate_team_penalties(
    team_paths: pd.DataFrame,
    rng: np.random.Generator,
    penalty_rate_per_match: float = DEFAULT_PENALTY_RATE_PER_MATCH,
) -> pd.DataFrame:
    """Simulate awarded penalties separately from open-play goals."""
    if penalty_rate_per_match < 0:
        raise ValueError("penalty_rate_per_match must be non-negative.")
    validate_columns(team_paths, ["simulation_id", "team", "matches_played"], "team paths")
    penalties = team_paths[["simulation_id", "team", "matches_played"]].copy()
    penalties["team_penalties"] = rng.poisson(
        penalties["matches_played"].to_numpy(dtype=float) * penalty_rate_per_match
    )
    return penalties


def player_goal_shares(player_features: pd.DataFrame) -> pd.Series:
    """Return transparent within-team non-penalty scoring shares."""
    xg_rate = numeric_series(player_features, "xg_per_90", 0.0).clip(lower=0)
    goal_rate = numeric_series(player_features, "goals_per_90", 0.0).clip(lower=0)
    scoring_signal = 0.70 * xg_rate + 0.30 * goal_rate
    team_total = scoring_signal.groupby(player_features["team"]).transform("sum")
    team_count = player_features.groupby("team")["player"].transform("count").clip(lower=1)
    return np.where(team_total > 0, scoring_signal / team_total, 1.0 / team_count)


def allocate_penalty_goals(
    player_outputs: pd.DataFrame,
    team_penalties: pd.DataFrame,
    penalty_takers: pd.DataFrame | None,
    rng: np.random.Generator,
) -> pd.Series:
    """Allocate separately simulated team penalties to eligible takers."""
    penalty_goals = pd.Series(0, index=player_outputs.index, dtype=int)
    if penalty_takers is None or penalty_takers.empty:
        return penalty_goals
    validate_columns(penalty_takers, ["team", "player", "penalty_share"], "penalty takers")
    takers = penalty_takers.copy()
    takers["penalty_share"] = numeric_series(takers, "penalty_share", 0.0).clip(0, 1)
    takers["penalty_conversion_rate"] = numeric_series(takers, "penalty_conversion_rate", 0.78).clip(0, 1)
    penalties_by_team = team_penalties.set_index(["simulation_id", "team"])["team_penalties"].to_dict()

    for (simulation_id, team), candidate_rows in player_outputs.groupby(["simulation_id", "team"]):
        team_takers = takers[takers["team"].eq(team)]
        if team_takers.empty:
            continue
        available = candidate_rows[["index", "team", "player"]].merge(
            team_takers[["team", "player", "penalty_share", "penalty_conversion_rate"]],
            on=["team", "player"],
            how="inner",
        )
        if available.empty:
            continue
        shares = available["penalty_share"].to_numpy(dtype=float)
        shares = shares / shares.sum() if shares.sum() > 0 else np.full(len(shares), 1 / len(shares))
        for _ in range(int(penalties_by_team.get((simulation_id, team), 0))):
            selected = int(rng.choice(len(available), p=shares))
            if rng.random() < float(available.iloc[selected]["penalty_conversion_rate"]):
                penalty_goals.loc[int(available.iloc[selected]["index"])] += 1
    return penalty_goals


def simulate_player_outputs(
    team_paths: pd.DataFrame,
    player_features: pd.DataFrame,
    penalty_takers: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
    top_scorer_dispersion: float = DEFAULT_TOP_SCORER_DISPERSION,
    penalty_rate_per_match: float = DEFAULT_PENALTY_RATE_PER_MATCH,
) -> pd.DataFrame:
    """Directly simulate player goals, penalties, assists, and minutes."""
    rng = rng or np.random.default_rng()
    validate_columns(
        team_paths,
        ["simulation_id", "team", "matches_played", "finish_rank"],
        "team paths",
    )
    validate_columns(
        player_features,
        ["team", "player", "position", "probability_in_squad", "projected_minutes_share"],
        "player features",
    )
    players = player_features.copy()
    players["player_goal_share"] = player_goal_shares(players)
    players["role_factor"] = players["position"].map(ROLE_FACTORS).fillna(0.75)
    players["minutes_share"] = numeric_series(
        players,
        "adjusted_projected_minutes_share",
        0.0,
    )
    if "adjusted_projected_minutes_share" not in players.columns:
        players["minutes_share"] = numeric_series(players, "projected_minutes_share", 0.65)
    players["probability_in_squad"] = numeric_series(players, "probability_in_squad", 0.75).clip(0, 1)
    players["assists_per_90"] = numeric_series(players, "assists_per_90", 0.0).clip(lower=0)
    players["attack_adjustment"] = numeric_series(players, "attack_adjustment", 0.0)
    players["team_attack_strength"] = numeric_series(
        players,
        "team_attack_strength",
        DEFAULT_TEAM_ATTACK_STRENGTH,
    ).clip(0.35, 3.00)

    outputs = team_paths.merge(players, on="team", how="inner")
    if outputs.empty:
        raise ValueError("No player features matched the simulated team paths.")
    outputs["included_in_squad"] = rng.random(len(outputs)) < outputs["probability_in_squad"]
    outputs["simulated_minutes"] = (
        outputs["matches_played"] * 90 * outputs["minutes_share"] * outputs["included_in_squad"]
    ).round().astype(int)
    outputs["open_play_goal_mean"] = (
        outputs["team_attack_strength"]
        * outputs["player_goal_share"]
        * (outputs["simulated_minutes"] / 90)
        * outputs["role_factor"]
        * (1 + outputs["attack_adjustment"]).clip(lower=0)
    )
    outputs["open_play_goals"] = [
        negative_binomial_draw(mean, top_scorer_dispersion, rng)
        for mean in outputs["open_play_goal_mean"]
    ]
    penalties = simulate_team_penalties(team_paths, rng, penalty_rate_per_match)
    outputs = outputs.reset_index(drop=True).reset_index()
    outputs["penalty_goals"] = allocate_penalty_goals(outputs, penalties, penalty_takers, rng)
    outputs["assist_mean"] = outputs["assists_per_90"] * outputs["simulated_minutes"] / 90
    outputs["assists"] = rng.poisson(outputs["assist_mean"].clip(lower=0))
    outputs["total_goals"] = outputs["open_play_goals"] + outputs["penalty_goals"]
    outputs["knockout_matches"] = (outputs["matches_played"] - 3).clip(lower=0)
    outputs["knockout_impact_score"] = outputs["total_goals"] * outputs["knockout_matches"] / outputs["matches_played"].clip(lower=1)
    return outputs.drop(columns="index")


def rank_top_scorer(player_outputs: pd.DataFrame) -> pd.DataFrame:
    """Rank Golden Boot candidates by goals, assists, then fewer minutes."""
    validate_columns(player_outputs, ["total_goals", "assists", "simulated_minutes"], "player outputs")
    return player_outputs.sort_values(
        ["total_goals", "assists", "simulated_minutes"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def score_mvp_candidates(
    player_outputs: pd.DataFrame,
    star_prior_temperature: float = DEFAULT_STAR_PRIOR_TEMPERATURE,
    team_success_cap: float = DEFAULT_TEAM_SUCCESS_CAP,
) -> pd.DataFrame:
    """Score MVP candidates with a fixed transparent heuristic."""
    if star_prior_temperature <= 0 or team_success_cap <= 0:
        raise ValueError("star_prior_temperature and team_success_cap must be positive.")
    scored = player_outputs.copy()
    scored["team_finish_score"] = scored["finish_rank"].map(TEAM_FINISH_SCORES).fillna(0.20).clip(upper=team_success_cap)
    scored["attacking_output_score"] = (scored["total_goals"] + 0.55 * scored["assists"]).clip(upper=8) / 8
    scored["minutes_score"] = (scored["simulated_minutes"] / (8 * 90)).clip(0, 1)
    scored["star_prior_score"] = numeric_series(scored, "star_prior", 0.0).clip(lower=0) ** (1 / star_prior_temperature)
    scored["position_bonus"] = scored["position"].map(POSITION_BONUSES).fillna(0.03)
    scored["mvp_score"] = (
        0.34 * scored["team_finish_score"]
        + 0.32 * scored["attacking_output_score"]
        + 0.16 * scored["knockout_impact_score"].clip(upper=4) / 4
        + 0.10 * scored["minutes_score"]
        + 0.06 * scored["star_prior_score"]
        + 0.02 * scored["position_bonus"]
    )
    return scored.sort_values("mvp_score", ascending=False, kind="stable").reset_index(drop=True)


def score_golden_glove_candidates(team_paths: pd.DataFrame, goalkeeper_stats: pd.DataFrame) -> pd.DataFrame:
    """Score Golden Glove candidates with a fixed transparent heuristic."""
    validate_columns(team_paths, ["simulation_id", "team", "matches_played", "clean_sheets", "finish_rank"], "team paths")
    validate_columns(
        goalkeeper_stats,
        ["goalkeeper", "team", "minutes", "saves", "goals_conceded", "save_percentage", "goals_prevented", "reputation_prior"],
        "goalkeeper stats",
    )
    historical_goalkeepers = goalkeeper_stats.rename(columns={"clean_sheets": "historical_clean_sheets"})
    scored = team_paths.merge(historical_goalkeepers, on="team", how="inner")
    scored["team_progression_score"] = scored["matches_played"].clip(upper=8) / 8
    scored["clean_sheet_score"] = scored["clean_sheets"] / scored["matches_played"].clip(lower=1)
    scored["save_score"] = numeric_series(scored, "save_percentage", 0.0).clip(0, 1)
    scored["goals_prevented_score"] = (numeric_series(scored, "goals_prevented", 0.0).clip(-5, 5) + 5) / 10
    scored["reputation_prior_score"] = numeric_series(scored, "reputation_prior", 0.0).clip(0, 1)
    scored["golden_glove_score"] = (
        0.40 * scored["team_progression_score"]
        + 0.25 * scored["clean_sheet_score"]
        + 0.15 * scored["save_score"]
        + 0.12 * scored["goals_prevented_score"]
        + 0.08 * scored["reputation_prior_score"]
    )
    return scored.sort_values("golden_glove_score", ascending=False, kind="stable").reset_index(drop=True)


def simulate_awards(
    team_paths: pd.DataFrame,
    player_features: pd.DataFrame,
    goalkeeper_stats: pd.DataFrame,
    penalty_takers: pd.DataFrame | None = None,
    random_seed: int = 2026,
    top_scorer_dispersion: float = DEFAULT_TOP_SCORER_DISPERSION,
    penalty_rate_per_match: float = DEFAULT_PENALTY_RATE_PER_MATCH,
    star_prior_temperature: float = DEFAULT_STAR_PRIOR_TEMPERATURE,
    team_success_cap: float = DEFAULT_TEAM_SUCCESS_CAP,
) -> pd.DataFrame:
    """Simulate one Top Scorer, MVP, and Golden Glove winner per tournament path."""
    rng = np.random.default_rng(random_seed)
    winners = []
    for simulation_id, simulation_paths in team_paths.groupby("simulation_id", sort=True):
        player_outputs = simulate_player_outputs(
            simulation_paths,
            player_features,
            penalty_takers=penalty_takers,
            rng=rng,
            top_scorer_dispersion=top_scorer_dispersion,
            penalty_rate_per_match=penalty_rate_per_match,
        )
        top_scorer = rank_top_scorer(player_outputs).iloc[0]
        mvp = score_mvp_candidates(
            player_outputs,
            star_prior_temperature=star_prior_temperature,
            team_success_cap=team_success_cap,
        ).iloc[0]
        golden_glove = score_golden_glove_candidates(simulation_paths, goalkeeper_stats).iloc[0]
        winners.append(
            {
                "simulation_id": simulation_id,
                "top_scorer": top_scorer["player"],
                "top_scorer_team": top_scorer["team"],
                "mvp": mvp["player"],
                "mvp_team": mvp["team"],
                "golden_glove": golden_glove["goalkeeper"],
                "golden_glove_team": golden_glove["team"],
            }
        )
    return pd.DataFrame(winners)


def award_probability_table(award_results: pd.DataFrame, award: str, team_column: str) -> pd.DataFrame:
    """Aggregate one award winner column into probabilities."""
    counts = award_results.groupby([award, team_column], dropna=False).size().rename("wins").reset_index()
    counts["probability"] = counts["wins"] / len(award_results)
    return counts.rename(columns={award: "player", team_column: "team"}).sort_values("probability", ascending=False)


def aggregate_award_probabilities(
    award_results: pd.DataFrame,
    output_dir: str | Path = "outputs",
) -> dict[str, pd.DataFrame]:
    """Write and return Top Scorer, MVP, and Golden Glove probabilities."""
    if award_results.empty:
        raise ValueError("award_results must not be empty.")
    outputs = {
        "top_scorer_probabilities": award_probability_table(award_results, "top_scorer", "top_scorer_team"),
        "mvp_probabilities": award_probability_table(award_results, "mvp", "mvp_team"),
        "golden_glove_probabilities": award_probability_table(award_results, "golden_glove", "golden_glove_team"),
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for file_name, probabilities in outputs.items():
        probabilities.to_csv(output_path / f"{file_name}.csv", index=False)
    return outputs
