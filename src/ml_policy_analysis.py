"""Walk-forward ML diagnostics for direct contest-score action selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from elo_analysis import ELO_BUCKETS
from evaluation import add_baseline_predictions
from prediction_optimizer import points_for_prediction
from rule_b_analysis import build_severity_axes


BOOTSTRAP_SAMPLES = 5000
POLICY_HOLDOUTS = [(2008, 2011), (2012, 2015), (2016, 2019), (2020, 2025)]
RANK_POLICY_THRESHOLDS = [40, 50, 60, 70, 80, 90, 100]
ELO_POLICY_THRESHOLDS = [100, 150, 200, 250, 300, 350]
MIN_THRESHOLD_MATCHES = 100
ACTION_SCORES = {
    "favorite_1_0": (1, 0),
    "favorite_2_0": (2, 0),
    "favorite_2_1": (2, 1),
    "favorite_3_0": (3, 0),
    "draw_1_1": (1, 1),
    "draw_0_0": (0, 0),
    "underdog_1_0": (0, 1),
}
ACTION_ORDER = list(ACTION_SCORES)
POLICY_FEATURES = [
    "causal_elo_gap",
    "ordinal_rank_gap",
    "ranking_points_gap",
    "neutral_int",
    "is_friendly",
    "is_world_cup",
    "year",
    "favorite_recent_points_per_match_5",
    "underdog_recent_points_per_match_5",
    "favorite_recent_goals_scored_5",
    "underdog_recent_goals_scored_5",
    "favorite_recent_goals_conceded_5",
    "underdog_recent_goals_conceded_5",
]
STRATEGIES = [
    "favorite_1_0",
    "rank_gap_fixed_rule",
    "elo_gap_fixed_rule",
    "ml_policy",
    "oracle_upper_bound",
]


def action_points(matches: pd.DataFrame, action: str) -> np.ndarray:
    """Return realized contest points for one favorite-oriented action."""
    favorite_goals, underdog_goals = ACTION_SCORES[action]
    return np.asarray(
        [
            points_for_prediction(favorite_goals, underdog_goals, int(actual_favorite), int(actual_underdog))
            for actual_favorite, actual_underdog in zip(matches["favorite_goals"], matches["underdog_goals"])
        ],
        dtype=int,
    )


def prepare_policy_matches(results: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """Build pre-match-only policy features and candidate-action rewards."""
    matches, _, _ = build_severity_axes(results, rankings)
    matches = matches.copy()
    matches["neutral_int"] = matches["neutral"].fillna(False).astype(bool).astype(int)
    tournament = matches["tournament"].fillna("").astype(str)
    matches["is_friendly"] = tournament.str.contains("friendly", case=False, regex=False).astype(int)
    matches["is_world_cup"] = tournament.eq("FIFA World Cup").astype(int)
    matches["tournament_type"] = np.select(
        [matches["is_friendly"].eq(1), matches["is_world_cup"].eq(1)],
        ["Friendly", "FIFA World Cup"],
        default="Competitive Other",
    )
    matches["favorite_recent_points_per_match_5"] = np.where(
        matches["favorite_is_a"],
        matches["team_a_recent_points_per_match_5"],
        matches["team_b_recent_points_per_match_5"],
    )
    matches["underdog_recent_points_per_match_5"] = np.where(
        matches["favorite_is_a"],
        matches["team_b_recent_points_per_match_5"],
        matches["team_a_recent_points_per_match_5"],
    )
    matches["favorite_recent_goals_scored_5"] = np.where(
        matches["favorite_is_a"],
        matches["team_a_recent_goals_scored_5"],
        matches["team_b_recent_goals_scored_5"],
    )
    matches["underdog_recent_goals_scored_5"] = np.where(
        matches["favorite_is_a"],
        matches["team_b_recent_goals_scored_5"],
        matches["team_a_recent_goals_scored_5"],
    )
    matches["favorite_recent_goals_conceded_5"] = np.where(
        matches["favorite_is_a"],
        matches["team_a_recent_goals_conceded_5"],
        matches["team_b_recent_goals_conceded_5"],
    )
    matches["underdog_recent_goals_conceded_5"] = np.where(
        matches["favorite_is_a"],
        matches["team_b_recent_goals_conceded_5"],
        matches["team_a_recent_goals_conceded_5"],
    )
    matches["elo_bucket"] = pd.cut(
        matches["causal_elo_gap"],
        bins=ELO_BUCKETS["bins"],
        labels=ELO_BUCKETS["labels"],
        right=False,
        include_lowest=True,
    )
    for action in ACTION_ORDER:
        matches[f"{action}_points"] = action_points(matches, action)
    return matches


def select_threshold(
    training_rows: pd.DataFrame,
    feature: str,
    thresholds: list[int],
) -> float:
    """Select a fold-local 2-0 threshold only when it beats the 1-0 default."""
    baseline = training_rows["favorite_1_0_points"].mean()
    candidates = []
    for threshold in thresholds:
        severe = training_rows[feature].notna() & training_rows[feature].ge(threshold)
        if int(severe.sum()) < MIN_THRESHOLD_MATCHES:
            continue
        actions = np.where(severe, "favorite_2_0", "favorite_1_0")
        points = np.asarray(
            [row[f"{action}_points"] for action, (_, row) in zip(actions, training_rows.iterrows())],
            dtype=float,
        )
        severe_rows = training_rows[severe]
        candidates.append(
            {
                "threshold": float(threshold),
                "avg_points": float(points.mean()),
                "severe_by_2_gte_by_1": bool(
                    severe_rows["favorite_win_by_2"].mean() >= severe_rows["favorite_win_by_1"].mean()
                ),
            }
        )
    eligible = [candidate for candidate in candidates if candidate["avg_points"] > baseline and candidate["severe_by_2_gte_by_1"]]
    return max(eligible, key=lambda candidate: candidate["avg_points"])["threshold"] if eligible else np.inf


def build_reward_model() -> Pipeline:
    """Return a compact CPU-friendly reward regressor."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    max_iter=80,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    l2_regularization=0.2,
                    random_state=42,
                ),
            ),
        ]
    )


def predict_ml_policy_actions(training_rows: pd.DataFrame, holdout_rows: pd.DataFrame) -> np.ndarray:
    """Fit per-action reward models on prior rows and choose the largest predicted reward."""
    predicted_rewards = []
    for action in ACTION_ORDER:
        model = build_reward_model()
        model.fit(training_rows[POLICY_FEATURES], training_rows[f"{action}_points"])
        predicted_rewards.append(model.predict(holdout_rows[POLICY_FEATURES]))
    rewards = np.column_stack(predicted_rewards)
    return np.asarray([ACTION_ORDER[index] for index in np.argmax(rewards, axis=1)], dtype=object)


def selected_action_points(rows: pd.DataFrame, actions) -> np.ndarray:
    """Return realized points for row-level selected actions."""
    return np.asarray(
        [row[f"{action}_points"] for action, (_, row) in zip(actions, rows.iterrows())],
        dtype=int,
    )


def build_walk_forward_policy_predictions(matches: pd.DataFrame) -> pd.DataFrame:
    """Train policy models on earlier years and concatenate later-year holdouts."""
    frames = []
    for start_year, end_year in POLICY_HOLDOUTS:
        training = matches[matches["year"] < start_year].copy()
        holdout = matches[matches["year"].between(start_year, end_year)].copy()
        if training.empty or holdout.empty:
            continue
        rank_threshold = select_threshold(training, "ordinal_rank_gap", RANK_POLICY_THRESHOLDS)
        elo_threshold = select_threshold(training, "causal_elo_gap", ELO_POLICY_THRESHOLDS)
        holdout["favorite_1_0_action"] = "favorite_1_0"
        holdout["rank_gap_fixed_rule_action"] = np.where(
            holdout["ordinal_rank_gap"].notna() & holdout["ordinal_rank_gap"].ge(rank_threshold),
            "favorite_2_0",
            "favorite_1_0",
        )
        holdout["elo_gap_fixed_rule_action"] = np.where(
            holdout["causal_elo_gap"].ge(elo_threshold),
            "favorite_2_0",
            "favorite_1_0",
        )
        holdout["ml_policy_action"] = predict_ml_policy_actions(training, holdout)
        reward_matrix = holdout[[f"{action}_points" for action in ACTION_ORDER]].to_numpy(dtype=float)
        holdout["oracle_upper_bound_action"] = [ACTION_ORDER[index] for index in np.argmax(reward_matrix, axis=1)]
        holdout["rank_gap_threshold_from_training"] = rank_threshold
        holdout["elo_gap_threshold_from_training"] = elo_threshold
        holdout["holdout_period"] = f"{start_year}_{end_year}"
        for strategy in STRATEGIES:
            holdout[f"{strategy}_points"] = selected_action_points(holdout, holdout[f"{strategy}_action"])
        frames.append(holdout)
    return pd.concat(frames, ignore_index=True)


def bootstrap_ci(strategy_points, baseline_points, random_seed: int = 2026) -> tuple[float, float]:
    """Return a paired-bootstrap CI for strategy minus favorite_1_0."""
    differences = np.asarray(strategy_points, dtype=float) - np.asarray(baseline_points, dtype=float)
    if not len(differences):
        return np.nan, np.nan
    rng = np.random.default_rng(random_seed)
    indexes = rng.integers(0, len(differences), size=(BOOTSTRAP_SAMPLES, len(differences)))
    means = differences[indexes].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_strategy(rows: pd.DataFrame, strategy: str, dimension: str, bucket_value: str) -> dict[str, object]:
    """Summarize one walk-forward strategy for one diagnostic slice."""
    actions = rows[f"{strategy}_action"]
    points = rows[f"{strategy}_points"].to_numpy(dtype=float)
    favorite_goals = np.asarray([ACTION_SCORES[action][0] for action in actions])
    underdog_goals = np.asarray([ACTION_SCORES[action][1] for action in actions])
    margins = favorite_goals - underdog_goals
    actual_favorite = rows["favorite_goals"].to_numpy(dtype=float)
    actual_underdog = rows["underdog_goals"].to_numpy(dtype=float)
    actual_margins = actual_favorite - actual_underdog
    ci_lower, ci_upper = bootstrap_ci(points, rows["favorite_1_0_points"])
    return {
        "scope": "full_history_walk_forward",
        "dimension": dimension,
        "bucket_value": bucket_value,
        "strategy": strategy,
        "match_count": len(rows),
        "avg_contest_points": float(points.mean()),
        "difference_vs_favorite_1_0": float((points - rows["favorite_1_0_points"]).mean()),
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "exact_score_rate": float(((favorite_goals == actual_favorite) & (underdog_goals == actual_underdog)).mean()),
        "correct_result_rate": float((np.sign(margins) == np.sign(actual_margins)).mean()),
        "correct_goal_difference_rate": float((margins == actual_margins).mean()),
    }


def build_policy_strategy_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report policy performance overall, by holdout period, year, and tournament type."""
    rows = []
    dimensions = [
        ("overall", pd.Series("all", index=predictions.index)),
        ("holdout_period", predictions["holdout_period"]),
        ("year", predictions["year"].astype(str)),
        ("tournament_type", predictions["tournament_type"]),
    ]
    for dimension, labels in dimensions:
        for bucket_value, indexes in labels.groupby(labels).groups.items():
            bucket = predictions.loc[indexes]
            for strategy in STRATEGIES:
                rows.append(summarize_strategy(bucket, strategy, dimension, str(bucket_value)))
    return pd.DataFrame(rows)


def build_policy_bucket_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report policy performance by causal Elo bucket."""
    rows = []
    for bucket_value, bucket_rows in predictions.groupby("elo_bucket", observed=True):
        for strategy in STRATEGIES:
            rows.append(summarize_strategy(bucket_rows, strategy, "elo_bucket", str(bucket_value)))
    return pd.DataFrame(rows)


def add_world_cup_calibrated_reference(output_dir: Path, comparison: pd.DataFrame) -> pd.DataFrame:
    """Append the existing calibrated-fallback World Cup reference comparison."""
    path = output_dir / "backtest_predictions.csv"
    if not path.exists():
        return comparison
    predictions = pd.read_csv(path)
    required = {
        "backtest_year", "goals_a", "goals_b", "rank_a", "rank_b",
        "calibrated_fallback_pred_a", "calibrated_fallback_pred_b",
        "favorite_1_0_points", "calibrated_optimizer_with_favorite_1_0_close_match_fallback_points",
    }
    if not required.issubset(predictions.columns):
        return comparison
    favorite = add_baseline_predictions(predictions, "favorite_1_0")
    rows = []
    strategy_specs = {
        "favorite_1_0": (favorite["pred_a"], favorite["pred_b"], predictions["favorite_1_0_points"]),
        "calibrated_optimizer_with_fallback": (
            predictions["calibrated_fallback_pred_a"],
            predictions["calibrated_fallback_pred_b"],
            predictions["calibrated_optimizer_with_favorite_1_0_close_match_fallback_points"],
        ),
    }
    for dimension, labels in [
        ("overall", pd.Series("all", index=predictions.index)),
        ("year", predictions["backtest_year"].astype(str)),
    ]:
        for bucket_value, indexes in labels.groupby(labels).groups.items():
            bucket = predictions.loc[indexes]
            for strategy, (pred_a, pred_b, points) in strategy_specs.items():
                selected_a = pred_a.loc[indexes].to_numpy(dtype=int)
                selected_b = pred_b.loc[indexes].to_numpy(dtype=int)
                selected_points = points.loc[indexes].to_numpy(dtype=float)
                baseline_points = predictions.loc[indexes, "favorite_1_0_points"].to_numpy(dtype=float)
                ci_lower, ci_upper = bootstrap_ci(selected_points, baseline_points)
                rows.append(
                    {
                        "scope": "world_cup_2014_2018_2022_reference",
                        "dimension": dimension,
                        "bucket_value": str(bucket_value),
                        "strategy": strategy,
                        "match_count": len(bucket),
                        "avg_contest_points": float(selected_points.mean()),
                        "difference_vs_favorite_1_0": float((selected_points - baseline_points).mean()),
                        "ci_lower_95": ci_lower,
                        "ci_upper_95": ci_upper,
                        "exact_score_rate": float(
                            ((selected_a == bucket["goals_a"].to_numpy()) & (selected_b == bucket["goals_b"].to_numpy())).mean()
                        ),
                        "correct_result_rate": float(
                            (
                                np.sign(selected_a - selected_b)
                                == np.sign(bucket["goals_a"].to_numpy() - bucket["goals_b"].to_numpy())
                            ).mean()
                        ),
                        "correct_goal_difference_rate": float(
                            (
                                selected_a - selected_b
                                == bucket["goals_a"].to_numpy() - bucket["goals_b"].to_numpy()
                            ).mean()
                        ),
                    }
                )
    return pd.concat([comparison, pd.DataFrame(rows)], ignore_index=True)


def count_relevant_2026_fixture_categories(output_dir: Path, bucket_comparison: pd.DataFrame) -> int:
    """Count 2026 fixtures in Elo buckets where ML has a positive OOS CI."""
    path = output_dir / "rule_b_2026_affected_fixtures.csv"
    if not path.exists():
        return 0
    fixtures = pd.read_csv(path)
    fixtures["elo_bucket"] = pd.cut(
        fixtures["causal_elo_gap"],
        bins=ELO_BUCKETS["bins"],
        labels=ELO_BUCKETS["labels"],
        right=False,
        include_lowest=True,
    )
    validated = bucket_comparison[
        bucket_comparison["strategy"].eq("ml_policy")
        & bucket_comparison["match_count"].ge(100)
        & bucket_comparison["difference_vs_favorite_1_0"].gt(0)
        & bucket_comparison["ci_lower_95"].gt(0)
    ]["bucket_value"]
    return int(fixtures["elo_bucket"].astype(str).isin(set(validated)).sum())


def write_ml_policy_summary(
    comparison: pd.DataFrame,
    bucket_comparison: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write the final direct-score policy recommendation."""
    overall = comparison[
        comparison["scope"].eq("full_history_walk_forward")
        & comparison["dimension"].eq("overall")
        & comparison["bucket_value"].eq("all")
    ].set_index("strategy")
    ml = overall.loc["ml_policy"]
    periods = comparison[
        comparison["scope"].eq("full_history_walk_forward")
        & comparison["dimension"].eq("holdout_period")
        & comparison["strategy"].eq("ml_policy")
    ]
    positive_periods = int(periods["difference_vs_favorite_1_0"].gt(0).sum())
    relevant_fixtures = count_relevant_2026_fixture_categories(output_dir, bucket_comparison)
    recommend_replace = bool(
        ml["difference_vs_favorite_1_0"] > 0
        and (ml["ci_lower_95"] > 0 or positive_periods >= 2)
        and relevant_fixtures >= 1
    )
    if recommend_replace:
        decision = "Reopen match strategy: ML policy passes the diagnostic replacement gate."
    else:
        decision = (
            "Advisory full-history group-approximation result: the ML policy does not replace its fixed-score baseline. "
            "Use penka_scoring_summary.txt for the authoritative World Cup recommendation."
        )
    lines = [
        "Direct Contest-Score ML Policy Summary",
        "======================================",
        "Scoring note: full-history rows use the real Penka group-stage scoring approximation.",
        "Policy features: causal Elo gap, FIFA rank gap, FIFA ranking-points gap, neutral flag, tournament flags, year, and leakage-safe recent form.",
        "Skipped leakage-prone features: expected goals and favorite win probability; full-history production walk-forward values are unavailable.",
        "Skipped candidate action: poisson_best_ev; full-history production walk-forward score matrices are unavailable.",
        "",
        "Overall full-history walk-forward comparison:",
        overall[
            ["match_count", "avg_contest_points", "difference_vs_favorite_1_0", "ci_lower_95", "ci_upper_95"]
        ].to_string(),
        "",
        f"ml_policy_positive_holdout_periods: {positive_periods}/{len(periods)}",
        f"ml_policy_relevant_2026_fixture_categories_count: {relevant_fixtures}",
        f"replacement_gate_passed: {recommend_replace}",
        f"decision: {decision}",
        "",
        "ML policy by holdout period:",
        periods[
            ["bucket_value", "match_count", "avg_contest_points", "difference_vs_favorite_1_0", "ci_lower_95", "ci_upper_95"]
        ].to_string(index=False),
        "",
        "ML policy by causal Elo bucket:",
        bucket_comparison[bucket_comparison["strategy"].eq("ml_policy")][
            ["bucket_value", "match_count", "avg_contest_points", "difference_vs_favorite_1_0", "ci_lower_95", "ci_upper_95"]
        ].to_string(index=False),
        "",
        "No final_match_recommendations files were changed automatically.",
    ]
    (output_dir / "ml_policy_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ml_policy_analysis(
    results: pd.DataFrame,
    rankings: pd.DataFrame,
    output_dir: str | Path = "outputs",
) -> dict[str, pd.DataFrame]:
    """Run walk-forward direct contest-score policy diagnostics."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    matches = prepare_policy_matches(results, rankings)
    predictions = build_walk_forward_policy_predictions(matches)
    comparison = build_policy_strategy_comparison(predictions)
    comparison = add_world_cup_calibrated_reference(output_path, comparison)
    bucket_comparison = build_policy_bucket_comparison(predictions)
    comparison.to_csv(output_path / "ml_policy_strategy_comparison.csv", index=False)
    bucket_comparison.to_csv(output_path / "ml_policy_bucket_comparison.csv", index=False)
    write_ml_policy_summary(comparison, bucket_comparison, output_path)
    return {
        "ml_policy_strategy_comparison": comparison,
        "ml_policy_bucket_comparison": bucket_comparison,
        "ml_policy_predictions": predictions,
    }
