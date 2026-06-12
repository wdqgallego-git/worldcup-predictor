"""Strategy, calibration, and policy decisions on the expanded tournament backtest.

Evaluates candidates against favorite_2_1 on World Cups + Euros + Copas with paired
bootstrap CIs, keeping the candidate-then-promote discipline: nothing replaces the
validated default unless it wins on the expanded set AND on the World Cup scope.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import add_baseline_predictions
from penka_backtesting import _paired_bootstrap_ci, _score_strategy, build_penka_calibration_comparison
from penka_scoring import GROUP
from poisson import score_probability_matrix, summarize_score_probs
from prediction_optimizer import get_safe_prediction


MAJOR_TOURNAMENTS = ("FIFA World Cup", "UEFA Euro", "Copa América")
TOTALS_SCALE_BOUNDS = (0.85, 1.30)
TOTALS_HISTORY_YEARS = 12
BUCKET_POLICY_GRID = {
    "strong_favorite_probability": (0.55, 0.60, 0.65, 0.70, 0.75),
    "close_lambda_difference": (0.10, 0.15, 0.20),
    "draw_probability_floor": (0.28, 0.31, 0.34),
}
BASELINE = "favorite_2_1"


def _optimizer_predictions(
    rows: pd.DataFrame,
    lambda_a_column: str,
    lambda_b_column: str,
    method: str = "independent",
    scale: pd.Series | float = 1.0,
    max_goals: int = 6,
) -> pd.DataFrame:
    """EV-optimized picks for given lambdas, optionally scaled per match."""
    predictions = rows.copy()
    scales = scale if isinstance(scale, pd.Series) else pd.Series(scale, index=rows.index)
    picks = []
    for index, match in rows.iterrows():
        factor = float(scales.loc[index])
        probabilities = score_probability_matrix(
            float(match[lambda_a_column]) * factor,
            float(match[lambda_b_column]) * factor,
            max_goals=max_goals,
            method=method,
        )
        safe = get_safe_prediction(probabilities, max_goals=max_goals, phase=match["penka_phase"])
        picks.append((safe["pred_a"], safe["pred_b"]))
    predictions[["pred_a", "pred_b"]] = pd.DataFrame(picks, index=rows.index)
    return predictions


def compute_totals_scale_factors(predictions: pd.DataFrame, results: pd.DataFrame) -> pd.Series:
    """Leakage-safe per-tournament totals recalibration factor.

    tau = (mean total goals in major tournaments inside a trailing window strictly
    before this tournament's training cutoff) / (model's mean predicted total).
    Uses only pre-cutoff outcomes and the model's own pre-match predictions.
    """
    history = results.copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history[history["tournament"].isin(MAJOR_TOURNAMENTS)]
    factors = {}
    for label, rows in predictions.groupby("backtest_label"):
        cutoff = pd.Timestamp(str(rows["training_cutoff"].iloc[0]))
        window = history[
            (history["date"] < cutoff)
            & (history["date"] >= cutoff - pd.DateOffset(years=TOTALS_HISTORY_YEARS))
        ]
        actual_mean = float((window["home_score"] + window["away_score"]).mean())
        predicted_mean = float(
            (rows["calibrated_expected_goals_a"] + rows["calibrated_expected_goals_b"]).mean()
        )
        factors[label] = float(np.clip(actual_mean / predicted_mean, *TOTALS_SCALE_BOUNDS))
    return predictions["backtest_label"].map(factors)


def _lambda_favorite_fixed(rows: pd.DataFrame, favorite_goals: int, underdog_goals: int) -> pd.DataFrame:
    """Fixed scoreline oriented by the calibrated-lambda favorite (1-1 when tied)."""
    predictions = rows.copy()
    lambda_a = rows["calibrated_expected_goals_a"]
    lambda_b = rows["calibrated_expected_goals_b"]
    predictions["pred_a"] = np.where(lambda_a >= lambda_b, favorite_goals, underdog_goals)
    predictions["pred_b"] = np.where(lambda_a >= lambda_b, underdog_goals, favorite_goals)
    tied = (lambda_a - lambda_b).abs() < 1e-9
    predictions.loc[tied, ["pred_a", "pred_b"]] = 1
    return predictions


def _add_policy_features(rows: pd.DataFrame, max_goals: int = 6) -> pd.DataFrame:
    """Match descriptors for the bucket policy, from calibrated lambdas only."""
    featured = rows.copy()
    favorite_probability, draw_probability = [], []
    for match in rows.itertuples(index=False):
        summary = summarize_score_probs(
            float(match.calibrated_expected_goals_a),
            float(match.calibrated_expected_goals_b),
            max_goals=max_goals,
        )
        favorite_probability.append(max(summary["team_a_win"], summary["team_b_win"]))
        draw_probability.append(summary["draw"])
    featured["policy_favorite_probability"] = favorite_probability
    featured["policy_draw_probability"] = draw_probability
    featured["policy_lambda_difference"] = (
        featured["calibrated_expected_goals_a"] - featured["calibrated_expected_goals_b"]
    ).abs()
    featured["policy_favorite_is_team_a"] = (
        featured["calibrated_expected_goals_a"] >= featured["calibrated_expected_goals_b"]
    )
    return featured


def _bucket_policy_predictions(
    featured: pd.DataFrame,
    strong_favorite_probability: float,
    close_lambda_difference: float,
    draw_probability_floor: float,
) -> pd.DataFrame:
    """1-1 in true coin-flips, 2-0 for strong favorites, 2-1 otherwise."""
    predictions = featured.copy()
    coin_flip = (
        predictions["policy_lambda_difference"].le(close_lambda_difference)
        & predictions["policy_draw_probability"].ge(draw_probability_floor)
    )
    strong = predictions["policy_favorite_probability"].ge(strong_favorite_probability) & ~coin_flip
    favorite_a = predictions["policy_favorite_is_team_a"]
    predictions["pred_a"] = np.where(coin_flip, 1, np.where(favorite_a, 2, np.where(strong, 0, 1)))
    predictions["pred_b"] = np.where(coin_flip, 1, np.where(favorite_a, np.where(strong, 0, 1), 2))
    return predictions


def tune_bucket_policy(featured: pd.DataFrame, train_mask: pd.Series) -> dict[str, float]:
    """Choose policy thresholds on the training tournaments only (Euros + Copas)."""
    training = featured[train_mask]
    best_params, best_average = None, -np.inf
    for strong in BUCKET_POLICY_GRID["strong_favorite_probability"]:
        for close in BUCKET_POLICY_GRID["close_lambda_difference"]:
            for draw_floor in BUCKET_POLICY_GRID["draw_probability_floor"]:
                scored = _score_strategy(
                    training, "candidate", _bucket_policy_predictions(training, strong, close, draw_floor)
                )
                average = float(scored["penka_points"].mean())
                if average > best_average:
                    best_average = average
                    best_params = {
                        "strong_favorite_probability": strong,
                        "close_lambda_difference": close,
                        "draw_probability_floor": draw_floor,
                        "training_average_penka_points": average,
                    }
    return best_params


def build_expanded_prediction_tables(
    group_rows: pd.DataFrame,
    results: pd.DataFrame,
    max_goals: int = 6,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """All strategy prediction tables on the group-stage subset."""
    totals_scale = compute_totals_scale_factors(group_rows, results)
    featured = _add_policy_features(group_rows, max_goals=max_goals)
    train_mask = featured["tournament_type"].ne("world_cup")
    policy_params = tune_bucket_policy(featured, train_mask)
    tables = {
        "favorite_1_0": add_baseline_predictions(group_rows, "favorite_1_0", max_goals=max_goals),
        "favorite_2_0": add_baseline_predictions(group_rows, "favorite_2_0", max_goals=max_goals),
        "favorite_2_1": add_baseline_predictions(group_rows, "favorite_2_1", max_goals=max_goals),
        "lambda_favorite_2_1": _lambda_favorite_fixed(group_rows, 2, 1),
        "lambda_favorite_1_0": _lambda_favorite_fixed(group_rows, 1, 0),
        "calibrated_optimizer": _optimizer_predictions(
            group_rows, "calibrated_expected_goals_a", "calibrated_expected_goals_b", max_goals=max_goals
        ),
        "calibrated_optimizer_dixon_coles": _optimizer_predictions(
            group_rows, "calibrated_expected_goals_a", "calibrated_expected_goals_b",
            method="dixon_coles", max_goals=max_goals,
        ),
        "totals_recalibrated_optimizer": _optimizer_predictions(
            group_rows, "calibrated_expected_goals_a", "calibrated_expected_goals_b",
            scale=totals_scale, max_goals=max_goals,
        ),
        "totals_recalibrated_dixon_coles": _optimizer_predictions(
            group_rows, "calibrated_expected_goals_a", "calibrated_expected_goals_b",
            method="dixon_coles", scale=totals_scale, max_goals=max_goals,
        ),
        "bucket_policy_tuned_on_euro_copa": _bucket_policy_predictions(
            featured,
            policy_params["strong_favorite_probability"],
            policy_params["close_lambda_difference"],
            policy_params["draw_probability_floor"],
        ),
    }
    diagnostics = {
        "totals_scale_factors": totals_scale.groupby(group_rows["backtest_label"]).first().to_dict(),
        "bucket_policy_params": policy_params,
    }
    return tables, diagnostics


def summarize_scope(
    points_by_strategy: dict[str, pd.Series],
    mask: pd.Series,
    scope: str,
) -> pd.DataFrame:
    """Per-strategy averages plus paired-bootstrap difference vs favorite_2_1."""
    baseline_points = points_by_strategy[BASELINE][mask].to_numpy(dtype=float)
    rows = []
    for strategy, points in points_by_strategy.items():
        selected = points[mask].to_numpy(dtype=float)
        ci_lower, ci_upper = _paired_bootstrap_ci(selected - baseline_points)
        rows.append(
            {
                "scope": scope,
                "strategy": strategy,
                "matches": int(mask.sum()),
                "average_penka_points": float(selected.mean()) if mask.any() else np.nan,
                "total_penka_points": int(selected.sum()) if mask.any() else 0,
                "difference_vs_favorite_2_1": float((selected - baseline_points).mean()) if mask.any() else np.nan,
                "ci_lower_95_vs_favorite_2_1": ci_lower,
                "ci_upper_95_vs_favorite_2_1": ci_upper,
            }
        )
    return pd.DataFrame(rows).sort_values("average_penka_points", ascending=False, kind="stable")


def run_expanded_analysis(
    predictions: pd.DataFrame,
    results: pd.DataFrame,
    output_dir: str | Path = "outputs",
    max_goals: int = 6,
) -> dict[str, object]:
    """Score every strategy on the group-stage subset and write decision artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    group_rows = predictions[predictions["penka_phase"].eq(GROUP)].copy().reset_index(drop=True)
    tables, diagnostics = build_expanded_prediction_tables(group_rows, results, max_goals=max_goals)
    points_by_strategy = {
        strategy: _score_strategy(group_rows, strategy, table)["penka_points"]
        for strategy, table in tables.items()
    }
    scopes = {
        "expanded_all_group": pd.Series(True, index=group_rows.index),
        "wc_only_group": group_rows["tournament_type"].eq("world_cup"),
        "euro_only_group": group_rows["tournament_type"].eq("euro"),
        "copa_only_group": group_rows["tournament_type"].eq("copa_america"),
    }
    scope_frames = [summarize_scope(points_by_strategy, mask, scope) for scope, mask in scopes.items()]
    per_tournament = [
        summarize_scope(points_by_strategy, group_rows["backtest_label"].eq(label), f"tournament_{label}")
        for label in group_rows["backtest_label"].unique()
    ]
    comparison = pd.concat(scope_frames + per_tournament, ignore_index=True)
    comparison.to_csv(output_path / "expanded_strategy_comparison.csv", index=False)
    calibration = build_penka_calibration_comparison(group_rows).assign(scope="expanded_all_group")
    calibration.to_csv(output_path / "expanded_calibration_comparison.csv", index=False)

    # Decision: candidate must beat favorite_2_1 on the expanded group scope with a
    # positive CI lower bound AND be ahead on the World Cup scope (the target domain).
    # bucket_policy is judged on wc_only_group, which it was never tuned on.
    expanded = comparison[comparison["scope"].eq("expanded_all_group")].set_index("strategy")
    wc_scope = comparison[comparison["scope"].eq("wc_only_group")].set_index("strategy")
    candidates = [strategy for strategy in tables if not strategy.startswith("favorite_")]
    decision_rows = []
    for strategy in candidates:
        passes_expanded = bool(expanded.loc[strategy, "ci_lower_95_vs_favorite_2_1"] > 0)
        ahead_on_wc = bool(wc_scope.loc[strategy, "difference_vs_favorite_2_1"] > 0)
        decision_rows.append(
            {
                "strategy": strategy,
                "expanded_average": float(expanded.loc[strategy, "average_penka_points"]),
                "expanded_difference_vs_favorite_2_1": float(expanded.loc[strategy, "difference_vs_favorite_2_1"]),
                "expanded_ci_lower": float(expanded.loc[strategy, "ci_lower_95_vs_favorite_2_1"]),
                "wc_difference_vs_favorite_2_1": float(wc_scope.loc[strategy, "difference_vs_favorite_2_1"]),
                "passes_expanded_ci_gate": passes_expanded,
                "ahead_on_world_cups": ahead_on_wc,
                "recommended": passes_expanded and ahead_on_wc,
            }
        )
    decisions = pd.DataFrame(decision_rows).sort_values("expanded_average", ascending=False, kind="stable")
    recommended = decisions[decisions["recommended"]]
    lines = [
        "Expanded Backtest Strategy Decision",
        "===================================",
        f"group_matches_evaluated: {len(group_rows)} across {group_rows['backtest_label'].nunique()} tournaments",
        f"scopes: expanded_all_group / wc_only_group / euro_only_group / copa_only_group / per tournament",
        f"baseline: {BASELINE} at {expanded.loc[BASELINE, 'average_penka_points']:.4f} pts/match (expanded), "
        f"{wc_scope.loc[BASELINE, 'average_penka_points']:.4f} (WC only)",
        f"totals_scale_factors: { {k: round(v, 3) for k, v in diagnostics['totals_scale_factors'].items()} }",
        f"bucket_policy_params (tuned on Euros+Copas only): {diagnostics['bucket_policy_params']}",
        "",
        "Candidate gate: positive 95% CI lower bound vs favorite_2_1 on expanded_all_group AND",
        "positive average difference on wc_only_group.",
        "",
    ]
    for row in decisions.itertuples(index=False):
        lines.append(
            f"{row.strategy}: expanded {row.expanded_average:.4f} "
            f"(diff {row.expanded_difference_vs_favorite_2_1:+.4f}, CI lower {row.expanded_ci_lower:+.4f}), "
            f"WC diff {row.wc_difference_vs_favorite_2_1:+.4f} -> "
            f"{'RECOMMENDED' if row.recommended else 'not promoted'}"
        )
    lines.append("")
    lines.append(
        f"Decision: promote {recommended.iloc[0]['strategy']} as the new default group strategy."
        if not recommended.empty
        else "Decision: keep the current validated default; no candidate cleared both gates."
    )
    (output_path / "expanded_strategy_decision.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    decisions.to_csv(output_path / "expanded_strategy_decisions.csv", index=False)
    return {
        "comparison": comparison,
        "decisions": decisions,
        "diagnostics": diagnostics,
    }
