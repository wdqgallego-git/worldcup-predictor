"""Authoritative strategy and calibration reports under the real Penka rules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import add_baseline_predictions, most_likely_poisson_score
from penka_scoring import GROUP, UNKNOWN_PHASE, prediction_category, score_prediction
from poisson import score_probability_matrix, summarize_score_probs
from prediction_optimizer import expected_points_for_prediction, get_result, get_safe_prediction


COMMON_SCORES = ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2))
FIXED_BASELINES = ("favorite_1_0", "favorite_2_0", "favorite_2_1", "favorite_3_0", "draw_0_0", "draw_1_1")
DEFAULT_BOOTSTRAP_SAMPLES = 3000


def _score_strings(predictions: pd.DataFrame) -> pd.Series:
    return predictions["pred_a"].astype(int).astype(str) + "-" + predictions["pred_b"].astype(int).astype(str)


def _poisson_optimizer_predictions(
    rows: pd.DataFrame,
    lambda_a_column: str,
    lambda_b_column: str,
    max_goals: int = 6,
) -> pd.DataFrame:
    predictions = rows.copy()
    scores = []
    for match in predictions.itertuples(index=False):
        probabilities = score_probability_matrix(
            float(getattr(match, lambda_a_column)),
            float(getattr(match, lambda_b_column)),
            max_goals=max_goals,
        )
        phase = getattr(match, "penka_phase", GROUP)
        safe = get_safe_prediction(probabilities, max_goals=max_goals, phase=phase)
        scores.append((safe["pred_a"], safe["pred_b"]))
    predictions[["pred_a", "pred_b"]] = pd.DataFrame(scores, index=predictions.index)
    return predictions


def _most_likely_predictions(rows: pd.DataFrame, max_goals: int = 6) -> pd.DataFrame:
    predictions = rows.copy()
    predictions[["pred_a", "pred_b"]] = pd.DataFrame(
        [
            most_likely_poisson_score(float(lambda_a), float(lambda_b), max_goals=max_goals)
            for lambda_a, lambda_b in zip(predictions["expected_goals_a"], predictions["expected_goals_b"])
        ],
        index=predictions.index,
    )
    return predictions


def _rank_gap_rule_predictions(rows: pd.DataFrame) -> pd.DataFrame:
    predictions = add_baseline_predictions(rows, "favorite_1_0")
    severe = (predictions["rank_a"] - predictions["rank_b"]).abs().ge(80)
    two_nil = add_baseline_predictions(rows, "favorite_2_0")
    predictions.loc[severe, ["pred_a", "pred_b"]] = two_nil.loc[severe, ["pred_a", "pred_b"]]
    return predictions


def _close_match_fallback(rows: pd.DataFrame, max_goals: int = 6) -> pd.DataFrame:
    predictions = _poisson_optimizer_predictions(
        rows,
        "calibrated_expected_goals_a",
        "calibrated_expected_goals_b",
        max_goals=max_goals,
    )
    favorite = add_baseline_predictions(rows, "favorite_1_0")
    close = (rows["calibrated_expected_goals_a"] - rows["calibrated_expected_goals_b"]).abs().le(0.20)
    predictions.loc[close, ["pred_a", "pred_b"]] = favorite.loc[close, ["pred_a", "pred_b"]]
    return predictions


def _score_strategy(rows: pd.DataFrame, strategy: str, predicted: pd.DataFrame) -> pd.DataFrame:
    scored = rows[["backtest_year", "penka_phase", "goals_a", "goals_b"]].copy()
    scored["strategy"] = strategy
    scored["pred_a"] = predicted["pred_a"].astype(int)
    scored["pred_b"] = predicted["pred_b"].astype(int)
    scored["prediction"] = _score_strings(scored)
    scored["penka_points"] = [
        score_prediction(int(pred_a), int(pred_b), int(actual_a), int(actual_b), phase)
        for pred_a, pred_b, actual_a, actual_b, phase in zip(
            scored["pred_a"], scored["pred_b"], scored["goals_a"], scored["goals_b"], scored["penka_phase"]
        )
    ]
    scored["penka_category"] = [
        prediction_category(int(pred_a), int(pred_b), int(actual_a), int(actual_b))
        for pred_a, pred_b, actual_a, actual_b in zip(
            scored["pred_a"], scored["pred_b"], scored["goals_a"], scored["goals_b"]
        )
    ]
    return scored


def build_strategy_points(backtests: pd.DataFrame, max_goals: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return match context and long per-strategy points under real Penka scoring."""
    context = backtests.copy().reset_index(drop=True)
    if "penka_phase" not in context.columns:
        raise ValueError("backtest_predictions.csv must include penka_phase.")
    context = context[context["penka_phase"].ne(UNKNOWN_PHASE)].copy().reset_index(drop=True)
    summaries = [
        summarize_score_probs(float(lambda_a), float(lambda_b), max_goals=max_goals)
        for lambda_a, lambda_b in zip(context["calibrated_expected_goals_a"], context["calibrated_expected_goals_b"])
    ]
    context["favorite_win_probability"] = [
        max(summary["team_a_win"], summary["team_b_win"]) for summary in summaries
    ]
    context["draw_probability"] = [summary["draw"] for summary in summaries]
    context["rank_gap_abs"] = (context["rank_a"] - context["rank_b"]).abs()
    context["expected_goal_difference_abs"] = (
        context["calibrated_expected_goals_a"] - context["calibrated_expected_goals_b"]
    ).abs()
    context["expected_total_goals"] = context["calibrated_expected_goals_a"] + context["calibrated_expected_goals_b"]
    context["rank_gap_bucket"] = pd.cut(
        context["rank_gap_abs"],
        bins=[-np.inf, 10, 25, 40, 60, 80, np.inf],
        labels=["0-10", "10-25", "25-40", "40-60", "60-80", "80+"],
    )
    context["favorite_win_probability_bucket"] = pd.cut(
        context["favorite_win_probability"],
        bins=[-np.inf, 0.50, 0.60, 0.70, 0.80, np.inf],
        labels=["up_to_0.50", "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80+"],
    )
    context["high_mismatch"] = (
        context["rank_gap_abs"].ge(60)
        | context["favorite_win_probability"].ge(0.70)
        | context["expected_goal_difference_abs"].ge(0.75)
    )
    prediction_tables = {
        baseline: add_baseline_predictions(context, baseline, max_goals=max_goals)
        for baseline in FIXED_BASELINES
    }
    prediction_tables["most_likely_score"] = _most_likely_predictions(context, max_goals=max_goals)
    prediction_tables["poisson_best_ev"] = _poisson_optimizer_predictions(
        context, "raw_expected_goals_a", "raw_expected_goals_b", max_goals=max_goals
    )
    prediction_tables["calibrated_optimizer"] = _poisson_optimizer_predictions(
        context, "calibrated_expected_goals_a", "calibrated_expected_goals_b", max_goals=max_goals
    )
    prediction_tables["calibrated_optimizer_with_close_match_fallback"] = _close_match_fallback(
        context, max_goals=max_goals
    )
    prediction_tables["rank_gap_fixed_score_rule"] = _rank_gap_rule_predictions(context)
    prediction_tables["previous_final_recommendation_rule"] = prediction_tables["favorite_1_0"].copy()
    points = pd.concat(
        [_score_strategy(context, strategy, predictions) for strategy, predictions in prediction_tables.items()],
        ignore_index=True,
    )
    points["match_index"] = np.tile(np.arange(len(context)), len(prediction_tables))
    return context, points


def _paired_bootstrap_ci(differences: np.ndarray, seed: int = 2026) -> tuple[float, float]:
    if len(differences) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(differences), size=(DEFAULT_BOOTSTRAP_SAMPLES, len(differences)))
    means = differences[indexes].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _focused_strategy_metrics(
    scored: pd.DataFrame,
    strategy: str,
    scope: str,
    baseline_points: pd.Series,
) -> dict[str, object]:
    """Return direct production-style group strategy metrics."""
    points = scored["penka_points"].to_numpy(dtype=float)
    baseline = baseline_points.loc[scored.index].to_numpy(dtype=float)
    ci_lower, ci_upper = _paired_bootstrap_ci(points - baseline)
    exact = scored["penka_category"].eq("exact_score")
    correct_result = [
        get_result(int(pred_a), int(pred_b)) == get_result(int(actual_a), int(actual_b))
        for pred_a, pred_b, actual_a, actual_b in zip(
            scored["pred_a"], scored["pred_b"], scored["goals_a"], scored["goals_b"]
        )
    ]
    correct_goal_difference = [
        int(pred_a) - int(pred_b) == int(actual_a) - int(actual_b)
        for pred_a, pred_b, actual_a, actual_b in zip(
            scored["pred_a"], scored["pred_b"], scored["goals_a"], scored["goals_b"]
        )
    ]
    return {
        "scope": scope,
        "strategy": strategy,
        "matches": len(scored),
        "total_penka_points": int(points.sum()),
        "average_penka_points": float(points.mean()),
        "exact_score_rate": float(exact.mean()),
        "correct_result_rate": float(np.mean(correct_result)),
        "correct_goal_difference_rate": float(np.mean(correct_goal_difference)),
        "difference_vs_favorite_1_0": float((points - baseline).mean()),
        "ci_lower_95_vs_favorite_1_0": ci_lower,
        "ci_upper_95_vs_favorite_1_0": ci_upper,
    }


def build_focused_group_strategy_comparison(backtests: pd.DataFrame) -> pd.DataFrame:
    """Compare the exact refreshable group-file strategy against favorite_1_0."""
    group = backtests[backtests["penka_phase"].eq(GROUP)].copy().reset_index(drop=True)
    favorite = add_baseline_predictions(group, "favorite_1_0", max_goals=5)
    production = _poisson_optimizer_predictions(
        group,
        "calibrated_expected_goals_a",
        "calibrated_expected_goals_b",
        max_goals=5,
    )
    scored_favorite = _score_strategy(group, "favorite_1_0", favorite)
    scored_production = _score_strategy(group, "penka_group_strategy", production)
    baseline_points = scored_favorite["penka_points"]
    rows = [
        _focused_strategy_metrics(scored_favorite, "favorite_1_0", "overall_group_stage", baseline_points),
        _focused_strategy_metrics(scored_production, "penka_group_strategy", "overall_group_stage", baseline_points),
    ]
    for year in sorted(group["backtest_year"].unique()):
        mask = group["backtest_year"].eq(year)
        year_favorite = scored_favorite[mask].copy()
        year_production = scored_production[mask].copy()
        year_baseline = scored_favorite["penka_points"]
        rows.extend(
            [
                _focused_strategy_metrics(year_favorite, "favorite_1_0", str(int(year)), year_baseline),
                _focused_strategy_metrics(year_production, "penka_group_strategy", str(int(year)), year_baseline),
            ]
        )
    return pd.DataFrame(rows)


def write_focused_group_strategy_summary(comparison: pd.DataFrame, output_dir: Path) -> Path:
    """Write the direct replacement decision for penka_group_predictions.csv."""
    overall = comparison[comparison["scope"].eq("overall_group_stage")].set_index("strategy")
    years = comparison[~comparison["scope"].eq("overall_group_stage")].pivot(
        index="scope",
        columns="strategy",
        values="total_penka_points",
    )
    production = overall.loc["penka_group_strategy"]
    favorite = overall.loc["favorite_1_0"]
    years_beaten = [year for year in years.index if years.loc[year, "penka_group_strategy"] > years.loc[year, "favorite_1_0"]]
    beats_overall = bool(production["average_penka_points"] > favorite["average_penka_points"])
    positive_ci = bool(production["ci_lower_95_vs_favorite_1_0"] > 0)
    replace = beats_overall and positive_ci and len(years_beaten) >= 2
    lines = [
        "Penka Group Strategy Replacement Summary",
        "========================================",
        "Strategy under test: selected pre-tournament lambda calibration -> independent Poisson matrix",
        "-> real group-phase Penka EV optimizer with max_goals=5, regular score caps, and exact-score tie-break window 0.03.",
        "",
        f"favorite_1_0_average_penka_points: {favorite['average_penka_points']:.6f}",
        f"penka_group_strategy_average_penka_points: {production['average_penka_points']:.6f}",
        f"penka_group_strategy_difference_vs_favorite_1_0: {production['difference_vs_favorite_1_0']:.6f}",
        f"paired_bootstrap_95_ci_vs_favorite_1_0: [{production['ci_lower_95_vs_favorite_1_0']:.6f}, {production['ci_upper_95_vs_favorite_1_0']:.6f}]",
        f"world_cups_beating_favorite_1_0: {len(years_beaten)}/3 ({', '.join(years_beaten) if years_beaten else 'none'})",
        f"beats_favorite_1_0_overall: {beats_overall}",
        f"positive_bootstrap_ci: {positive_ci}",
        f"replace_pure_favorite_1_0_baseline: {replace}",
        "",
        "Decision: keep oriented favorite_1_0 as the validated default submission baseline."
        if not replace
        else "Decision: penka_group_strategy clears the replacement gate.",
        "Use penka_group_predictions.csv as a refreshable analytical review file, not an automatic final submission file.",
    ]
    path = output_dir / "penka_strategy_summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def summarize_strategy_scope(
    points: pd.DataFrame,
    match_indexes: pd.Index,
    scope: str,
    bucket_type: str = "all",
    bucket_value: str = "all",
) -> pd.DataFrame:
    """Summarize strategies and paired bootstrap differences for one match subset."""
    selected = points[points["match_index"].isin(match_indexes)].copy()
    pivot = selected.pivot(index="match_index", columns="strategy", values="penka_points")
    rows = []
    for strategy, strategy_rows in selected.groupby("strategy", sort=False):
        categories = strategy_rows["penka_category"]
        points_values = strategy_rows["penka_points"].to_numpy(dtype=float)
        ci_vs_favorite = _paired_bootstrap_ci(points_values - pivot.loc[strategy_rows["match_index"], "favorite_1_0"].to_numpy())
        ci_vs_optimizer = _paired_bootstrap_ci(points_values - pivot.loc[strategy_rows["match_index"], "calibrated_optimizer"].to_numpy())
        rows.append(
            {
                "scope": scope,
                "bucket_name": bucket_type,
                "bucket_value": str(bucket_value),
                "strategy": strategy,
                "match_count": len(strategy_rows),
                "average_penka_points": float(points_values.mean()),
                "total_penka_points": int(points_values.sum()),
                "exact_score_rate": float(categories.eq("exact_score").mean()),
                "correct_goal_difference_or_draw_rate": float(categories.eq("correct_goal_difference_or_draw").mean()),
                "correct_winner_only_rate": float(categories.eq("correct_winner").mean()),
                "wrong_rate": float(categories.eq("wrong").mean()),
                "difference_vs_favorite_1_0": float(
                    points_values.mean() - pivot.loc[strategy_rows["match_index"], "favorite_1_0"].mean()
                ),
                "ci_lower_95_vs_favorite_1_0": ci_vs_favorite[0],
                "ci_upper_95_vs_favorite_1_0": ci_vs_favorite[1],
                "difference_vs_calibrated_optimizer": float(
                    points_values.mean() - pivot.loc[strategy_rows["match_index"], "calibrated_optimizer"].mean()
                ),
                "ci_lower_95_vs_calibrated_optimizer": ci_vs_optimizer[0],
                "ci_upper_95_vs_calibrated_optimizer": ci_vs_optimizer[1],
            }
        )
    return pd.DataFrame(rows)


def build_strategy_reports(context: pd.DataFrame, points: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build overall, phase, year, and leakage-safe bucket Penka comparisons."""
    all_indexes = context.index
    overall = summarize_strategy_scope(points, all_indexes, "overall_historical_world_cup")
    group = summarize_strategy_scope(points, context.index[context["penka_phase"].eq(GROUP)], "group_stage_only")
    knockout = summarize_strategy_scope(points, context.index[context["penka_phase"].ne(GROUP)], "knockout_only")
    by_year = pd.concat(
        [
            summarize_strategy_scope(points, rows.index, "world_cup_year", "year", str(year))
            for year, rows in context.groupby("backtest_year")
        ],
        ignore_index=True,
    )
    bucket_parts = []
    dimensions = {
        "rank_gap_bucket": "rank_gap_bucket",
        "favorite_win_probability_bucket": "favorite_win_probability_bucket",
        "high_mismatch": "high_mismatch",
    }
    if "elo_bucket" in context.columns:
        dimensions["elo_gap_bucket"] = "elo_bucket"
    for bucket_name, column in dimensions.items():
        for bucket_value, rows in context.groupby(column, observed=True, dropna=False):
            bucket_parts.append(
                summarize_strategy_scope(points, rows.index, "strategy_bucket", bucket_name, str(bucket_value))
            )
    return {
        "penka_scoring_strategy_comparison": overall,
        "penka_group_strategy_comparison": group,
        "penka_knockout_strategy_comparison": knockout,
        "penka_strategy_by_year": by_year,
        "penka_strategy_by_bucket": pd.concat(bucket_parts, ignore_index=True),
    }


def build_penka_calibration_comparison(backtests: pd.DataFrame, max_goals: int = 6) -> pd.DataFrame:
    """Compare lambda mappings by real Penka points, calibration, and common score modes."""
    methods = {
        "uncalibrated": ("uncalibrated_expected_goals_a", "uncalibrated_expected_goals_b"),
        "linear": ("linear_expected_goals_a", "linear_expected_goals_b"),
        "isotonic": ("isotonic_expected_goals_a", "isotonic_expected_goals_b"),
    }
    raw_total = backtests["raw_expected_goals_a"] + backtests["raw_expected_goals_b"]
    top_decile = raw_total.ge(raw_total.quantile(0.90))
    actual_total = backtests["goals_a"] + backtests["goals_b"]
    rows = []
    for method, (lambda_a_column, lambda_b_column) in methods.items():
        optimized = _poisson_optimizer_predictions(backtests, lambda_a_column, lambda_b_column, max_goals=max_goals)
        scored = _score_strategy(backtests, method, optimized)
        predicted_total = backtests[lambda_a_column] + backtests[lambda_b_column]
        row = {
            "calibration_method": method,
            "matches": len(backtests),
            "average_penka_points": float(scored["penka_points"].mean()),
            "exact_score_rate": float(scored["penka_category"].eq("exact_score").mean()),
            "correct_goal_difference_or_draw_rate": float(
                scored["penka_category"].eq("correct_goal_difference_or_draw").mean()
            ),
            "correct_winner_only_rate": float(scored["penka_category"].eq("correct_winner").mean()),
            "wrong_rate": float(scored["penka_category"].eq("wrong").mean()),
            "predicted_mean_total_goals": float(predicted_total.mean()),
            "actual_mean_total_goals": float(actual_total.mean()),
            "top_lambda_decile_predicted_mean_total_goals": float(predicted_total[top_decile].mean()),
            "top_lambda_decile_actual_mean_total_goals": float(actual_total[top_decile].mean()),
        }
        matrices = [
            score_probability_matrix(float(lambda_a), float(lambda_b), max_goals=max_goals)
            for lambda_a, lambda_b in zip(backtests[lambda_a_column], backtests[lambda_b_column])
        ]
        for goals_a, goals_b in COMMON_SCORES:
            label = f"{goals_a}_{goals_b}"
            row[f"predicted_probability_{label}"] = float(np.mean([matrix[goals_a, goals_b] for matrix in matrices]))
            row[f"actual_rate_{label}"] = float(
                ((backtests["goals_a"] == goals_a) & (backtests["goals_b"] == goals_b)).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("average_penka_points", ascending=False, kind="stable")


def write_penka_summary(reports: dict[str, pd.DataFrame], calibration: pd.DataFrame, output_dir: Path) -> Path:
    """Write the blunt strategy decision questions requested for Penka."""
    overall = reports["penka_scoring_strategy_comparison"].set_index("strategy")
    group = reports["penka_group_strategy_comparison"].set_index("strategy")
    knockout = reports["penka_knockout_strategy_comparison"].set_index("strategy")
    bucket = reports["penka_strategy_by_bucket"]
    mismatch = bucket[(bucket["bucket_name"].eq("high_mismatch")) & (bucket["bucket_value"].eq("True"))]
    mismatch = mismatch.set_index("strategy")
    best_overall = overall["average_penka_points"].idxmax()
    best_group = group["average_penka_points"].idxmax()
    best_knockout = knockout["average_penka_points"].idxmax()
    two_nil_mismatch = (
        mismatch.loc["favorite_2_0", "average_penka_points"] > mismatch.loc["favorite_1_0", "average_penka_points"]
        if {"favorite_2_0", "favorite_1_0"}.issubset(mismatch.index)
        else False
    )
    draw_knockout_best = best_knockout in {"draw_0_0", "draw_1_1"}
    lines = [
        "Real Penka Scoring Strategy Summary",
        "===================================",
        "Scoring is phase-specific and mutually exclusive. Knockout predictions are 90-minute scores and may be draws.",
        "",
        f"best_overall_strategy: {best_overall}",
        f"best_group_stage_strategy: {best_group}",
        f"best_knockout_sanity_check_strategy: {best_knockout}",
        f"favorite_1_0_overall_average_penka_points: {overall.loc['favorite_1_0', 'average_penka_points']:.6f}",
        f"calibrated_optimizer_overall_average_penka_points: {overall.loc['calibrated_optimizer', 'average_penka_points']:.6f}",
        f"Under real Penka scoring, does favorite_1_0 still beat the optimizer? {overall.loc['favorite_1_0', 'average_penka_points'] > overall.loc['calibrated_optimizer', 'average_penka_points']}",
        f"Under real Penka group scoring, does favorite_1_0 still win group-stage predictions? {best_group == 'favorite_1_0'}",
        f"Does 2-0 become better in the broad high-mismatch bucket? {two_nil_mismatch}",
        "Does 2-1 become better in moderate-favorite/high-total-goals buckets? Review penka_strategy_by_bucket.csv; do not automate from sparse slices.",
        f"Do draws become valuable in close/knockout buckets? knockout sanity-check winner is draw: {draw_knockout_best}",
        f"recommended_group_stage_strategy: {best_group}",
        "recommended_knockout_strategy: EV-maximize each confirmed fixture analytically with its Penka phase; historical knockout rows are diagnostic only.",
        "Should the old favorite_1_0 everywhere conclusion be discarded? Yes. It was derived under the wrong points formula.",
        "",
        f"best_calibration_under_penka_backtest: {calibration.iloc[0]['calibration_method']}",
    ]
    path = output_dir / "penka_scoring_summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_penka_backtesting_reports(
    backtests: pd.DataFrame,
    output_dir: str | Path = "outputs",
    max_goals: int = 6,
) -> dict[str, pd.DataFrame]:
    """Generate authoritative Penka strategy and lambda-calibration reports."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    excluded = int(backtests.get("penka_phase", pd.Series(UNKNOWN_PHASE, index=backtests.index)).eq(UNKNOWN_PHASE).sum())
    context, points = build_strategy_points(backtests, max_goals=max_goals)
    reports = build_strategy_reports(context, points)
    for name, report in reports.items():
        report.to_csv(output_path / f"{name}.csv", index=False)
    calibration = build_penka_calibration_comparison(context, max_goals=max_goals)
    calibration.to_csv(output_path / "penka_calibration_comparison.csv", index=False)
    calibration_summary = [
        "Penka Lambda Calibration Summary",
        "================================",
        "Calibration is compared using real mutually exclusive Penka points, exact-score rate, category rates,",
        "overall mean total goals, top-lambda decile behavior, and modal score calibration.",
        f"excluded_unknown_phase_rows: {excluded}",
        f"best_method_by_average_penka_points: {calibration.iloc[0]['calibration_method']}",
    ]
    (output_path / "penka_calibration_summary.txt").write_text("\n".join(calibration_summary) + "\n", encoding="utf-8")
    write_penka_summary(reports, calibration, output_path)
    focused_group = build_focused_group_strategy_comparison(backtests)
    focused_group.to_csv(output_path / "penka_strategy_backtest_comparison.csv", index=False)
    write_focused_group_strategy_summary(focused_group, output_path)
    return {
        **reports,
        "penka_calibration_comparison": calibration,
        "penka_strategy_backtest_comparison": focused_group,
    }
