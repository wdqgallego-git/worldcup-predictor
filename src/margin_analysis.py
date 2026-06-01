"""Full-history empirical winning-margin analysis for ranking favorites."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features import build_training_table
from poisson import score_probability_matrix
from prediction_optimizer import points_for_prediction


MIN_NEUTRAL_SAMPLE_MATCHES = 100
HIGH_DRAW_PROBABILITY = 0.30
MODEL_SAFE_MAX_GOALS = 6

RANK_BUCKETS = {
    "bins": [0, 10, 25, 40, 60, 80, np.inf],
    "labels": ["0_10", "10_25", "25_40", "40_60", "60_80", "80_plus"],
}
PROBABILITY_BUCKETS = {
    "bins": [0.40, 0.50, 0.60, 0.70, 0.80, np.inf],
    "labels": ["0.40_0.50", "0.50_0.60", "0.60_0.70", "0.70_0.80", "0.80_plus"],
}
XG_DIFF_BUCKETS = {
    "bins": [0.0, 0.25, 0.50, 0.75, 1.00, 1.50, np.inf],
    "labels": ["0.0_0.25", "0.25_0.50", "0.50_0.75", "0.75_1.00", "1.00_1.50", "1.50_plus"],
}
FIXED_SCORE_RULES = {
    "favorite_1_0": (1, 0),
    "favorite_2_0": (2, 0),
    "favorite_3_0": (3, 0),
    "favorite_2_1": (2, 1),
}


def _challenge_points_array(pred_favorite: int, pred_underdog: int, matches: pd.DataFrame) -> np.ndarray:
    """Return company challenge points for one oriented fixed-score rule."""
    return np.asarray(
        [
            points_for_prediction(pred_favorite, pred_underdog, int(actual_favorite), int(actual_underdog))
            for actual_favorite, actual_underdog in zip(matches["favorite_goals"], matches["underdog_goals"])
        ],
        dtype=int,
    )


def _score_utility_matrix(max_goals: int = MODEL_SAFE_MAX_GOALS) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Precompute challenge-point utilities for fast historical optimizer scoring."""
    scores = [(goals_a, goals_b) for goals_a in range(max_goals + 1) for goals_b in range(max_goals + 1)]
    utilities = np.asarray(
        [
            [
                [
                    points_for_prediction(pred_a, pred_b, actual_a, actual_b)
                    for actual_b in range(max_goals + 1)
                ]
                for actual_a in range(max_goals + 1)
            ]
            for pred_a, pred_b in scores
        ],
        dtype=float,
    )
    return scores, utilities


def add_leakage_safe_severity_features(training_df: pd.DataFrame) -> pd.DataFrame:
    """Add strictly pre-match heuristic lambdas and ranking-favorite severity features."""
    matches = training_df[
        training_df["goals_a"].notna()
        & training_df["goals_b"].notna()
        & np.isfinite(training_df["goals_a"])
        & np.isfinite(training_df["goals_b"])
    ].sort_values("date", kind="stable").copy()
    prior_mean_goals = (
        (matches["goals_a"] + matches["goals_b"])
        .shift(1)
        .expanding(min_periods=1)
        .mean()
        .div(2.0)
        .fillna(1.20)
    )
    home_edge = np.where(matches["neutral"].fillna(False).astype(bool), 0.0, 0.10)
    matches["heuristic_expected_goals_a"] = np.clip(
        0.45 * prior_mean_goals
        + 0.35 * matches["team_a_recent_goals_scored_5"]
        + 0.20 * matches["team_b_recent_goals_conceded_5"]
        + home_edge,
        0.05,
        6.0,
    )
    matches["heuristic_expected_goals_b"] = np.clip(
        0.45 * prior_mean_goals
        + 0.35 * matches["team_b_recent_goals_scored_5"]
        + 0.20 * matches["team_a_recent_goals_conceded_5"],
        0.05,
        6.0,
    )
    matches = matches[
        matches["ranking_date_a"].notna()
        & matches["ranking_date_b"].notna()
        & matches["rank_a"].notna()
        & matches["rank_b"].notna()
        & matches["rank_a"].ne(matches["rank_b"])
    ].copy()
    matches["favorite_is_a"] = matches["rank_a"] < matches["rank_b"]
    matches["rank_diff_abs"] = (matches["rank_a"] - matches["rank_b"]).abs()
    matches["favorite_goals"] = np.where(matches["favorite_is_a"], matches["goals_a"], matches["goals_b"])
    matches["underdog_goals"] = np.where(matches["favorite_is_a"], matches["goals_b"], matches["goals_a"])
    matches["favorite_margin"] = matches["favorite_goals"] - matches["underdog_goals"]
    matches["actual_total_goals"] = matches["goals_a"] + matches["goals_b"]
    matches["expected_goal_difference"] = (
        matches["heuristic_expected_goals_a"] - matches["heuristic_expected_goals_b"]
    ).abs()
    matches["expected_total_goals"] = (
        matches["heuristic_expected_goals_a"] + matches["heuristic_expected_goals_b"]
    )
    matches["year"] = matches["date"].dt.year
    tournament = matches["tournament"].fillna("").astype(str)
    matches["is_friendly"] = tournament.str.contains("friendly", case=False, regex=False)
    matches["is_world_cup"] = tournament.eq("FIFA World Cup")

    scores, utilities = _score_utility_matrix()
    favorite_win_probabilities = []
    draw_probabilities = []
    safe_favorite_goals = []
    safe_underdog_goals = []
    for match in matches.itertuples(index=False):
        probabilities = score_probability_matrix(
            match.heuristic_expected_goals_a,
            match.heuristic_expected_goals_b,
            max_goals=MODEL_SAFE_MAX_GOALS,
        )
        favorite_win_probability = (
            float(np.tril(probabilities, -1).sum())
            if match.favorite_is_a
            else float(np.triu(probabilities, 1).sum())
        )
        favorite_win_probabilities.append(favorite_win_probability)
        draw_probabilities.append(float(np.trace(probabilities)))
        expected_points = (utilities * probabilities).sum(axis=(1, 2))
        safe_a, safe_b = scores[int(np.argmax(expected_points))]
        safe_favorite_goals.append(safe_a if match.favorite_is_a else safe_b)
        safe_underdog_goals.append(safe_b if match.favorite_is_a else safe_a)
    matches["favorite_win_probability"] = favorite_win_probabilities
    matches["draw_probability"] = draw_probabilities
    matches["model_safe_favorite_goals"] = safe_favorite_goals
    matches["model_safe_underdog_goals"] = safe_underdog_goals
    matches["favorite_result"] = np.select(
        [
            matches["favorite_margin"] < 0,
            matches["favorite_margin"] == 0,
            matches["favorite_margin"] == 1,
            matches["favorite_margin"] == 2,
            matches["favorite_margin"] >= 3,
        ],
        ["favorite_loss", "draw", "favorite_win_by_1", "favorite_win_by_2", "favorite_win_by_3_plus"],
        default="unknown",
    )
    return matches


def build_analysis_samples(matches: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return robustness samples used for the full-history comparison."""
    since_2000 = matches[matches["date"] >= pd.Timestamp("2000-01-01")].copy()
    samples = {
        "all_matches_since_2000": since_2000,
        "competitive_since_2000": since_2000[~since_2000["is_friendly"]].copy(),
        "all_matches_since_2010": since_2000[since_2000["date"] >= pd.Timestamp("2010-01-01")].copy(),
        "world_cup_reference": since_2000[since_2000["is_world_cup"]].copy(),
    }
    neutral = since_2000[since_2000["neutral"].fillna(False).astype(bool)].copy()
    if len(neutral) >= MIN_NEUTRAL_SAMPLE_MATCHES:
        samples["neutral_site_since_2000"] = neutral
    return samples


def summarize_margin_distribution(rows: pd.DataFrame, sample_name: str, bucket_name: str, bucket_value: str) -> dict[str, object]:
    """Summarize empirical favorite winning margins for one bucket."""
    margins = rows["favorite_margin"]
    return {
        "sample": sample_name,
        "bucket_name": bucket_name,
        "bucket_value": bucket_value,
        "match_count": len(rows),
        "favorite_win_rate": float(margins.gt(0).mean()),
        "draw_rate": float(margins.eq(0).mean()),
        "upset_rate": float(margins.lt(0).mean()),
        "favorite_win_by_1_rate": float(margins.eq(1).mean()),
        "favorite_win_by_2_rate": float(margins.eq(2).mean()),
        "favorite_win_by_3_plus_rate": float(margins.ge(3).mean()),
        "avg_favorite_margin": float(margins.mean()),
        "avg_total_goals": float(rows["actual_total_goals"].mean()),
    }


def bucket_sample(rows: pd.DataFrame, column: str, bucket_spec: dict[str, object]) -> pd.Series:
    """Return stable left-inclusive severity labels."""
    return pd.cut(
        rows[column],
        bins=bucket_spec["bins"],
        labels=bucket_spec["labels"],
        right=False,
        include_lowest=True,
    )


def build_distribution_report(
    samples: dict[str, pd.DataFrame],
    column: str,
    bucket_name: str,
    bucket_spec: dict[str, object],
) -> pd.DataFrame:
    """Build one empirical margin-distribution report."""
    report_rows = []
    for sample_name, sample in samples.items():
        bucketed = sample.copy()
        bucketed["_bucket"] = bucket_sample(bucketed, column, bucket_spec)
        for bucket_value, rows in bucketed.groupby("_bucket", observed=True):
            report_rows.append(summarize_margin_distribution(rows, sample_name, bucket_name, str(bucket_value)))
    return pd.DataFrame(report_rows)


def summarize_fixed_score_rule(
    rows: pd.DataFrame,
    sample_name: str,
    bucket_name: str,
    bucket_value: str,
    rule: str,
) -> dict[str, object]:
    """Summarize empirical company-game EV for one score-prediction rule."""
    if rule == "model_safe_prediction":
        points = np.asarray(
            [
                points_for_prediction(int(pred_favorite), int(pred_underdog), int(actual_favorite), int(actual_underdog))
                for pred_favorite, pred_underdog, actual_favorite, actual_underdog in zip(
                    rows["model_safe_favorite_goals"],
                    rows["model_safe_underdog_goals"],
                    rows["favorite_goals"],
                    rows["underdog_goals"],
                )
            ],
            dtype=int,
        )
        pred_favorite = rows["model_safe_favorite_goals"]
        pred_underdog = rows["model_safe_underdog_goals"]
    else:
        score = (1, 1) if rule == "favorite_1_1_high_draw_only" else FIXED_SCORE_RULES[rule]
        points = _challenge_points_array(*score, rows)
        pred_favorite = pd.Series(score[0], index=rows.index)
        pred_underdog = pd.Series(score[1], index=rows.index)
    actual_favorite = rows["favorite_goals"]
    actual_underdog = rows["underdog_goals"]
    predicted_margin = pred_favorite - pred_underdog
    actual_margin = actual_favorite - actual_underdog
    predicted_result = np.sign(predicted_margin)
    actual_result = np.sign(actual_margin)
    return {
        "sample": sample_name,
        "bucket_name": bucket_name,
        "bucket_value": bucket_value,
        "rule": rule,
        "match_count": len(rows),
        "avg_challenge_points": float(points.mean()),
        "exact_score_rate": float(((pred_favorite == actual_favorite) & (pred_underdog == actual_underdog)).mean()),
        "correct_result_rate": float((predicted_result == actual_result).mean()),
        "correct_goal_difference_rate": float((predicted_margin == actual_margin).mean()),
    }


def build_fixed_score_ev_report(samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Report fixed-score empirical EV across severity buckets and robustness samples."""
    dimensions = {
        "rank_diff_abs": ("rank_diff_abs", RANK_BUCKETS),
        "favorite_win_probability": ("favorite_win_probability", PROBABILITY_BUCKETS),
        "expected_goal_difference": ("expected_goal_difference", XG_DIFF_BUCKETS),
    }
    report_rows = []
    standard_rules = [*FIXED_SCORE_RULES, "model_safe_prediction"]
    for sample_name, sample in samples.items():
        for bucket_name, (column, bucket_spec) in dimensions.items():
            bucketed = sample.copy()
            bucketed["_bucket"] = bucket_sample(bucketed, column, bucket_spec)
            for bucket_value, rows in bucketed.groupby("_bucket", observed=True):
                for rule in standard_rules:
                    report_rows.append(
                        summarize_fixed_score_rule(rows, sample_name, bucket_name, str(bucket_value), rule)
                    )
        high_draw = sample[sample["draw_probability"] >= HIGH_DRAW_PROBABILITY]
        if not high_draw.empty:
            for rule in [*standard_rules, "favorite_1_1_high_draw_only"]:
                report_rows.append(
                    summarize_fixed_score_rule(
                        high_draw,
                        sample_name,
                        "high_draw_probability",
                        f"{HIGH_DRAW_PROBABILITY:.2f}_plus",
                        rule,
                    )
                )
    return pd.DataFrame(report_rows)


def find_rule_advantage(
    ev_report: pd.DataFrame,
    challenger: str,
    min_matches: int = 50,
) -> pd.DataFrame:
    """Return buckets where a challenger beats favorite_1_0 with enough history."""
    fixed = ev_report[
        ev_report["rule"].isin(["favorite_1_0", challenger])
        & ev_report["bucket_name"].ne("high_draw_probability")
    ]
    pivot = fixed.pivot_table(
        index=["sample", "bucket_name", "bucket_value", "match_count"],
        columns="rule",
        values="avg_challenge_points",
    ).reset_index()
    pivot["point_advantage"] = pivot[challenger] - pivot["favorite_1_0"]
    return pivot[(pivot["match_count"] >= min_matches) & (pivot["point_advantage"] > 0)].sort_values(
        "point_advantage",
        ascending=False,
        kind="stable",
    )


def find_stable_rank_threshold(ev_report: pd.DataFrame) -> str | None:
    """Return a robust rank-gap threshold only when 2-0 repeatedly beats 1-0."""
    core_samples = {"all_matches_since_2000", "competitive_since_2000", "all_matches_since_2010"}
    rank_order = ["40_60", "60_80", "80_plus"]
    fixed = ev_report[
        ev_report["sample"].isin(core_samples)
        & ev_report["bucket_name"].eq("rank_diff_abs")
        & ev_report["bucket_value"].isin(rank_order)
        & ev_report["rule"].isin(["favorite_1_0", "favorite_2_0"])
    ]
    pivot = fixed.pivot_table(
        index=["sample", "bucket_value", "match_count"],
        columns="rule",
        values="avg_challenge_points",
    ).reset_index()
    pivot["two_nil_advantage"] = pivot["favorite_2_0"] - pivot["favorite_1_0"]
    for threshold in rank_order:
        eligible_labels = rank_order[rank_order.index(threshold) :]
        rows = pivot[pivot["bucket_value"].isin(eligible_labels) & pivot["match_count"].ge(50)]
        if not rows.empty and rows["sample"].nunique() >= 3 and rows["two_nil_advantage"].gt(0).all():
            return threshold.split("_")[0]
    return None


def find_robust_advantage_buckets(
    ev_report: pd.DataFrame,
    challenger: str,
    min_matches: int = 50,
) -> pd.DataFrame:
    """Return severity buckets where a challenger wins across the three core samples."""
    core_samples = {"all_matches_since_2000", "competitive_since_2000", "all_matches_since_2010"}
    fixed = ev_report[
        ev_report["sample"].isin(core_samples)
        & ev_report["rule"].isin(["favorite_1_0", challenger])
    ]
    pivot = fixed.pivot_table(
        index=["sample", "bucket_name", "bucket_value", "match_count"],
        columns="rule",
        values="avg_challenge_points",
    ).reset_index()
    pivot["point_advantage"] = pivot[challenger] - pivot["favorite_1_0"]
    eligible = pivot[pivot["match_count"].ge(min_matches)]
    rows = []
    for (bucket_name, bucket_value), bucket_rows in eligible.groupby(["bucket_name", "bucket_value"]):
        if bucket_rows["sample"].nunique() == len(core_samples) and bucket_rows["point_advantage"].gt(0).all():
            rows.append(
                {
                    "bucket_name": bucket_name,
                    "bucket_value": bucket_value,
                    "minimum_match_count": int(bucket_rows["match_count"].min()),
                    "minimum_point_advantage": float(bucket_rows["point_advantage"].min()),
                    "maximum_point_advantage": float(bucket_rows["point_advantage"].max()),
                }
            )
    return pd.DataFrame(rows)


def write_fixed_score_summary(
    samples: dict[str, pd.DataFrame],
    ev_report: pd.DataFrame,
    path: str | Path,
) -> None:
    """Save the empirical fixed-score decision report."""
    two_nil_advantages = find_rule_advantage(ev_report, "favorite_2_0")
    three_nil_advantages = find_rule_advantage(ev_report, "favorite_3_0")
    model_advantages = find_rule_advantage(ev_report, "model_safe_prediction")
    robust_two_nil = find_robust_advantage_buckets(ev_report, "favorite_2_0")
    robust_three_nil = find_robust_advantage_buckets(ev_report, "favorite_3_0")
    robust_model_safe = find_robust_advantage_buckets(ev_report, "model_safe_prediction")
    robust_draw = find_robust_advantage_buckets(ev_report, "favorite_1_1_high_draw_only")
    stable_threshold = find_stable_rank_threshold(ev_report)
    high_draw = ev_report[
        ev_report["bucket_name"].eq("high_draw_probability")
        & ev_report["rule"].eq("favorite_1_1_high_draw_only")
    ]
    if stable_threshold is not None:
        decision = f"B. Use favorite_1_0 with favorite_2_0 when rank_diff_abs >= {stable_threshold}."
    else:
        decision = "A. Use favorite_1_0 everywhere."
    lines = [
        "Full-History Fixed-Score Rule Summary",
        "=====================================",
        "Scoring note: non-World-Cup full-history rows use the real Penka group-stage scoring approximation.",
        "Favorite orientation: latest FIFA ranking strictly before each match.",
        "Expected-goal severity: transparent heuristic using only prior expanding goal rates and pre-match recent form.",
        "",
        "Robustness sample match counts:",
    ]
    lines.extend(f"- {sample_name}: {len(rows)}" for sample_name, rows in samples.items())
    lines.extend(
        [
            "",
            f"Does favorite_2_0 beat favorite_1_0 in any bucket with >= 50 matches? {not two_nil_advantages.empty}",
            f"Does favorite_3_0 beat favorite_1_0 in any bucket with >= 50 matches? {not three_nil_advantages.empty}",
            f"Stable rank-gap threshold where 2-0 should replace 1-0: {stable_threshold or 'None'}",
            f"Model-safe prediction beats favorite_1_0 in any >= 50-match bucket: {not model_advantages.empty}",
            f"Robust 2-0 severity buckets across the three core samples: {len(robust_two_nil)}",
            f"Robust 3-0 severity buckets across the three core samples: {len(robust_three_nil)}",
            f"Robust model-safe severity buckets across the three core samples: {len(robust_model_safe)}",
            f"Does 1-1 beat favorite_1_0 robustly in the high-draw sample? {not robust_draw.empty}",
            f"High-draw reference rows reported: {int(high_draw['match_count'].sum()) if not high_draw.empty else 0}",
            "",
            f"Recommendation: {decision}",
            "C. Do not switch to the model-safe heuristic automatically; it is diagnostic rather than the production calibrated optimizer.",
            "D. Use 1-1 only if its high-draw comparison is robustly positive across core samples.",
            "No final_match_recommendations files were changed automatically.",
            "",
            "Top favorite_2_0 advantages with >= 50 matches:",
            two_nil_advantages.head(10).to_string(index=False) if not two_nil_advantages.empty else "None",
            "",
            "Top favorite_3_0 advantages with >= 50 matches:",
            three_nil_advantages.head(10).to_string(index=False) if not three_nil_advantages.empty else "None",
            "",
            "Robust favorite_2_0 severity buckets across core samples:",
            robust_two_nil.to_string(index=False) if not robust_two_nil.empty else "None",
            "",
            "Robust favorite_3_0 severity buckets across core samples:",
            robust_three_nil.to_string(index=False) if not robust_three_nil.empty else "None",
            "",
            "Robust high-draw 1-1 comparison across core samples:",
            robust_draw.to_string(index=False) if not robust_draw.empty else "None",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_margin_distribution_analysis(
    results: pd.DataFrame,
    rankings: pd.DataFrame,
    output_dir: str | Path = "outputs",
) -> dict[str, pd.DataFrame]:
    """Run full-history margin and fixed-score EV analysis and save outputs."""
    training_df, _ = build_training_table(results, rankings, start_date="2000-01-01")
    matches = add_leakage_safe_severity_features(training_df)
    samples = build_analysis_samples(matches)
    rank_report = build_distribution_report(samples, "rank_diff_abs", "rank_diff_abs", RANK_BUCKETS)
    probability_report = build_distribution_report(
        samples,
        "favorite_win_probability",
        "favorite_win_probability",
        PROBABILITY_BUCKETS,
    )
    xgdiff_report = build_distribution_report(
        samples,
        "expected_goal_difference",
        "expected_goal_difference",
        XG_DIFF_BUCKETS,
    )
    fixed_score_report = build_fixed_score_ev_report(samples)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rank_report.to_csv(output_path / "margin_distribution_by_rank_bucket.csv", index=False)
    probability_report.to_csv(output_path / "margin_distribution_by_probability_bucket.csv", index=False)
    xgdiff_report.to_csv(output_path / "margin_distribution_by_xgdiff_bucket.csv", index=False)
    fixed_score_report.to_csv(output_path / "fixed_score_ev_by_bucket.csv", index=False)
    write_fixed_score_summary(samples, fixed_score_report, output_path / "fixed_score_rule_summary.txt")
    sample_counts = pd.DataFrame(
        [{"sample": sample_name, "match_count": len(rows)} for sample_name, rows in samples.items()]
    )
    return {
        "margin_distribution_by_rank_bucket": rank_report,
        "margin_distribution_by_probability_bucket": probability_report,
        "margin_distribution_by_xgdiff_bucket": xgdiff_report,
        "fixed_score_ev_by_bucket": fixed_score_report,
        "sample_counts": sample_counts,
    }
