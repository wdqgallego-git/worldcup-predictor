"""Advisory-only verification for a narrow favorite 2-0 score rule."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import load_fixtures
from features import add_ranking_features, build_training_table, prepare_match_table
from prediction_optimizer import points_for_prediction


ELO_INITIAL_RATING = 1500.0
ELO_K_FACTOR = 20.0
BOOTSTRAP_SAMPLES = 5000
ORIGINAL_RANK_THRESHOLD = 80.0
TRAINING_RANK_THRESHOLDS = [40, 50, 60, 70, 80, 90, 100]
MIN_BUCKET_MATCHES = 50
TRAINING_END = pd.Timestamp("2014-12-31")
HOLDOUT_START = pd.Timestamp("2015-01-01")
HOLDOUT_END = pd.Timestamp("2025-12-31")


def expected_elo_score(rating_a: float, rating_b: float) -> float:
    """Return Team A Elo win expectation before a match."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def add_causal_elo_features(results: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Store pre-match Elo values and update ratings only after each result."""
    matches = prepare_match_table(results)
    ratings: defaultdict[str, float] = defaultdict(lambda: ELO_INITIAL_RATING)
    rows = []
    for match in matches.itertuples(index=False):
        pre_elo_a = float(ratings[str(match.team_a)])
        pre_elo_b = float(ratings[str(match.team_b)])
        rows.append({"pre_elo_a": pre_elo_a, "pre_elo_b": pre_elo_b})
        if pd.isna(match.goals_a) or pd.isna(match.goals_b):
            continue
        if not np.isfinite(float(match.goals_a)) or not np.isfinite(float(match.goals_b)):
            continue
        actual_a = 1.0 if match.goals_a > match.goals_b else 0.5 if match.goals_a == match.goals_b else 0.0
        expected_a = expected_elo_score(pre_elo_a, pre_elo_b)
        adjustment = ELO_K_FACTOR * (actual_a - expected_a)
        ratings[str(match.team_a)] = pre_elo_a + adjustment
        ratings[str(match.team_b)] = pre_elo_b - adjustment
    return pd.concat([matches, pd.DataFrame(rows, index=matches.index)], axis=1), dict(ratings)


def latest_causal_elo_before_date(results: pd.DataFrame, cutoff_date: pd.Timestamp) -> dict[str, float]:
    """Return causal Elo state using only completed matches before cutoff_date."""
    matches = prepare_match_table(results)
    ratings: defaultdict[str, float] = defaultdict(lambda: ELO_INITIAL_RATING)
    for match in matches[matches["date"] < pd.Timestamp(cutoff_date)].itertuples(index=False):
        if pd.isna(match.goals_a) or pd.isna(match.goals_b):
            continue
        if not np.isfinite(float(match.goals_a)) or not np.isfinite(float(match.goals_b)):
            continue
        pre_elo_a = float(ratings[str(match.team_a)])
        pre_elo_b = float(ratings[str(match.team_b)])
        actual_a = 1.0 if match.goals_a > match.goals_b else 0.5 if match.goals_a == match.goals_b else 0.0
        adjustment = ELO_K_FACTOR * (actual_a - expected_elo_score(pre_elo_a, pre_elo_b))
        ratings[str(match.team_a)] = pre_elo_a + adjustment
        ratings[str(match.team_b)] = pre_elo_b - adjustment
    return dict(ratings)


def attach_causal_elo_to_training_table(training_df: pd.DataFrame, elo_matches: pd.DataFrame) -> pd.DataFrame:
    """Attach Elo features to the chronological training rows without a future-looking join."""
    eligible_elo = elo_matches[elo_matches["date"] >= pd.Timestamp("2000-01-01")].reset_index(drop=True)
    training = training_df.sort_values("date", kind="stable").reset_index(drop=True).copy()
    if len(training) != len(eligible_elo):
        raise ValueError(
            "Causal Elo rows do not align with the training table. "
            f"training_rows={len(training)}, elo_rows={len(eligible_elo)}"
        )
    training["pre_elo_a"] = eligible_elo["pre_elo_a"]
    training["pre_elo_b"] = eligible_elo["pre_elo_b"]
    return training


def build_severity_axes(results: pd.DataFrame, rankings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Build leakage-safe favorite labels and structural severity axes since 2000."""
    training_df, _ = build_training_table(results, rankings, start_date="2000-01-01")
    elo_matches, latest_elo = add_causal_elo_features(results)
    matches = attach_causal_elo_to_training_table(training_df, elo_matches)
    total_rows = len(matches)
    missing_score = matches["goals_a"].isna() | matches["goals_b"].isna()
    nonfinite_score = (
        ~np.isfinite(pd.to_numeric(matches["goals_a"], errors="coerce"))
        | ~np.isfinite(pd.to_numeric(matches["goals_b"], errors="coerce"))
    )
    matches = matches[~missing_score & ~nonfinite_score].copy()
    matches["has_ordinal_rank_gap"] = (
        matches["ranking_date_a"].notna()
        & matches["ranking_date_b"].notna()
        & matches["rank_a"].notna()
        & matches["rank_b"].notna()
        & matches["rank_a"].ne(matches["rank_b"])
    )
    matches["has_ranking_points_gap"] = (
        matches["ranking_date_a"].notna()
        & matches["ranking_date_b"].notna()
        & matches["ranking_points_a"].notna()
        & matches["ranking_points_b"].notna()
    )
    matches["elo_tied"] = matches["pre_elo_a"].eq(matches["pre_elo_b"])
    matches["favorite_source"] = np.where(matches["has_ordinal_rank_gap"], "fifa_rank", "causal_elo_fallback")
    matches["favorite_is_a"] = np.where(
        matches["has_ordinal_rank_gap"],
        matches["rank_a"] < matches["rank_b"],
        matches["pre_elo_a"] > matches["pre_elo_b"],
    )
    unresolved_favorite = ~matches["has_ordinal_rank_gap"] & matches["elo_tied"]
    matches = matches[~unresolved_favorite].copy()
    matches["favorite_team"] = np.where(matches["favorite_is_a"], matches["team_a"], matches["team_b"])
    matches["underdog_team"] = np.where(matches["favorite_is_a"], matches["team_b"], matches["team_a"])
    matches["favorite_goals"] = np.where(matches["favorite_is_a"], matches["goals_a"], matches["goals_b"])
    matches["underdog_goals"] = np.where(matches["favorite_is_a"], matches["goals_b"], matches["goals_a"])
    matches["favorite_margin"] = matches["favorite_goals"] - matches["underdog_goals"]
    matches["favorite_win_by_1"] = matches["favorite_margin"].eq(1)
    matches["favorite_win_by_2"] = matches["favorite_margin"].eq(2)
    matches["favorite_win_by_3_plus"] = matches["favorite_margin"].ge(3)
    matches["draw"] = matches["favorite_margin"].eq(0)
    matches["upset"] = matches["favorite_margin"].lt(0)
    matches["favorite_result"] = np.select(
        [
            matches["upset"],
            matches["draw"],
            matches["favorite_win_by_1"],
            matches["favorite_win_by_2"],
            matches["favorite_win_by_3_plus"],
        ],
        ["favorite_loss", "draw", "favorite_win_by_1", "favorite_win_by_2", "favorite_win_by_3_plus"],
        default="unknown",
    )
    matches["ordinal_rank_gap"] = (matches["rank_a"] - matches["rank_b"]).abs().where(matches["has_ordinal_rank_gap"])
    matches["ranking_points_gap"] = (
        matches["ranking_points_a"] - matches["ranking_points_b"]
    ).abs().where(matches["has_ranking_points_gap"])
    matches["causal_elo_gap"] = (matches["pre_elo_a"] - matches["pre_elo_b"]).abs()
    matches["actual_total_goals"] = matches["goals_a"] + matches["goals_b"]
    matches["year"] = matches["date"].dt.year

    excluded_missing_score = int((missing_score | nonfinite_score).sum())
    excluded_unresolved_favorite = int(unresolved_favorite.sum())
    quality_rows = [
        {"metric": "candidate_matches_since_2000", "count": total_rows, "notes": "Rows before outcome-quality filtering."},
        {"metric": "matches_used", "count": len(matches), "notes": "Valid outcome and identifiable pre-match favorite."},
        {"metric": "matches_with_ordinal_rank_gap", "count": int(matches["ordinal_rank_gap"].notna().sum()), "notes": "Latest FIFA ranks strictly before match date."},
        {"metric": "matches_with_ranking_points_gap", "count": int(matches["ranking_points_gap"].notna().sum()), "notes": "Latest FIFA ranking points strictly before match date."},
        {"metric": "matches_with_causal_elo_gap", "count": int(matches["causal_elo_gap"].notna().sum()), "notes": "Sequential Elo stored before each result update."},
        {"metric": "favorites_selected_by_fifa_rank", "count": int(matches["favorite_source"].eq("fifa_rank").sum()), "notes": "Preferred favorite-identification source."},
        {"metric": "favorites_selected_by_causal_elo_fallback", "count": int(matches["favorite_source"].eq("causal_elo_fallback").sum()), "notes": "Used only when FIFA rank was unavailable or tied."},
        {"metric": "excluded_missing_or_nonfinite_score", "count": excluded_missing_score, "notes": "Cannot evaluate empirical winning margin."},
        {"metric": "excluded_unresolved_favorite", "count": excluded_unresolved_favorite, "notes": "FIFA rank unavailable or tied and causal Elo also tied."},
        {"metric": "model_derived_axes_skipped", "count": 2, "notes": "favorite_win_probability and expected_goal_difference skipped: full-history production walk-forward values are unavailable."},
    ]
    return matches, pd.DataFrame(quality_rows), latest_elo


def score_rule(matches: pd.DataFrame, favorite_goals: int, underdog_goals: int) -> np.ndarray:
    """Return empirical company-game points for an oriented favorite score."""
    return np.asarray(
        [
            points_for_prediction(favorite_goals, underdog_goals, int(actual_favorite), int(actual_underdog))
            for actual_favorite, actual_underdog in zip(matches["favorite_goals"], matches["underdog_goals"])
        ],
        dtype=int,
    )


def bootstrap_difference(points_2_0: np.ndarray, points_1_0: np.ndarray, random_seed: int = 2026) -> tuple[float, float]:
    """Return paired-bootstrap 95% interval for 2-0 minus 1-0 points."""
    differences = np.asarray(points_2_0, dtype=float) - np.asarray(points_1_0, dtype=float)
    if not len(differences):
        return np.nan, np.nan
    rng = np.random.default_rng(random_seed)
    indexes = rng.integers(0, len(differences), size=(BOOTSTRAP_SAMPLES, len(differences)))
    means = differences[indexes].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_extreme_bucket(matches: pd.DataFrame, axis_name: str, threshold: float) -> dict[str, object]:
    """Summarize fixed-score EV and margin mechanism above one severity threshold."""
    rows = matches[matches[axis_name].notna() & matches[axis_name].ge(threshold)].copy()
    points_1_0 = score_rule(rows, 1, 0)
    points_2_0 = score_rule(rows, 2, 0)
    points_3_0 = score_rule(rows, 3, 0)
    ci_lower, ci_upper = bootstrap_difference(points_2_0, points_1_0)
    return {
        "axis_name": axis_name,
        "threshold_used": float(threshold),
        "match_count": len(rows),
        "favorite_1_0_ev": float(points_1_0.mean()) if len(rows) else np.nan,
        "favorite_2_0_ev": float(points_2_0.mean()) if len(rows) else np.nan,
        "favorite_3_0_ev": float(points_3_0.mean()) if len(rows) else np.nan,
        "difference_2_0_minus_1_0": float((points_2_0 - points_1_0).mean()) if len(rows) else np.nan,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "favorite_win_by_1_rate": float(rows["favorite_win_by_1"].mean()) if len(rows) else np.nan,
        "favorite_win_by_2_rate": float(rows["favorite_win_by_2"].mean()) if len(rows) else np.nan,
        "favorite_win_by_3_plus_rate": float(rows["favorite_win_by_3_plus"].mean()) if len(rows) else np.nan,
        "draw_rate": float(rows["draw"].mean()) if len(rows) else np.nan,
        "upset_rate": float(rows["upset"].mean()) if len(rows) else np.nan,
    }


def choose_top_decile_threshold(matches: pd.DataFrame, axis_name: str) -> float | None:
    """Choose an approximately top-decile severity threshold with enough matches."""
    values = matches[axis_name].dropna()
    if len(values) < MIN_BUCKET_MATCHES:
        return None
    threshold = float(values.quantile(0.90))
    if int(values.ge(threshold).sum()) >= MIN_BUCKET_MATCHES:
        return threshold
    ordered = values.sort_values(ascending=False)
    return float(ordered.iloc[MIN_BUCKET_MATCHES - 1])


def build_axis_convergence_report(matches: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Test convergence across ordinal rank, ranking points, and causal Elo."""
    thresholds = {
        "ordinal_rank_gap": ORIGINAL_RANK_THRESHOLD,
        "ranking_points_gap": choose_top_decile_threshold(matches, "ranking_points_gap"),
        "causal_elo_gap": choose_top_decile_threshold(matches, "causal_elo_gap"),
    }
    rows = [
        summarize_extreme_bucket(matches, axis_name, threshold)
        for axis_name, threshold in thresholds.items()
        if threshold is not None
    ]
    report = pd.DataFrame(rows)
    report["positive_point_estimate"] = report["difference_2_0_minus_1_0"] > 0
    report["nonnegative_ci_lower"] = report["ci_lower_95"] >= 0
    report["margin_mechanism_confirmed"] = report["favorite_win_by_2_rate"] >= report["favorite_win_by_1_rate"]
    strong_axes = report["positive_point_estimate"] & report["nonnegative_ci_lower"] & report["margin_mechanism_confirmed"]
    positive_axes = report["positive_point_estimate"]
    main_rank = report[report["axis_name"].eq("ordinal_rank_gap")].iloc[0]
    if int(strong_axes.sum()) >= 2:
        tier = "Strong 2-0 exception"
    elif int(positive_axes.sum()) >= 2 and bool(main_rank["margin_mechanism_confirmed"]):
        tier = "Tentative 2-0 exception"
    else:
        tier = "Reject 2-0 exception"
    return report, tier


def summarize_rank_threshold(rows: pd.DataFrame, threshold: float, section: str) -> dict[str, object]:
    """Summarize one ordinal-rank threshold for training or holdout rows."""
    bucket = rows[rows["ordinal_rank_gap"].notna() & rows["ordinal_rank_gap"].ge(threshold)].copy()
    points_1_0 = score_rule(bucket, 1, 0)
    points_2_0 = score_rule(bucket, 2, 0)
    ci_lower, ci_upper = bootstrap_difference(points_2_0, points_1_0)
    return {
        "section": section,
        "threshold": float(threshold),
        "match_count": len(bucket),
        "favorite_1_0_ev": float(points_1_0.mean()) if len(bucket) else np.nan,
        "favorite_2_0_ev": float(points_2_0.mean()) if len(bucket) else np.nan,
        "difference_2_0_minus_1_0": float((points_2_0 - points_1_0).mean()) if len(bucket) else np.nan,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "favorite_win_by_1_rate": float(bucket["favorite_win_by_1"].mean()) if len(bucket) else np.nan,
        "favorite_win_by_2_rate": float(bucket["favorite_win_by_2"].mean()) if len(bucket) else np.nan,
        "favorite_win_by_3_plus_rate": float(bucket["favorite_win_by_3_plus"].mean()) if len(bucket) else np.nan,
    }


def build_holdout_report(matches: pd.DataFrame) -> tuple[pd.DataFrame, float | None]:
    """Select a rank threshold on 2000-2014 and test it on 2015-2025."""
    training = matches[matches["date"].le(TRAINING_END)]
    holdout = matches[matches["date"].between(HOLDOUT_START, HOLDOUT_END)]
    training_rows = pd.DataFrame(
        [summarize_rank_threshold(training, threshold, "training_scan_2000_2014") for threshold in TRAINING_RANK_THRESHOLDS]
    )
    candidates = training_rows[
        training_rows["match_count"].ge(MIN_BUCKET_MATCHES)
        & training_rows["favorite_win_by_2_rate"].ge(training_rows["favorite_win_by_1_rate"])
    ]
    selected_threshold = (
        float(candidates.sort_values("difference_2_0_minus_1_0", ascending=False, kind="stable").iloc[0]["threshold"])
        if not candidates.empty
        else None
    )
    report_frames = [training_rows]
    if selected_threshold is not None:
        selected_holdout = summarize_rank_threshold(holdout, selected_threshold, "holdout_selected_threshold_2015_2025")
        selected_holdout["selected_threshold_from_training"] = selected_threshold
        report_frames.append(pd.DataFrame([selected_holdout]))
    reference_holdout = summarize_rank_threshold(holdout, ORIGINAL_RANK_THRESHOLD, "holdout_original_rank80_reference")
    reference_holdout["selected_threshold_from_training"] = selected_threshold
    report_frames.append(pd.DataFrame([reference_holdout]))
    return pd.concat(report_frames, ignore_index=True), selected_threshold


def build_margin_breakdown(matches: pd.DataFrame, selected_threshold: float | None) -> pd.DataFrame:
    """Report the mechanism behind original and clean holdout rank thresholds."""
    holdout = matches[matches["date"].between(HOLDOUT_START, HOLDOUT_END)]
    sections = [("original_rank80_full_history", matches, ORIGINAL_RANK_THRESHOLD)]
    if selected_threshold is not None:
        sections.append(("holdout_selected_threshold_2015_2025", holdout, selected_threshold))
    rows = []
    for section, source, threshold in sections:
        bucket = source[source["ordinal_rank_gap"].notna() & source["ordinal_rank_gap"].ge(threshold)]
        by_1 = float(bucket["favorite_win_by_1"].mean()) if len(bucket) else np.nan
        by_2 = float(bucket["favorite_win_by_2"].mean()) if len(bucket) else np.nan
        rows.append(
            {
                "section": section,
                "threshold": float(threshold),
                "match_count": len(bucket),
                "favorite_win_by_1_rate": by_1,
                "favorite_win_by_2_rate": by_2,
                "favorite_win_by_3_plus_rate": float(bucket["favorite_win_by_3_plus"].mean()) if len(bucket) else np.nan,
                "draw_rate": float(bucket["draw"].mean()) if len(bucket) else np.nan,
                "upset_rate": float(bucket["upset"].mean()) if len(bucket) else np.nan,
                "average_favorite_margin": float(bucket["favorite_margin"].mean()) if len(bucket) else np.nan,
                "average_total_goals": float(bucket["actual_total_goals"].mean()) if len(bucket) else np.nan,
                "favorite_win_by_2_gte_by_1": bool(by_2 >= by_1),
                "mechanism_note": (
                    "Margin shift confirmed."
                    if by_2 >= by_1
                    else "Flag: 2-0 EV may reflect exact-score noise rather than a stable margin shift."
                ),
            }
        )
    return pd.DataFrame(rows)


def latest_fixture_rankings(fixtures: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """Add latest FIFA rankings strictly before each 2026 group fixture."""
    return add_ranking_features(fixtures, rankings)


def build_fixture_application_report(
    rankings: pd.DataFrame,
    latest_elo: dict[str, float],
    selected_threshold: float | None,
    convergence: pd.DataFrame,
) -> pd.DataFrame:
    """Report how many 2026 group fixtures would receive the advisory 2-0 rule."""
    fixtures = load_fixtures()
    group = fixtures[fixtures["stage"].eq("group_stage") & ~fixtures["has_placeholder_team"]].copy()
    group = latest_fixture_rankings(group, rankings)
    group["elo_a"] = group["team_a"].map(lambda team: latest_elo.get(str(team), ELO_INITIAL_RATING))
    group["elo_b"] = group["team_b"].map(lambda team: latest_elo.get(str(team), ELO_INITIAL_RATING))
    group["has_rank_favorite"] = group["rank_a"].notna() & group["rank_b"].notna() & group["rank_a"].ne(group["rank_b"])
    group["favorite_is_a"] = np.where(group["has_rank_favorite"], group["rank_a"] < group["rank_b"], group["elo_a"] > group["elo_b"])
    group["favorite"] = np.where(group["favorite_is_a"], group["team_a"], group["team_b"])
    group["underdog"] = np.where(group["favorite_is_a"], group["team_b"], group["team_a"])
    group["ordinal_rank_gap"] = (group["rank_a"] - group["rank_b"]).abs()
    group["ranking_points_gap"] = (group["ranking_points_a"] - group["ranking_points_b"]).abs()
    group["causal_elo_gap"] = (group["elo_a"] - group["elo_b"]).abs()
    points_threshold = float(convergence.loc[convergence["axis_name"].eq("ranking_points_gap"), "threshold_used"].iloc[0])
    elo_threshold = float(convergence.loc[convergence["axis_name"].eq("causal_elo_gap"), "threshold_used"].iloc[0])
    group["clears_original_rank80"] = group["ordinal_rank_gap"].ge(ORIGINAL_RANK_THRESHOLD)
    group["clears_holdout_selected_rank_threshold"] = (
        group["ordinal_rank_gap"].ge(selected_threshold) if selected_threshold is not None else False
    )
    group["clears_top_decile_ranking_points_gap"] = group["ranking_points_gap"].ge(points_threshold)
    group["clears_top_decile_causal_elo_gap"] = group["causal_elo_gap"].ge(elo_threshold)
    group["current_favorite_1_0_recommendation"] = np.where(group["favorite_is_a"], "1-0", "0-1")
    group["proposed_favorite_2_0_recommendation"] = np.where(group["favorite_is_a"], "2-0", "0-2")
    columns = [
        "match_id", "group", "date", "team_a", "team_b", "favorite", "underdog", "rank_a", "rank_b",
        "ranking_date_a", "ranking_date_b", "ordinal_rank_gap", "ranking_points_gap", "causal_elo_gap",
        "current_favorite_1_0_recommendation", "proposed_favorite_2_0_recommendation",
        "clears_original_rank80", "clears_holdout_selected_rank_threshold",
        "clears_top_decile_ranking_points_gap", "clears_top_decile_causal_elo_gap",
    ]
    return group[columns].sort_values("match_id", kind="stable").reset_index(drop=True)


def create_rule_b_advisory_copy(fixtures: pd.DataFrame, output_dir: Path) -> None:
    """Create a Rule B advisory copy without overwriting production recommendations."""
    source_path = output_dir / "final_match_recommendations.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"Final match recommendations are required for the advisory copy: {source_path}")
    recommendations = pd.read_csv(source_path)
    advisory = recommendations.merge(
        fixtures[
            [
                "match_id", "ordinal_rank_gap", "clears_holdout_selected_rank_threshold",
                "proposed_favorite_2_0_recommendation",
            ]
        ],
        on="match_id",
        how="left",
        validate="one_to_one",
    )
    advisory["rule_b_advisory_prediction"] = np.where(
        advisory["clears_holdout_selected_rank_threshold"].fillna(False),
        advisory["proposed_favorite_2_0_recommendation"],
        advisory["favorite_1_0_prediction"],
    )
    advisory["rule_b_advisory_note"] = np.where(
        advisory["clears_holdout_selected_rank_threshold"].fillna(False),
        "Advisory Rule B exception: rank gap clears accepted holdout-selected threshold.",
        "Advisory Rule B default: favorite_1_0.",
    )
    advisory.to_csv(output_dir / "final_match_recommendations_rule_b_advisory.csv", index=False)
    with pd.ExcelWriter(output_dir / "final_match_recommendations_rule_b_advisory.xlsx", engine="openpyxl") as writer:
        advisory.to_excel(writer, sheet_name="rule_b_advisory", index=False)


def write_rule_b_summary(
    output_dir: Path,
    convergence: pd.DataFrame,
    convergence_tier: str,
    holdout: pd.DataFrame,
    selected_threshold: float | None,
    breakdown: pd.DataFrame,
    fixtures: pd.DataFrame,
    gate_passed: bool,
) -> None:
    """Write the advisory-only Rule B gate decision."""
    holdout_selected = holdout[holdout["section"].eq("holdout_selected_threshold_2015_2025")]
    relevant_breakdown = breakdown[breakdown["section"].eq("holdout_selected_threshold_2015_2025")]
    affected = int(fixtures["clears_holdout_selected_rank_threshold"].sum()) if selected_threshold is not None else 0
    ranking_dates = sorted(
        {
            str(pd.Timestamp(date).date())
            for date in pd.concat([fixtures["ranking_date_a"], fixtures["ranking_date_b"]]).dropna().unique()
        }
    )
    axes_available = ", ".join(convergence["axis_name"].tolist())
    strong_axes = int(
        (
            convergence["positive_point_estimate"]
            & convergence["nonnegative_ci_lower"]
            & convergence["margin_mechanism_confirmed"]
        ).sum()
    )
    positive_axes = int(convergence["positive_point_estimate"].sum())
    if gate_passed:
        decision = "A. Encode Rule B advisory"
    elif convergence_tier == "Tentative 2-0 exception":
        decision = "B. Tentative only, keep 1-0"
    else:
        decision = "C. Reject, keep 1-0 everywhere"
    lines = [
        "Rule B Advisory Decision Summary",
        "================================",
        "Scoring note: full-history rows use the real Penka group-stage scoring approximation.",
        f"axes_available: {axes_available}",
        "axes_skipped_due_to_leakage_or_missing_data: favorite_win_probability, expected_goal_difference",
        f"axis_convergence_result: {convergence_tier}",
        f"axes_with_positive_2_0_point_estimate: {positive_axes}",
        f"axes_passing_strong_gate: {strong_axes}",
        f"clean_holdout_selected_rank_threshold: {selected_threshold if selected_threshold is not None else 'None'}",
        (
            "clean_holdout_result: "
            f"matches={int(holdout_selected.iloc[0]['match_count'])}, "
            f"difference_2_0_minus_1_0={holdout_selected.iloc[0]['difference_2_0_minus_1_0']:.6f}, "
            f"ci=[{holdout_selected.iloc[0]['ci_lower_95']:.6f}, {holdout_selected.iloc[0]['ci_upper_95']:.6f}]"
            if not holdout_selected.empty
            else "clean_holdout_result: unavailable"
        ),
        (
            "margin_mechanism_result: "
            f"by_1={relevant_breakdown.iloc[0]['favorite_win_by_1_rate']:.6f}, "
            f"by_2={relevant_breakdown.iloc[0]['favorite_win_by_2_rate']:.6f}, "
            f"by_3_plus={relevant_breakdown.iloc[0]['favorite_win_by_3_plus_rate']:.6f}, "
            f"by_2_gte_by_1={bool(relevant_breakdown.iloc[0]['favorite_win_by_2_gte_by_1'])}"
            if not relevant_breakdown.empty
            else "margin_mechanism_result: unavailable"
        ),
        f"affected_2026_group_fixtures: {affected}",
        f"latest_available_fixture_ranking_date: {max(ranking_dates) if ranking_dates else 'unavailable'}",
        f"fixture_ranking_dates_used: {', '.join(ranking_dates) if ranking_dates else 'unavailable'}",
        "june_2026_rankings_available: False",
        f"final_decision: {decision}",
        "production_final_match_recommendations_overwritten: False",
    ]
    (output_dir / "rule_b_decision_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_rule_b_verification(
    results: pd.DataFrame,
    rankings: pd.DataFrame,
    output_dir: str | Path = "outputs",
) -> dict[str, object]:
    """Run the advisory-only Rule B verification suite."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    matches, quality, _ = build_severity_axes(results, rankings)
    convergence, convergence_tier = build_axis_convergence_report(matches)
    holdout, selected_threshold = build_holdout_report(matches)
    breakdown = build_margin_breakdown(matches, selected_threshold)
    group_fixtures = load_fixtures()
    first_group_fixture_date = group_fixtures.loc[group_fixtures["stage"].eq("group_stage"), "date"].min()
    fixture_elo = latest_causal_elo_before_date(results, first_group_fixture_date)
    fixture_application = build_fixture_application_report(rankings, fixture_elo, selected_threshold, convergence)
    quality.to_csv(output_path / "severity_axes_quality_report.csv", index=False)
    convergence.to_csv(output_path / "two_zero_axis_convergence.csv", index=False)
    holdout.to_csv(output_path / "two_zero_holdout_report.csv", index=False)
    breakdown.to_csv(output_path / "rank80_margin_breakdown.csv", index=False)
    fixture_application.to_csv(output_path / "rule_b_2026_affected_fixtures.csv", index=False)

    holdout_selected = holdout[holdout["section"].eq("holdout_selected_threshold_2015_2025")]
    breakdown_selected = breakdown[breakdown["section"].eq("holdout_selected_threshold_2015_2025")]
    affected = int(fixture_application["clears_holdout_selected_rank_threshold"].sum()) if selected_threshold is not None else 0
    gate_passed = bool(
        convergence_tier == "Strong 2-0 exception"
        and not holdout_selected.empty
        and holdout_selected.iloc[0]["difference_2_0_minus_1_0"] > 0
        and not breakdown_selected.empty
        and bool(breakdown_selected.iloc[0]["favorite_win_by_2_gte_by_1"])
        and affected >= 1
    )
    if gate_passed:
        create_rule_b_advisory_copy(fixture_application, output_path)
    write_rule_b_summary(
        output_path,
        convergence,
        convergence_tier,
        holdout,
        selected_threshold,
        breakdown,
        fixture_application,
        gate_passed,
    )
    return {
        "severity_axes_quality_report": quality,
        "two_zero_axis_convergence": convergence,
        "two_zero_holdout_report": holdout,
        "rank80_margin_breakdown": breakdown,
        "rule_b_2026_affected_fixtures": fixture_application,
        "convergence_tier": convergence_tier,
        "selected_threshold": selected_threshold,
        "gate_passed": gate_passed,
    }
