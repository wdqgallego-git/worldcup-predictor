"""Limited, transparent expected-goals calibration and strategy diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from model import (
    MAX_EXPECTED_GOALS,
    MIN_EXPECTED_GOALS,
    build_candidate_estimators,
    date_based_split,
    fit_goal_pair,
    predict_goal_pair,
)
from poisson import score_probability_matrix
from prediction_optimizer import get_safe_prediction, points_for_prediction


LINEAR_INTERCEPT_BOUNDS = (-0.50, 0.75)
LINEAR_SLOPE_BOUNDS = (0.50, 1.50)
KNOCKOUT_MULTIPLIER_BOUNDS = (0.95, 1.05)
ISOTONIC_STABILITY_TOLERANCE = 0.05
DEFAULT_BOOTSTRAP_SAMPLES = 5000


@dataclass
class LambdaCalibrator:
    """Apply one stable calibration mapping to expected-goal values."""

    method: str = "linear"
    intercept: float = 0.0
    slope: float = 1.0
    isotonic_x: tuple[float, ...] = ()
    isotonic_y: tuple[float, ...] = ()

    def transform(self, values) -> np.ndarray:
        """Return clipped calibrated expected goals."""
        raw = np.asarray(values, dtype=float)
        if self.method == "linear":
            adjusted = self.intercept + self.slope * raw
        elif self.method == "isotonic":
            adjusted = np.interp(raw, self.isotonic_x, self.isotonic_y)
        else:
            raise ValueError(f"Unsupported calibration method: {self.method}")
        return np.clip(adjusted, MIN_EXPECTED_GOALS, MAX_EXPECTED_GOALS)


def flatten_goal_pairs(predictions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Flatten Team A and Team B raw lambdas and outcomes into paired arrays."""
    raw = np.concatenate(
        [
            predictions["raw_expected_goals_a"].to_numpy(dtype=float),
            predictions["raw_expected_goals_b"].to_numpy(dtype=float),
        ]
    )
    actual = np.concatenate(
        [
            predictions["goals_a"].to_numpy(dtype=float),
            predictions["goals_b"].to_numpy(dtype=float),
        ]
    )
    return raw, actual


def fit_linear_calibrator(raw_lambdas, actual_goals) -> LambdaCalibrator:
    """Fit lambda_adj = a + b * lambda_raw with conservative bounds."""
    raw = np.asarray(raw_lambdas, dtype=float)
    actual = np.asarray(actual_goals, dtype=float)
    if len(raw) < 2:
        return LambdaCalibrator()
    slope, intercept = np.polyfit(raw, actual, deg=1)
    return LambdaCalibrator(
        method="linear",
        intercept=float(np.clip(intercept, *LINEAR_INTERCEPT_BOUNDS)),
        slope=float(np.clip(slope, *LINEAR_SLOPE_BOUNDS)),
    )


def fit_isotonic_calibrator(raw_lambdas, actual_goals) -> LambdaCalibrator:
    """Fit a monotonic isotonic challenger with clipped out-of-range behavior."""
    raw = np.asarray(raw_lambdas, dtype=float)
    actual = np.asarray(actual_goals, dtype=float)
    model = IsotonicRegression(
        y_min=MIN_EXPECTED_GOALS,
        y_max=MAX_EXPECTED_GOALS,
        out_of_bounds="clip",
    ).fit(raw, actual)
    return LambdaCalibrator(
        method="isotonic",
        isotonic_x=tuple(float(value) for value in model.X_thresholds_),
        isotonic_y=tuple(float(value) for value in model.y_thresholds_),
    )


def apply_lambda_calibration(predictions: pd.DataFrame, calibrator: LambdaCalibrator) -> pd.DataFrame:
    """Add raw and calibrated lambda columns while making calibrated values active."""
    adjusted = predictions.copy()
    if "raw_expected_goals_a" not in adjusted.columns:
        adjusted["raw_expected_goals_a"] = adjusted["expected_goals_a"]
    if "raw_expected_goals_b" not in adjusted.columns:
        adjusted["raw_expected_goals_b"] = adjusted["expected_goals_b"]
    adjusted["calibrated_expected_goals_a"] = calibrator.transform(adjusted["raw_expected_goals_a"])
    adjusted["calibrated_expected_goals_b"] = calibrator.transform(adjusted["raw_expected_goals_b"])
    adjusted["expected_goals_a"] = adjusted["calibrated_expected_goals_a"]
    adjusted["expected_goals_b"] = adjusted["calibrated_expected_goals_b"]
    adjusted["calibration_method"] = calibrator.method
    return adjusted


def add_raw_lambda_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return predictions with explicit raw lambda columns."""
    raw = predictions.copy()
    raw["raw_expected_goals_a"] = raw["expected_goals_a"]
    raw["raw_expected_goals_b"] = raw["expected_goals_b"]
    return raw


def optimized_challenge_points(predictions: pd.DataFrame, max_goals: int = 6) -> np.ndarray:
    """Return per-match challenge points after expected-points score optimization."""
    points = []
    for match in predictions.itertuples(index=False):
        probabilities = score_probability_matrix(
            match.expected_goals_a,
            match.expected_goals_b,
            max_goals=max_goals,
        )
        safe = get_safe_prediction(probabilities, max_goals=max_goals)
        points.append(points_for_prediction(safe["pred_a"], safe["pred_b"], int(match.goals_a), int(match.goals_b)))
    return np.asarray(points, dtype=int)


def evaluate_calibrator(calibrator: LambdaCalibrator, predictions: pd.DataFrame, max_goals: int = 6) -> dict[str, float]:
    """Evaluate challenge points, bias, and high-lambda stability."""
    adjusted = apply_lambda_calibration(predictions, calibrator)
    raw_total = predictions["raw_expected_goals_a"] + predictions["raw_expected_goals_b"]
    calibrated_total = adjusted["expected_goals_a"] + adjusted["expected_goals_b"]
    actual_total = adjusted["goals_a"] + adjusted["goals_b"]
    high_lambda = raw_total >= raw_total.quantile(0.75)
    return {
        "method": calibrator.method,
        "avg_challenge_points": float(optimized_challenge_points(adjusted, max_goals=max_goals).mean()),
        "overall_abs_bias": float(abs(calibrated_total.mean() - actual_total.mean())),
        "high_lambda_abs_bias": float(abs(calibrated_total[high_lambda].mean() - actual_total[high_lambda].mean())),
    }


def fit_and_select_calibrator(
    calibration_predictions: pd.DataFrame,
    max_goals: int = 6,
) -> tuple[LambdaCalibrator, pd.DataFrame]:
    """Fit linear default and select isotonic only after a stable out-of-sample win."""
    ordered = calibration_predictions.sort_values("date", kind="stable").reset_index(drop=True)
    split_position = max(1, min(len(ordered) - 1, int(len(ordered) * 0.70)))
    fit_rows = ordered.iloc[:split_position]
    evaluation_rows = ordered.iloc[split_position:]
    fit_raw, fit_actual = flatten_goal_pairs(fit_rows)
    challengers = [
        fit_linear_calibrator(fit_raw, fit_actual),
        fit_isotonic_calibrator(fit_raw, fit_actual),
    ]
    evaluation = pd.DataFrame(
        [evaluate_calibrator(calibrator, evaluation_rows, max_goals=max_goals) for calibrator in challengers]
    )
    linear = evaluation[evaluation["method"].eq("linear")].iloc[0]
    isotonic = evaluation[evaluation["method"].eq("isotonic")].iloc[0]
    use_isotonic = (
        isotonic["avg_challenge_points"] > linear["avg_challenge_points"]
        and isotonic["high_lambda_abs_bias"] <= linear["high_lambda_abs_bias"] + ISOTONIC_STABILITY_TOLERANCE
    )
    selected_method = "isotonic" if use_isotonic else "linear"
    full_raw, full_actual = flatten_goal_pairs(ordered)
    selected = (
        fit_isotonic_calibrator(full_raw, full_actual)
        if selected_method == "isotonic"
        else fit_linear_calibrator(full_raw, full_actual)
    )
    evaluation["selected"] = evaluation["method"].eq(selected_method)
    return selected, evaluation


def fit_pre_tournament_calibrator(
    training_df: pd.DataFrame,
    feature_cols: list[str],
    selected_model_name: str,
    reference_date: str,
    max_goals: int = 6,
) -> tuple[LambdaCalibrator, pd.DataFrame]:
    """Fit calibration on a chronological pre-tournament holdout."""
    fit_rows, calibration_rows, _ = date_based_split(training_df, reference_date)
    estimator = build_candidate_estimators()[selected_model_name]
    model_pair = fit_goal_pair(estimator, fit_rows, feature_cols, reference_date)
    expected_a, expected_b = predict_goal_pair(model_pair, calibration_rows, feature_cols)
    calibration_predictions = calibration_rows.copy()
    calibration_predictions["raw_expected_goals_a"] = expected_a
    calibration_predictions["raw_expected_goals_b"] = expected_b
    return fit_and_select_calibrator(calibration_predictions, max_goals=max_goals)


def add_historical_match_context(predictions: pd.DataFrame) -> pd.DataFrame:
    """Mark old-format World Cup group and knockout rows for calibration reporting."""
    contextual = predictions.sort_values(["backtest_year", "date"], kind="stable").copy()
    contextual["_tournament_match_number"] = contextual.groupby("backtest_year").cumcount() + 1
    contextual["match_context"] = np.where(contextual["_tournament_match_number"] <= 48, "group", "knockout")
    return contextual.drop(columns="_tournament_match_number")


def build_calibration_report(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report predicted versus actual mean total goals overall and by lambda bucket."""
    reported = add_historical_match_context(predictions)
    reported["raw_total_lambda"] = reported["raw_expected_goals_a"] + reported["raw_expected_goals_b"]
    reported["calibrated_total_lambda"] = (
        reported["calibrated_expected_goals_a"] + reported["calibrated_expected_goals_b"]
    )
    reported["actual_total_goals"] = reported["goals_a"] + reported["goals_b"]
    reported["lambda_bucket"] = pd.cut(
        reported["raw_total_lambda"],
        bins=[-np.inf, 2.0, 3.0, np.inf],
        labels=["low_up_to_2.0", "medium_2.0_3.0", "high_above_3.0"],
    )
    rows = []
    dimensions = [("all", pd.Series(True, index=reported.index))]
    dimensions.extend((context, reported["match_context"].eq(context)) for context in ("group", "knockout"))
    for context, context_mask in dimensions:
        context_rows = reported[context_mask]
        for bucket, bucket_rows in [("all", context_rows), *list(context_rows.groupby("lambda_bucket", observed=True))]:
            rows.append(
                {
                    "context": context,
                    "lambda_bucket": str(bucket),
                    "matches": len(bucket_rows),
                    "raw_predicted_mean_total_goals": float(bucket_rows["raw_total_lambda"].mean()),
                    "calibrated_predicted_mean_total_goals": float(bucket_rows["calibrated_total_lambda"].mean()),
                    "actual_mean_total_goals": float(bucket_rows["actual_total_goals"].mean()),
                    "raw_bias": float(bucket_rows["raw_total_lambda"].mean() - bucket_rows["actual_total_goals"].mean()),
                    "calibrated_bias": float(
                        bucket_rows["calibrated_total_lambda"].mean() - bucket_rows["actual_total_goals"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def estimate_knockout_context_multiplier(predictions: pd.DataFrame) -> float:
    """Estimate a constrained knockout total-goal adjustment from historical World Cups."""
    reported = add_historical_match_context(predictions)
    knockout = reported[reported["match_context"].eq("knockout")]
    predicted_total = (knockout["calibrated_expected_goals_a"] + knockout["calibrated_expected_goals_b"]).mean()
    actual_total = (knockout["goals_a"] + knockout["goals_b"]).mean()
    if not np.isfinite(predicted_total) or predicted_total <= 0:
        return 1.0
    return float(np.clip(actual_total / predicted_total, *KNOCKOUT_MULTIPLIER_BOUNDS))


def paired_bootstrap_report(
    strategy_points: pd.DataFrame,
    baseline_column: str = "favorite_1_0_points",
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    random_seed: int = 2026,
) -> pd.DataFrame:
    """Report paired-bootstrap confidence intervals for strategy point differences."""
    rng = np.random.default_rng(random_seed)
    baseline = strategy_points[baseline_column].to_numpy(dtype=float)
    rows = []
    for column in strategy_points.columns:
        if column == baseline_column or not column.endswith("_points"):
            continue
        differences = strategy_points[column].to_numpy(dtype=float) - baseline
        sample_indexes = rng.integers(0, len(differences), size=(n_bootstrap, len(differences)))
        bootstrap_means = differences[sample_indexes].mean(axis=1)
        rows.append(
            {
                "strategy": column.removesuffix("_points"),
                "baseline": baseline_column.removesuffix("_points"),
                "matches": len(differences),
                "mean_point_difference": float(differences.mean()),
                "ci_lower_95": float(np.quantile(bootstrap_means, 0.025)),
                "ci_upper_95": float(np.quantile(bootstrap_means, 0.975)),
                "bootstrap_probability_strategy_better": float(np.mean(bootstrap_means > 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_point_difference", ascending=False, kind="stable")

