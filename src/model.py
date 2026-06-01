"""Train, evaluate, persist, and use expected-goals models."""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import MODEL_PARAMS_POISSON_HGB, MODEL_PARAMS_SQUARED_HGB, PREDICTION_REFERENCE_DATE
from penka_scoring import GROUP, score_prediction


MIN_EXPECTED_GOALS = 0.05
MAX_EXPECTED_GOALS = 6.0
RECENCY_HALF_LIFE_DAYS = 730.0
VALIDATION_FRACTION = 0.20
SCORE_GRID_MAX_GOALS = 6
TARGET_COLUMNS = ["goals_a", "goals_b"]


class AverageGoalsRegressor(BaseEstimator, RegressorMixin):
    """Predict a weighted average goal count."""

    def fit(self, x, y, sample_weight=None):
        values = np.asarray(y, dtype=float)
        self.mean_goals_ = float(np.average(values, weights=sample_weight))
        return self

    def predict(self, x):
        return np.full(len(x), self.mean_goals_, dtype=float)


def validate_model_inputs(training_df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Raise clear errors for missing training inputs."""
    required = set(feature_cols) | {"date", *TARGET_COLUMNS}
    missing = sorted(required - set(training_df.columns))
    if missing:
        raise ValueError(
            f"training_df is missing required columns: {missing}. "
            f"Available columns: {sorted(training_df.columns.tolist())}"
        )


def calculate_recency_weights(dates: pd.Series, reference_date: str | pd.Timestamp) -> np.ndarray:
    """Return exp(-days_old / 730) weights for historical matches."""
    parsed_dates = pd.to_datetime(dates, errors="raise")
    days_old = (pd.Timestamp(reference_date) - parsed_dates).dt.days.clip(lower=0)
    return np.exp(-days_old.to_numpy(dtype=float) / RECENCY_HALF_LIFE_DAYS)


def date_based_split(
    training_df: pd.DataFrame,
    reference_date: str | pd.Timestamp = PREDICTION_REFERENCE_DATE,
    validation_fraction: float = VALIDATION_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split chronologically, reserving the latest historical dates for validation."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    eligible = training_df.copy()
    eligible["date"] = pd.to_datetime(eligible["date"], errors="raise")
    eligible = eligible[eligible["date"] < pd.Timestamp(reference_date)].sort_values("date", kind="stable")
    if len(eligible) < 2:
        raise ValueError("At least two historical matches before reference_date are required.")

    cutoff_position = max(1, min(len(eligible) - 1, int(len(eligible) * (1 - validation_fraction))))
    validation_start = eligible.iloc[cutoff_position]["date"]
    train_df = eligible[eligible["date"] < validation_start].copy()
    validation_df = eligible[eligible["date"] >= validation_start].copy()
    if train_df.empty or validation_df.empty:
        raise ValueError("Date-based split produced an empty train or validation set.")
    return train_df, validation_df, validation_start


def build_candidate_estimators() -> dict[str, BaseEstimator]:
    """Create CPU-friendly expected-goals candidate models."""
    return {
        "hist_gradient_boosting_poisson": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingRegressor(**MODEL_PARAMS_POISSON_HGB)),
            ]
        ),
        "hist_gradient_boosting_squared_error": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingRegressor(**MODEL_PARAMS_SQUARED_HGB)),
            ]
        ),
        "poisson_regressor": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", PoissonRegressor(alpha=1.0, max_iter=150)),
            ]
        ),
        "average_goals": AverageGoalsRegressor(),
    }


def fit_goal_pair(
    estimator: BaseEstimator,
    training_df: pd.DataFrame,
    feature_cols: list[str],
    reference_date: str | pd.Timestamp,
) -> dict[str, BaseEstimator]:
    """Fit separate Team A and Team B goal estimators."""
    x_train = training_df[feature_cols]
    sample_weight = calculate_recency_weights(training_df["date"], reference_date)
    model_a = clone(estimator)
    model_b = clone(estimator)
    fit_params = {"model__sample_weight": sample_weight} if isinstance(estimator, Pipeline) else {"sample_weight": sample_weight}
    model_a.fit(x_train, training_df["goals_a"], **fit_params)
    model_b.fit(x_train, training_df["goals_b"], **fit_params)
    return {"team_a": model_a, "team_b": model_b}


def clip_expected_goals(values) -> np.ndarray:
    """Clip expected goals to the supported prediction range."""
    return np.clip(np.asarray(values, dtype=float), MIN_EXPECTED_GOALS, MAX_EXPECTED_GOALS)


def predict_goal_pair(model_pair: dict[str, BaseEstimator], features: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Predict clipped Team A and Team B expected goals."""
    x = features[feature_cols]
    expected_a = clip_expected_goals(model_pair["team_a"].predict(x))
    expected_b = clip_expected_goals(model_pair["team_b"].predict(x))
    return expected_a, expected_b


def poisson_negative_log_likelihood(actual_goals, expected_goals) -> float:
    """Return mean Poisson negative log likelihood."""
    actual = np.asarray(actual_goals, dtype=float)
    expected = clip_expected_goals(expected_goals)
    log_factorial = np.asarray([math.lgamma(goal + 1.0) for goal in actual])
    return float(np.mean(expected - actual * np.log(expected) + log_factorial))


def optimize_scoreline(expected_a: float, expected_b: float, max_goals: int = SCORE_GRID_MAX_GOALS) -> tuple[int, int]:
    """Return the most likely scoreline from an independent Poisson grid."""
    goals = np.arange(max_goals + 1)
    probabilities_a = np.exp(-expected_a) * np.power(expected_a, goals) / np.asarray(
        [math.factorial(goal) for goal in goals]
    )
    probabilities_b = np.exp(-expected_b) * np.power(expected_b, goals) / np.asarray(
        [math.factorial(goal) for goal in goals]
    )
    best_index = np.unravel_index(np.argmax(np.outer(probabilities_a, probabilities_b)), (len(goals), len(goals)))
    return int(goals[best_index[0]]), int(goals[best_index[1]])


def match_result(goals_a: float, goals_b: float) -> str:
    """Return a stable result label for a scoreline."""
    if goals_a > goals_b:
        return "team_a_win"
    if goals_b > goals_a:
        return "team_b_win"
    return "draw"


def historical_challenge_points(actual_a, actual_b, expected_a, expected_b) -> float:
    """Return group-stage Penka points for validation score modes."""
    total_points = 0
    for goals_a, goals_b, lambda_a, lambda_b in zip(actual_a, actual_b, expected_a, expected_b):
        predicted_a, predicted_b = optimize_scoreline(lambda_a, lambda_b)
        total_points += score_prediction(
            predicted_a,
            predicted_b,
            int(goals_a),
            int(goals_b),
            GROUP,
        )
    return total_points / len(expected_a)


def calculate_goal_metrics(validation_df: pd.DataFrame, expected_a, expected_b) -> dict[str, float]:
    """Calculate validation metrics for one expected-goals model pair."""
    actual_a = validation_df["goals_a"].to_numpy(dtype=float)
    actual_b = validation_df["goals_b"].to_numpy(dtype=float)
    expected_a = clip_expected_goals(expected_a)
    expected_b = clip_expected_goals(expected_b)
    actual = np.concatenate([actual_a, actual_b])
    expected = np.concatenate([expected_a, expected_b])
    return {
        "mae": float(mean_absolute_error(actual, expected)),
        "rmse": float(mean_squared_error(actual, expected) ** 0.5),
        "poisson_nll": poisson_negative_log_likelihood(actual, expected),
        "historical_challenge_points": historical_challenge_points(actual_a, actual_b, expected_a, expected_b),
    }


def evaluate_goal_models(
    training_df: pd.DataFrame,
    feature_cols: list[str],
    reference_date: str = PREDICTION_REFERENCE_DATE,
) -> pd.DataFrame:
    """Evaluate all candidate models on a date-based validation split."""
    validate_model_inputs(training_df, feature_cols)
    train_df, validation_df, validation_start = date_based_split(training_df, reference_date)
    rows = []
    for model_name, estimator in build_candidate_estimators().items():
        model_pair = fit_goal_pair(estimator, train_df, feature_cols, reference_date)
        expected_a, expected_b = predict_goal_pair(model_pair, validation_df, feature_cols)
        rows.append(
            {
                "model_name": model_name,
                **calculate_goal_metrics(validation_df, expected_a, expected_b),
                "validation_start": validation_start,
                "validation_rows": len(validation_df),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["historical_challenge_points", "poisson_nll", "mae"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def train_goal_models(
    training_df: pd.DataFrame,
    feature_cols: list[str],
    reference_date: str = PREDICTION_REFERENCE_DATE,
) -> dict[str, object]:
    """Select the best validation model and refit it on all eligible history."""
    evaluation = evaluate_goal_models(training_df, feature_cols, reference_date)
    selected_model_name = str(evaluation.iloc[0]["model_name"])
    eligible = training_df.copy()
    eligible["date"] = pd.to_datetime(eligible["date"], errors="raise")
    eligible = eligible[eligible["date"] < pd.Timestamp(reference_date)].copy()
    estimator = build_candidate_estimators()[selected_model_name]
    selected_models = fit_goal_pair(estimator, eligible, feature_cols, reference_date)
    return {
        "selected_model_name": selected_model_name,
        "models": selected_models,
        "feature_cols": list(feature_cols),
        "reference_date": str(reference_date),
        "validation_results": evaluation,
    }


def predict_expected_goals(models: dict[str, object], fixture_features: pd.DataFrame) -> pd.DataFrame:
    """Return fixtures with clipped Team A and Team B expected goals."""
    feature_cols = list(models["feature_cols"])
    expected_a, expected_b = predict_goal_pair(models["models"], fixture_features, feature_cols)
    predictions = fixture_features.copy()
    predictions["expected_goals_a"] = expected_a
    predictions["expected_goals_b"] = expected_b
    return predictions


def save_models(models: dict[str, object], path: str | Path = "outputs/goal_models.joblib") -> None:
    """Persist trained goal models."""
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, model_path)


def load_models(path: str | Path = "outputs/goal_models.joblib") -> dict[str, object]:
    """Load persisted goal models."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Goal model file not found: {model_path}")
    return joblib.load(model_path)
