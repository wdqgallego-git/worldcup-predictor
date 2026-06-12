"""Optional bookmaker priors: de-vig, market-implied lambdas, and gated blending."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from sklearn.linear_model import LogisticRegression

from penka_scoring import GROUP, score_prediction
from poisson import summarize_score_probs


MATCH_ODDS_REQUIRED_COLUMNS = [
    "match_id", "date", "team_a", "team_b", "bookmaker", "odds_team_a",
    "odds_draw", "odds_team_b", "market_type", "source_timestamp",
]
MATCH_ODDS_OPTIONAL_COLUMNS = [
    "over_under_line", "odds_over", "odds_under", "spread",
    "hours_before_kickoff", "implied_prob_team_a", "implied_prob_draw", "implied_prob_team_b",
]
MATCH_ODDS_COLUMNS = MATCH_ODDS_REQUIRED_COLUMNS + MATCH_ODDS_OPTIONAL_COLUMNS
ODDS_STATUS_OK = "ok"
ODDS_STATUS_BLANK = "blank_odds_model_fallback"
MIN_PLAUSIBLE_OVERROUND = -0.02
MAX_PLAUSIBLE_OVERROUND = 0.25
MAX_PLAUSIBLE_DECIMAL_ODDS = 200.0
MARKET_LAMBDA_BOUNDS = (0.05, 5.5)
DEFAULT_TOTAL_GOALS_PRIOR = 2.6
FUTURES_ODDS_COLUMNS = ["team", "bookmaker", "market", "decimal_odds", "source_timestamp"]
FUTURES_MARKETS = {"champion", "runner_up", "top_scorer", "golden_boot"}
PROBABILITY_COLUMNS = ["prob_team_a", "prob_draw", "prob_team_b"]
OUTPUT_FILES = {
    "backtest": "odds_prior_backtest.csv",
    "summary": "odds_prior_summary.txt",
    "features": "market_prior_features.csv",
    "disagreements": "model_vs_market_disagreements.csv",
}


def _validate_columns(df: pd.DataFrame, required_columns, dataset_name: str) -> None:
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}. Available: {sorted(df.columns)}")


def _validate_decimal_odds(values: pd.Series, label: str) -> pd.Series:
    odds = pd.to_numeric(values, errors="coerce")
    invalid = odds.isna() | ~np.isfinite(odds) | odds.le(1.0)
    if invalid.any():
        raise ValueError(f"{label} contains {int(invalid.sum())} invalid decimal odd(s); require finite values > 1.0.")
    return odds.astype(float)


def simple_implied_probabilities(decimal_odds) -> np.ndarray:
    """Return unnormalized implied probabilities from decimal odds."""
    odds = np.asarray(decimal_odds, dtype=float)
    if odds.ndim != 1 or len(odds) < 2 or not np.all(np.isfinite(odds)) or np.any(odds <= 1.0):
        raise ValueError("Decimal odds must be a finite one-dimensional array with values > 1.0.")
    return 1.0 / odds


def normalized_implied_probabilities(decimal_odds) -> np.ndarray:
    """Remove the bookmaker overround by proportional normalization."""
    implied = simple_implied_probabilities(decimal_odds)
    return implied / implied.sum()


def calculate_overround(decimal_odds) -> float:
    """Return sum(implied probabilities) - 1."""
    return float(simple_implied_probabilities(decimal_odds).sum() - 1.0)


def _shin_probabilities_for_z(implied: np.ndarray, insider_fraction: float) -> np.ndarray:
    total = implied.sum()
    z = float(insider_fraction)
    if z >= 1.0:
        raise ValueError("Shin insider fraction must be below 1.")
    return (np.sqrt(z * z + 4.0 * (1.0 - z) * implied * implied / total) - z) / (2.0 * (1.0 - z))


def shin_adjusted_probabilities(decimal_odds, tolerance: float = 1e-12) -> tuple[np.ndarray, float]:
    """Return Shin-adjusted probabilities and the inferred insider fraction."""
    implied = simple_implied_probabilities(decimal_odds)
    if implied.sum() <= 1.0 + tolerance:
        return implied / implied.sum(), 0.0
    low, high = 0.0, 0.999999
    low_sum = _shin_probabilities_for_z(implied, low).sum()
    high_sum = _shin_probabilities_for_z(implied, high).sum()
    if not (low_sum >= 1.0 and high_sum <= 1.0):
        return implied / implied.sum(), 0.0
    for _ in range(100):
        middle = (low + high) / 2.0
        probability_sum = _shin_probabilities_for_z(implied, middle).sum()
        if probability_sum > 1.0:
            low = middle
        else:
            high = middle
    z = (low + high) / 2.0
    probabilities = _shin_probabilities_for_z(implied, z)
    probabilities = np.clip(probabilities, 0.0, None)
    return probabilities / probabilities.sum(), float(z)


def _read_optional_csv(path: str | Path, required_columns, dataset_name: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame(columns=required_columns)
    table = pd.read_csv(csv_path)
    _validate_columns(table, required_columns, dataset_name)
    return table


_PROBABILITY_OUTPUT_COLUMNS = [
    "simple_prob_team_a", "simple_prob_draw", "simple_prob_team_b", "overround",
    "normalized_prob_team_a", "normalized_prob_draw", "normalized_prob_team_b",
    "shin_prob_team_a", "shin_prob_draw", "shin_prob_team_b", "shin_insider_fraction",
    "prob_over",
]


def _standardize_odds_team_names(odds: pd.DataFrame) -> pd.DataFrame:
    """Map entered team names to canonical fixture names when the mapping is available."""
    mapping_path = Path("data/raw/team_name_mapping.csv")
    if not mapping_path.exists():
        return odds
    from data_loader import apply_team_mapping, load_team_name_mapping

    return apply_team_mapping(odds, ["team_a", "team_b"], load_team_name_mapping(mapping_path))


def _validate_match_odds_row(row: pd.Series) -> list[str]:
    """Return row-level data-entry flags. Blank 1X2 odds are a fallback, not an error."""
    flags: list[str] = []
    one_x_two = [row["odds_team_a"], row["odds_draw"], row["odds_team_b"]]
    filled = [value for value in one_x_two if pd.notna(value)]
    if not filled:
        return [ODDS_STATUS_BLANK]
    if len(filled) < 3:
        return ["invalid_incomplete_1x2"]
    if any(value <= 1.0 for value in filled):
        flags.append("invalid_odds_not_above_1")
    elif any(value > MAX_PLAUSIBLE_DECIMAL_ODDS for value in filled):
        flags.append("suspect_extreme_odds")
    else:
        overround = float(sum(1.0 / value for value in filled) - 1.0)
        if overround < MIN_PLAUSIBLE_OVERROUND:
            flags.append(f"suspect_arbitrage_overround_{overround:.3f}_check_for_fat_finger")
        elif overround > MAX_PLAUSIBLE_OVERROUND:
            flags.append(f"suspect_high_overround_{overround:.3f}_check_for_fat_finger")
    if pd.notna(row.get("over_under_line")):
        over, under = row.get("odds_over"), row.get("odds_under")
        if pd.isna(over) or pd.isna(under) or over <= 1.0 or under <= 1.0:
            flags.append("invalid_over_under_odds")
    if pd.notna(row.get("hours_before_kickoff")) and float(row["hours_before_kickoff"]) < 0:
        flags.append("invalid_post_kickoff_snapshot")
    return flags or [ODDS_STATUS_OK]


def load_match_odds(path: str | Path = "data/raw/match_odds.csv", strict: bool = False) -> pd.DataFrame:
    """Load optional 1X2 match odds with row-level validation flags and probabilities.

    Rows with completely blank 1X2 odds are kept and marked ODDS_STATUS_BLANK so the
    pipeline can fall back to model-only picks. Fat-fingered or implausible rows are
    flagged by row and excluded from the consensus instead of crashing the run.
    """
    odds = _read_optional_csv(path, MATCH_ODDS_REQUIRED_COLUMNS, "match odds")
    for column in MATCH_ODDS_OPTIONAL_COLUMNS:
        if column not in odds.columns:
            odds[column] = np.nan
    if odds.empty:
        empty = odds.assign(**{column: pd.Series(dtype=float) for column in _PROBABILITY_OUTPUT_COLUMNS})
        empty["odds_status"] = pd.Series(dtype=str)
        return empty
    odds = odds.copy()
    odds = _standardize_odds_team_names(odds)
    odds["date"] = pd.to_datetime(odds["date"], errors="raise")
    odds["source_timestamp"] = pd.to_datetime(odds["source_timestamp"], errors="coerce", utc=True)
    for column in ("odds_team_a", "odds_draw", "odds_team_b", "over_under_line", "odds_over", "odds_under", "spread", "hours_before_kickoff"):
        odds[column] = pd.to_numeric(odds[column], errors="coerce")
    odds["odds_status"] = [";".join(_validate_match_odds_row(row)) for _, row in odds.iterrows()]
    probability_rows = []
    for _, row in odds.iterrows():
        if row["odds_status"] != ODDS_STATUS_OK:
            probability_rows.append({column: np.nan for column in _PROBABILITY_OUTPUT_COLUMNS})
            continue
        decimal_odds = [row["odds_team_a"], row["odds_draw"], row["odds_team_b"]]
        simple = simple_implied_probabilities(decimal_odds)
        normalized = normalized_implied_probabilities(decimal_odds)
        shin, insider_fraction = shin_adjusted_probabilities(decimal_odds)
        prob_over = np.nan
        if pd.notna(row["over_under_line"]) and pd.notna(row["odds_over"]) and pd.notna(row["odds_under"]):
            prob_over = float(normalized_implied_probabilities([row["odds_over"], row["odds_under"]])[0])
        probability_rows.append(
            {
                "simple_prob_team_a": simple[0],
                "simple_prob_draw": simple[1],
                "simple_prob_team_b": simple[2],
                "overround": simple.sum() - 1.0,
                "normalized_prob_team_a": normalized[0],
                "normalized_prob_draw": normalized[1],
                "normalized_prob_team_b": normalized[2],
                "shin_prob_team_a": shin[0],
                "shin_prob_draw": shin[1],
                "shin_prob_team_b": shin[2],
                "shin_insider_fraction": insider_fraction,
                "prob_over": prob_over,
            }
        )
    loaded = pd.concat([odds.reset_index(drop=True), pd.DataFrame(probability_rows)], axis=1)
    loaded = _flag_bookmaker_disagreements(loaded)
    loaded["implied_prob_team_a"] = loaded["shin_prob_team_a"]
    loaded["implied_prob_draw"] = loaded["shin_prob_draw"]
    loaded["implied_prob_team_b"] = loaded["shin_prob_team_b"]
    flagged = loaded[~loaded["odds_status"].isin([ODDS_STATUS_OK, ODDS_STATUS_BLANK])]
    if not flagged.empty:
        report = flagged[["match_id", "team_a", "team_b", "bookmaker", "odds_status"]].to_string(index=False)
        if strict:
            raise ValueError(f"match_odds contains flagged rows:\n{report}")
        print(f"WARNING: excluding {len(flagged)} flagged match-odds row(s) from the consensus:\n{report}")
    return loaded


def _flag_bookmaker_disagreements(loaded: pd.DataFrame, tolerance: float = 0.15) -> pd.DataFrame:
    """Flag books whose de-vig probabilities sit far from the per-match median.

    Catches fat-fingered entries (e.g. 12.0 typed for 1.20) that still produce a
    plausible overround. With only two books neither can be arbitrated, so both are
    flagged and the match falls back to the model-only pick.
    """
    table = loaded.copy()
    shin_columns = ["shin_prob_team_a", "shin_prob_draw", "shin_prob_team_b"]
    usable = table[table["odds_status"].eq(ODDS_STATUS_OK)]
    for match_id, rows in usable.groupby("match_id"):
        if len(rows) < 2:
            continue
        deviations = (rows[shin_columns] - rows[shin_columns].median()).abs().max(axis=1)
        disagreeing = deviations[deviations > tolerance].index
        if len(disagreeing) == 0:
            continue
        flagged_indexes = rows.index if len(rows) == 2 else disagreeing
        table.loc[flagged_indexes, "odds_status"] = "suspect_bookmaker_disagreement"
    return table


CONSENSUS_COLUMNS = [
    "match_id", "date", "team_a", "team_b", "market_type", "bookmaker_count",
    "bookmakers", *PROBABILITY_COLUMNS, "normalized_prob_team_a", "normalized_prob_draw",
    "normalized_prob_team_b", "mean_overround", "mean_shin_insider_fraction",
    "over_under_line", "prob_over", "spread", "hours_before_kickoff", "source_timestamp",
    "market_favorite", "market_favorite_probability", "market_draw_probability",
    "market_strength_gap",
]


def build_match_odds_consensus(match_odds: pd.DataFrame) -> pd.DataFrame:
    """Average Shin-adjusted bookmaker probabilities into one market consensus."""
    odds = match_odds.copy()
    if "odds_status" in odds.columns:
        odds = odds[odds["odds_status"].eq(ODDS_STATUS_OK)].copy()
    if odds.empty:
        return pd.DataFrame(columns=CONSENSUS_COLUMNS)
    _validate_columns(
        odds,
        MATCH_ODDS_REQUIRED_COLUMNS + ["shin_prob_team_a", "shin_prob_draw", "shin_prob_team_b"],
        "loaded match odds",
    )
    for column in ("over_under_line", "prob_over", "spread", "hours_before_kickoff"):
        if column not in odds.columns:
            odds[column] = np.nan
    if "source_timestamp" not in odds.columns:
        odds["source_timestamp"] = pd.NaT
    group_columns = ["match_id", "date", "team_a", "team_b", "market_type"]
    consensus = odds.groupby(group_columns, dropna=False, as_index=False).agg(
        bookmaker_count=("bookmaker", "nunique"),
        bookmakers=("bookmaker", lambda books: ", ".join(sorted(set(map(str, books))))),
        prob_team_a=("shin_prob_team_a", "mean"),
        prob_draw=("shin_prob_draw", "mean"),
        prob_team_b=("shin_prob_team_b", "mean"),
        normalized_prob_team_a=("normalized_prob_team_a", "mean"),
        normalized_prob_draw=("normalized_prob_draw", "mean"),
        normalized_prob_team_b=("normalized_prob_team_b", "mean"),
        mean_overround=("overround", "mean"),
        mean_shin_insider_fraction=("shin_insider_fraction", "mean"),
        over_under_line=("over_under_line", "mean"),
        prob_over=("prob_over", "mean"),
        spread=("spread", "mean"),
        hours_before_kickoff=("hours_before_kickoff", "min"),
        source_timestamp=("source_timestamp", "max"),
    )
    total = consensus[PROBABILITY_COLUMNS].sum(axis=1)
    consensus[PROBABILITY_COLUMNS] = consensus[PROBABILITY_COLUMNS].div(total, axis=0)
    outcomes = consensus[PROBABILITY_COLUMNS].to_numpy()
    favorite_indexes = outcomes.argmax(axis=1)
    labels = consensus[["team_a", "team_b"]].copy()
    consensus["market_favorite"] = [
        team_a if index == 0 else "draw" if index == 1 else team_b
        for index, team_a, team_b in zip(favorite_indexes, labels["team_a"], labels["team_b"])
    ]
    consensus["market_favorite_probability"] = outcomes.max(axis=1)
    consensus["market_draw_probability"] = consensus["prob_draw"]
    consensus["market_strength_gap"] = (consensus["prob_team_a"] - consensus["prob_team_b"]).abs()
    return consensus


def load_futures_odds(path: str | Path = "data/raw/futures_odds.csv") -> pd.DataFrame:
    """Load optional tournament futures odds."""
    futures = _read_optional_csv(path, FUTURES_ODDS_COLUMNS, "futures odds")
    if futures.empty:
        return futures.assign(implied_probability=pd.Series(dtype=float))
    futures = futures.copy()
    futures["market"] = futures["market"].astype(str).str.strip().str.lower()
    unsupported = sorted(set(futures["market"]) - FUTURES_MARKETS)
    if unsupported:
        raise ValueError(f"Unsupported futures markets: {unsupported}. Expected: {sorted(FUTURES_MARKETS)}")
    futures["decimal_odds"] = _validate_decimal_odds(futures["decimal_odds"], "futures_odds.decimal_odds")
    futures["source_timestamp"] = pd.to_datetime(futures["source_timestamp"], errors="raise", utc=True)
    futures["implied_probability"] = 1.0 / futures["decimal_odds"]
    return futures


def build_futures_consensus(futures_odds: pd.DataFrame) -> pd.DataFrame:
    """Normalize each bookmaker futures book, average books, then renormalize consensus."""
    futures = futures_odds.copy()
    if futures.empty:
        return pd.DataFrame(columns=["market", "team", "bookmaker_count", "market_probability"])
    bookmaker_totals = futures.groupby(["market", "bookmaker"])["implied_probability"].transform("sum")
    futures["bookmaker_normalized_probability"] = futures["implied_probability"] / bookmaker_totals
    consensus = futures.groupby(["market", "team"], as_index=False).agg(
        bookmaker_count=("bookmaker", "nunique"),
        market_probability=("bookmaker_normalized_probability", "mean"),
    )
    consensus["market_probability"] = consensus["market_probability"] / consensus.groupby("market")["market_probability"].transform("sum")
    return consensus.sort_values(["market", "market_probability"], ascending=[True, False], kind="stable")


def blend_probability_vectors(model_probabilities, market_probabilities, market_weight: float = 0.50) -> np.ndarray:
    """Blend model and market probabilities without making the prior mandatory."""
    if not 0.0 <= market_weight <= 1.0:
        raise ValueError("market_weight must be between 0 and 1.")
    model = np.asarray(model_probabilities, dtype=float)
    market = np.asarray(market_probabilities, dtype=float)
    if model.shape != market.shape:
        raise ValueError("Model and market probabilities must have identical shapes.")
    if model.ndim not in {1, 2}:
        raise ValueError("Probability arrays must be one- or two-dimensional.")
    if np.any(~np.isfinite(model)) or np.any(~np.isfinite(market)) or np.any(model < 0) or np.any(market < 0):
        raise ValueError("Probabilities must be finite and non-negative.")
    blended = (1.0 - market_weight) * model + market_weight * market
    totals = blended.sum(axis=-1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Probability vectors must contain positive mass.")
    return blended / totals


def implied_total_goals_from_over_under(line: float, prob_over: float | None = None) -> float:
    """Return the market-implied expected total goals from an over/under quote.

    With de-vig over probability available, solves Poisson(T): P(total >= ceil(line)) = prob_over.
    Without it, the line itself is the best available total-goals anchor.
    """
    line = float(line)
    if not np.isfinite(line) or line <= 0:
        raise ValueError(f"over/under line must be positive and finite. Received: {line!r}")
    if prob_over is None or not np.isfinite(prob_over):
        return line
    if not 0.0 < prob_over < 1.0:
        raise ValueError(f"prob_over must be strictly between 0 and 1. Received: {prob_over!r}")
    goals_needed = math.floor(line) + 1

    def _objective(total: float) -> float:
        cumulative = sum(math.exp(-total) * total**k / math.factorial(k) for k in range(goals_needed))
        return (1.0 - cumulative) - prob_over

    low, high = 0.05, 9.0
    if _objective(low) > 0 or _objective(high) < 0:
        return line
    return float(brentq(_objective, low, high, xtol=1e-6))


def _wdl_probabilities(lambda_a: float, lambda_b: float, max_goals: int = 10, method: str = "independent") -> np.ndarray:
    summary = summarize_score_probs(lambda_a, lambda_b, max_goals=max_goals, method=method)
    return np.asarray([summary["team_a_win"], summary["draw"], summary["team_b_win"]], dtype=float)


def derive_market_lambdas(
    prob_team_a: float,
    prob_draw: float,
    prob_team_b: float,
    total_goals_target: float | None = None,
    supremacy_target: float | None = None,
    max_goals: int = 10,
    method: str = "independent",
) -> dict[str, float]:
    """Derive market-implied expected goals (lambda_a, lambda_b) from de-vig probabilities.

    1X2 probabilities pin the win/draw/win split; an over/under-implied total pins
    lambda_a + lambda_b and an Asian-handicap supremacy pins lambda_a - lambda_b.
    Anchors enter as soft penalties so inconsistent markets still produce a fit.
    """
    market = np.asarray([prob_team_a, prob_draw, prob_team_b], dtype=float)
    if np.any(~np.isfinite(market)) or np.any(market <= 0):
        raise ValueError(f"1X2 probabilities must be finite and positive. Received: {market}")
    market = market / market.sum()

    def _objective(params: np.ndarray) -> float:
        lambda_a, lambda_b = params
        error = float(np.square(_wdl_probabilities(lambda_a, lambda_b, max_goals, method) - market).sum())
        if total_goals_target is not None:
            error += 0.25 * (lambda_a + lambda_b - total_goals_target) ** 2
        if supremacy_target is not None:
            error += 0.25 * (lambda_a - lambda_b - supremacy_target) ** 2
        return error

    total_guess = total_goals_target if total_goals_target is not None else DEFAULT_TOTAL_GOALS_PRIOR
    supremacy_guess = (
        supremacy_target if supremacy_target is not None else 2.0 * float(market[0] - market[2])
    )
    starts = [
        ((total_guess + supremacy_guess) / 2.0, (total_guess - supremacy_guess) / 2.0),
        (total_guess / 2.0, total_guess / 2.0),
    ]
    best = None
    for start in starts:
        clipped = np.clip(start, *MARKET_LAMBDA_BOUNDS)
        fit = minimize(_objective, clipped, method="L-BFGS-B", bounds=[MARKET_LAMBDA_BOUNDS] * 2)
        if best is None or fit.fun < best.fun:
            best = fit
    lambda_a, lambda_b = (float(value) for value in best.x)
    fitted = _wdl_probabilities(lambda_a, lambda_b, max_goals, method)
    return {
        "market_lambda_a": lambda_a,
        "market_lambda_b": lambda_b,
        "market_lambda_fit_error": float(np.abs(fitted - market).max()),
        "market_lambda_converged": bool(best.success),
    }


def blend_market_lambdas(
    model_lambda_a: float,
    model_lambda_b: float,
    market_lambda_a: float,
    market_lambda_b: float,
    market_weight: float,
) -> tuple[float, float]:
    """Convex blend of model and market expected goals."""
    if not 0.0 <= market_weight <= 1.0:
        raise ValueError("market_weight must be between 0 and 1.")
    values = [model_lambda_a, model_lambda_b, market_lambda_a, market_lambda_b]
    if any(not np.isfinite(value) or value < 0 for value in values):
        raise ValueError(f"Expected goals must be finite and non-negative. Received: {values}")
    return (
        (1.0 - market_weight) * model_lambda_a + market_weight * market_lambda_a,
        (1.0 - market_weight) * model_lambda_b + market_weight * market_lambda_b,
    )


def load_market_blend_gate(path: str | Path = "outputs/market_blend_gate.json") -> dict[str, object]:
    """Load the backtest promotion-gate decision; the gate fails closed when absent."""
    gate_path = Path(path)
    if not gate_path.exists():
        return {
            "promote": False,
            "promoted_strategy": None,
            "reasons_not_promoted": [
                "market blend gate has not been evaluated; run scripts/run_backtest.py with historical odds"
            ],
        }
    decision = json.loads(gate_path.read_text(encoding="utf-8"))
    decision.setdefault("promote", False)
    decision.setdefault("promoted_strategy", "market_blended_optimizer" if decision["promote"] else None)
    decision.setdefault("reasons_not_promoted", [])
    return decision


def add_market_lambdas_to_consensus(consensus: pd.DataFrame, method: str = "independent") -> pd.DataFrame:
    """Add market-implied lambda columns to a match-odds consensus table."""
    table = consensus.copy()
    lambda_columns = ["market_lambda_a", "market_lambda_b", "market_lambda_fit_error", "market_lambda_converged"]
    if table.empty:
        for column in lambda_columns:
            table[column] = pd.Series(dtype=float)
        return table
    fits = []
    for row in table.itertuples(index=False):
        total_target = None
        line = getattr(row, "over_under_line", np.nan)
        if pd.notna(line):
            prob_over = getattr(row, "prob_over", np.nan)
            total_target = implied_total_goals_from_over_under(float(line), float(prob_over) if pd.notna(prob_over) else None)
        spread = getattr(row, "spread", np.nan)
        # The spread column is the Asian-handicap line applied to team_a (negative when
        # team_a is favored), so the implied supremacy lambda_a - lambda_b is its negation.
        supremacy_target = -float(spread) if pd.notna(spread) else None
        fits.append(
            derive_market_lambdas(
                row.prob_team_a,
                row.prob_draw,
                row.prob_team_b,
                total_goals_target=total_target,
                supremacy_target=supremacy_target,
                method=method,
            )
        )
    return pd.concat([table.reset_index(drop=True), pd.DataFrame(fits)], axis=1)


def blend_futures_prior(
    model_probabilities: pd.DataFrame,
    futures_consensus: pd.DataFrame,
    market: str,
    model_probability_column: str,
    candidate_column: str = "team",
    market_weight: float = 0.50,
) -> pd.DataFrame:
    """Return an optional normalized futures prior blend for external review."""
    model = model_probabilities[[candidate_column, model_probability_column]].rename(
        columns={candidate_column: "candidate", model_probability_column: "model_probability"}
    )
    market_rows = futures_consensus[futures_consensus["market"].eq(market)][["team", "market_probability"]].rename(
        columns={"team": "candidate"}
    )
    blended = model.merge(market_rows, on="candidate", how="outer").fillna(0.0)
    probability_values = blend_probability_vectors(
        blended["model_probability"].to_numpy(),
        blended["market_probability"].to_numpy(),
        market_weight=market_weight,
    )
    blended["blended_probability"] = probability_values
    blended["market"] = market
    blended["market_weight"] = market_weight
    return blended.sort_values("blended_probability", ascending=False, kind="stable")


def _model_match_probabilities(rows: pd.DataFrame) -> pd.DataFrame:
    model = rows.copy()
    lambda_a = "calibrated_expected_goals_a" if "calibrated_expected_goals_a" in model else "expected_goals_a"
    lambda_b = "calibrated_expected_goals_b" if "calibrated_expected_goals_b" in model else "expected_goals_b"
    probabilities = [
        summarize_score_probs(float(home), float(away), max_goals=7)
        for home, away in zip(model[lambda_a], model[lambda_b])
    ]
    model["model_prob_team_a"] = [row["team_a_win"] for row in probabilities]
    model["model_prob_draw"] = [row["draw"] for row in probabilities]
    model["model_prob_team_b"] = [row["team_b_win"] for row in probabilities]
    return model


def _swap_consensus_orientation(consensus: pd.DataFrame) -> pd.DataFrame:
    """Return the consensus with team_a/team_b perspective swapped consistently."""
    swapped = consensus.copy()
    swap_pairs = [
        ("team_a", "team_b"),
        ("prob_team_a", "prob_team_b"),
        ("normalized_prob_team_a", "normalized_prob_team_b"),
        ("market_lambda_a", "market_lambda_b"),
    ]
    for left_column, right_column in swap_pairs:
        if left_column in swapped.columns and right_column in swapped.columns:
            swapped[[left_column, right_column]] = swapped[[right_column, left_column]].to_numpy()
    if "spread" in swapped.columns:
        swapped["spread"] = -swapped["spread"]
    return swapped


def attach_market_consensus(matches: pd.DataFrame, consensus: pd.DataFrame) -> pd.DataFrame:
    """Attach market consensus rows to fixtures, matching either team orientation."""
    if matches.empty or consensus.empty:
        return pd.DataFrame()
    left = matches.copy()
    left["date"] = pd.to_datetime(left["date"], errors="raise").dt.normalize()
    market = consensus.copy()
    market["date"] = pd.to_datetime(market["date"], errors="raise").dt.normalize()
    oriented = pd.concat([market, _swap_consensus_orientation(market)], ignore_index=True)
    oriented = oriented.drop_duplicates(subset=["date", "team_a", "team_b"], keep="first")
    drop_columns = [column for column in ("match_id", "market_type") if column in oriented.columns]
    oriented = oriented.drop(columns=drop_columns)
    attached = left.merge(oriented, on=["date", "team_a", "team_b"], how="left", suffixes=("", "_market"))
    return attached[attached["prob_team_a"].notna()].copy()


# Backward-compatible alias for earlier diagnostic callers.
_attach_market_consensus = attach_market_consensus


def _outcome_index(goals_a: int, goals_b: int) -> int:
    return 0 if goals_a > goals_b else 1 if goals_a == goals_b else 2


def _score_policy(probabilities: np.ndarray, goals_a: np.ndarray, goals_b: np.ndarray) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=float)
    actual = np.asarray([_outcome_index(int(home), int(away)) for home, away in zip(goals_a, goals_b)])
    predicted = probabilities.argmax(axis=1)
    one_hot = np.eye(3)[actual]
    selected = np.clip(probabilities[np.arange(len(probabilities)), actual], 1e-12, 1.0)
    scorelines = [(1, 0) if outcome == 0 else (1, 1) if outcome == 1 else (0, 1) for outcome in predicted]
    penka = [
        score_prediction(pred_a, pred_b, int(actual_a), int(actual_b), GROUP)
        for (pred_a, pred_b), actual_a, actual_b in zip(scorelines, goals_a, goals_b)
    ]
    confidence = probabilities.max(axis=1)
    correct = predicted == actual
    bins = np.linspace(0.0, 1.0, 6)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        selected_rows = (confidence >= low) & (confidence <= high if high == 1 else confidence < high)
        if selected_rows.any():
            ece += selected_rows.mean() * abs(confidence[selected_rows].mean() - correct[selected_rows].mean())
    return {
        "matches": len(probabilities),
        "total_penka_points": int(sum(penka)),
        "avg_penka_points": float(np.mean(penka)),
        "wdl_log_loss": float(-np.log(selected).mean()),
        "wdl_brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "calibration_error": float(ece),
    }


def _historical_odds_backtest(backtests: pd.DataFrame, consensus: pd.DataFrame) -> pd.DataFrame:
    """Compare optional market diagnostics on historical rows with matched odds only."""
    columns = ["scope", "year", "strategy", "available", "status", "matches", "total_penka_points", "avg_penka_points", "wdl_log_loss", "wdl_brier", "calibration_error"]
    if backtests.empty or consensus.empty:
        return pd.DataFrame([{"scope": "historical_matched_odds", "year": "", "strategy": "odds_unavailable", "available": False, "status": "No historical odds rows matched; comparison skipped."}], columns=columns)
    model = _model_match_probabilities(backtests)
    matched = _attach_market_consensus(model, consensus)
    if matched.empty:
        return pd.DataFrame([{"scope": "historical_matched_odds", "year": "", "strategy": "odds_unavailable", "available": False, "status": "No historical odds rows matched; comparison skipped."}], columns=columns)
    scored_rows = []
    predicted_by_strategy: dict[str, list[pd.DataFrame]] = {}
    for year, holdout in matched.groupby("backtest_year", sort=True):
        model_probs = holdout[["model_prob_team_a", "model_prob_draw", "model_prob_team_b"]].to_numpy()
        market_probs = holdout[PROBABILITY_COLUMNS].to_numpy()
        strategies = {
            "no_odds_model": model_probs,
            "odds_benchmark_only": market_probs,
            "odds_as_blended_prior": blend_probability_vectors(model_probs, market_probs),
        }
        earlier = matched[matched["date"] < holdout["date"].min()]
        if len(earlier) >= 30 and earlier[["goals_a", "goals_b"]].apply(lambda row: _outcome_index(row.iloc[0], row.iloc[1]), axis=1).nunique() >= 2:
            train_x = np.column_stack(
                [
                    earlier[["model_prob_team_a", "model_prob_draw", "model_prob_team_b"]].to_numpy(),
                    earlier[PROBABILITY_COLUMNS].to_numpy(),
                ]
            )
            train_y = np.asarray([_outcome_index(a, b) for a, b in zip(earlier["goals_a"], earlier["goals_b"])])
            classifier = LogisticRegression(max_iter=300, random_state=42).fit(train_x, train_y)
            raw = classifier.predict_proba(np.column_stack([model_probs, market_probs]))
            feature_probs = np.zeros((len(holdout), 3))
            feature_probs[:, classifier.classes_.astype(int)] = raw
            strategies["odds_as_model_feature"] = feature_probs
        for strategy, probabilities in strategies.items():
            metrics = _score_policy(probabilities, holdout["goals_a"].to_numpy(), holdout["goals_b"].to_numpy())
            scored_rows.append({"scope": "historical_matched_odds", "year": int(year), "strategy": strategy, "available": True, "status": "evaluated", **metrics})
            frame = holdout[["goals_a", "goals_b"]].copy()
            frame["probabilities"] = list(probabilities)
            predicted_by_strategy.setdefault(strategy, []).append(frame)
    for strategy, parts in predicted_by_strategy.items():
        rows = pd.concat(parts, ignore_index=True)
        metrics = _score_policy(np.vstack(rows["probabilities"]), rows["goals_a"].to_numpy(), rows["goals_b"].to_numpy())
        scored_rows.append({"scope": "historical_matched_odds", "year": "overall", "strategy": strategy, "available": True, "status": "evaluated", **metrics})
    return pd.DataFrame(scored_rows, columns=columns)


def _build_2026_market_features(final_predictions: pd.DataFrame, consensus: pd.DataFrame) -> pd.DataFrame:
    """Compare optional market probabilities with generated 2026 model probabilities."""
    columns = [
        "match_id", "date", "team_a", "team_b", "bookmaker_count", "prob_team_a", "prob_draw",
        "prob_team_b", "market_favorite", "market_favorite_probability", "market_draw_probability",
        "market_strength_gap", "model_prob_team_a", "model_prob_draw", "model_prob_team_b",
        "model_favorite", "max_absolute_probability_gap", "large_disagreement",
    ]
    if final_predictions.empty or consensus.empty:
        return pd.DataFrame(columns=columns)
    predictions = final_predictions.rename(
        columns={"p_team_a_win": "model_prob_team_a", "p_draw": "model_prob_draw", "p_team_b_win": "model_prob_team_b"}
    )
    attached = _attach_market_consensus(predictions, consensus)
    if attached.empty:
        return pd.DataFrame(columns=columns)
    model_probs = attached[["model_prob_team_a", "model_prob_draw", "model_prob_team_b"]].to_numpy()
    market_probs = attached[PROBABILITY_COLUMNS].to_numpy()
    attached["model_favorite"] = [
        team_a if index == 0 else "draw" if index == 1 else team_b
        for index, team_a, team_b in zip(model_probs.argmax(axis=1), attached["team_a"], attached["team_b"])
    ]
    attached["max_absolute_probability_gap"] = np.abs(model_probs - market_probs).max(axis=1)
    attached["large_disagreement"] = attached["max_absolute_probability_gap"].ge(0.15) | attached["model_favorite"].ne(attached["market_favorite"])
    return attached[columns].copy()


def _build_futures_disagreements(futures: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Return futures-vs-model disagreements where comparable outputs exist."""
    rows = []
    mappings = {
        "champion": ("champion_probabilities.csv", "team", "champion_probability"),
        "runner_up": ("runner_up_probabilities.csv", "team", "runner_up_probability"),
        "top_scorer": ("top_scorer_probabilities.csv", "player", "probability"),
        "golden_boot": ("top_scorer_probabilities.csv", "player", "probability"),
    }
    for market, market_rows in futures.groupby("market"):
        if market not in mappings:
            continue
        file_name, candidate_column, probability_column = mappings[market]
        path = output_dir / file_name
        if not path.exists():
            continue
        model = pd.read_csv(path)
        if not {candidate_column, probability_column}.issubset(model):
            continue
        merged = market_rows.merge(model[[candidate_column, probability_column]], left_on="team", right_on=candidate_column, how="left")
        for row in merged.itertuples(index=False):
            model_probability = getattr(row, probability_column) if pd.notna(getattr(row, probability_column)) else np.nan
            gap = float(row.market_probability - model_probability) if np.isfinite(model_probability) else np.nan
            rows.append(
                {
                    "comparison_type": "futures",
                    "market": market,
                    "match_id": "",
                    "candidate": row.team,
                    "model_probability": model_probability,
                    "market_probability": row.market_probability,
                    "probability_gap": gap,
                    "large_disagreement": bool(np.isfinite(gap) and abs(gap) >= 0.05),
                }
            )
    return pd.DataFrame(rows)


def validate_odds_priors_sanity() -> None:
    """Prove normalization, overround, consensus, futures, and Shin behavior."""
    odds = [1.95, 3.50, 4.20]
    simple = simple_implied_probabilities(odds)
    normalized = normalized_implied_probabilities(odds)
    shin, insider_fraction = shin_adjusted_probabilities(odds)
    assert simple.sum() > 1.0
    assert calculate_overround(odds) > 0.0
    assert np.isclose(normalized.sum(), 1.0)
    assert np.isclose(shin.sum(), 1.0)
    assert np.all(shin >= 0)
    assert insider_fraction > 0.0
    assert np.isclose(blend_probability_vectors([0.6, 0.4], [0.4, 0.6]).sum(), 1.0)
    try:
        simple_implied_probabilities([1.0, 2.0, 3.0])
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid decimal odds must fail loudly.")
    known_wdl = _wdl_probabilities(1.8, 0.9)
    derived = derive_market_lambdas(*known_wdl)
    assert abs(derived["market_lambda_a"] - 1.8) < 0.1, derived
    assert abs(derived["market_lambda_b"] - 0.9) < 0.1, derived
    pinned = derive_market_lambdas(*known_wdl, total_goals_target=2.7, supremacy_target=0.9)
    assert abs(pinned["market_lambda_a"] + pinned["market_lambda_b"] - 2.7) < 0.2, pinned
    blended_lambdas = blend_market_lambdas(1.0, 1.0, 2.0, 0.5, 0.5)
    assert np.isclose(blended_lambdas[0], 1.5) and np.isclose(blended_lambdas[1], 0.75)
    assert implied_total_goals_from_over_under(2.5, 0.5) > 2.0
    assert implied_total_goals_from_over_under(2.5) == 2.5


def run_optional_odds_diagnostics(
    output_dir: str | Path = "outputs",
    match_odds_path: str | Path = "data/raw/match_odds.csv",
    futures_odds_path: str | Path = "data/raw/futures_odds.csv",
) -> dict[str, object]:
    """Write optional market diagnostics without changing production predictions."""
    validate_odds_priors_sanity()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    loaded_match_odds = load_match_odds(match_odds_path)
    match_consensus = build_match_odds_consensus(loaded_match_odds)
    loaded_futures = load_futures_odds(futures_odds_path)
    futures_consensus = build_futures_consensus(loaded_futures)
    backtest_path = output_path / "backtest_predictions.csv"
    backtests = pd.read_csv(backtest_path) if backtest_path.exists() else pd.DataFrame()
    odds_backtest = _historical_odds_backtest(backtests, match_consensus)
    odds_backtest.to_csv(output_path / OUTPUT_FILES["backtest"], index=False)
    final_path = output_path / "final_predictions.csv"
    final_predictions = pd.read_csv(final_path) if final_path.exists() else pd.DataFrame()
    features = _build_2026_market_features(final_predictions, match_consensus)
    features.to_csv(output_path / OUTPUT_FILES["features"], index=False)
    match_disagreements = features[features.get("large_disagreement", pd.Series(False, index=features.index))].copy()
    if not match_disagreements.empty:
        match_disagreements = match_disagreements.assign(
            comparison_type="match",
            market="match_result",
            candidate=match_disagreements["team_a"] + " vs " + match_disagreements["team_b"],
            model_probability=np.nan,
            market_probability=np.nan,
            probability_gap=match_disagreements["max_absolute_probability_gap"],
        )[["comparison_type", "market", "match_id", "candidate", "model_probability", "market_probability", "probability_gap", "large_disagreement"]]
    else:
        match_disagreements = pd.DataFrame(columns=["comparison_type", "market", "match_id", "candidate", "model_probability", "market_probability", "probability_gap", "large_disagreement"])
    futures_disagreements = _build_futures_disagreements(futures_consensus, output_path)
    disagreements = pd.concat([match_disagreements, futures_disagreements], ignore_index=True)
    disagreements.to_csv(output_path / OUTPUT_FILES["disagreements"], index=False)

    evaluated = odds_backtest[odds_backtest["available"].eq(True)] if "available" in odds_backtest else pd.DataFrame()
    overall = evaluated[evaluated["year"].astype(str).eq("overall")] if not evaluated.empty else pd.DataFrame()
    improved = False
    if not overall.empty and "no_odds_model" in set(overall["strategy"]):
        baseline = float(overall.loc[overall["strategy"].eq("no_odds_model"), "avg_penka_points"].iloc[0])
        improved = bool(overall.loc[overall["strategy"].ne("no_odds_model"), "avg_penka_points"].gt(baseline).any())
    summary_lines = [
        "Optional Odds Prior Summary",
        "===========================",
        "Odds remain optional. Final prediction files were not modified.",
        f"match_odds_file_exists: {Path(match_odds_path).exists()}",
        f"match_odds_rows_loaded: {len(loaded_match_odds)}",
        f"match_odds_consensus_rows: {len(match_consensus)}",
        f"futures_odds_file_exists: {Path(futures_odds_path).exists()}",
        f"futures_odds_rows_loaded: {len(loaded_futures)}",
        f"futures_consensus_rows: {len(futures_consensus)}",
        "shin_adjustment_implemented: True",
        "shin_adjustment_sanity_check_passed: True",
        f"historical_odds_rows_evaluated: {int(overall['matches'].max()) if not overall.empty else 0}",
        f"odds_improved_any_historical_backtest: {improved if not overall.empty else 'unavailable'}",
        f"market_prior_features_2026_rows: {len(features)}",
        f"large_model_market_disagreements: {len(disagreements)}",
        "champion_probability_calibration: unavailable unless historical futures snapshots and realized tournaments are supplied.",
        "",
        (
            "Recommendation: use populated odds as a benchmark and optional blended prior; reopen model-feature use only after leakage-safe historical odds backtests."
            if len(loaded_match_odds) or len(loaded_futures)
            else "Recommendation: odds support is ready but diagnostic-only. Populate sourced odds before using market benchmarks, features, or priors."
        ),
    ]
    (output_path / OUTPUT_FILES["summary"]).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {
        "match_odds": loaded_match_odds,
        "match_consensus": match_consensus,
        "futures_odds": loaded_futures,
        "futures_consensus": futures_consensus,
        "odds_prior_backtest": odds_backtest,
        "market_prior_features": features,
        "model_vs_market_disagreements": disagreements,
    }


if __name__ == "__main__":
    outputs = run_optional_odds_diagnostics()
    print((Path("outputs") / OUTPUT_FILES["summary"]).read_text(encoding="utf-8"))
