"""Expected-points strategy for independent company-game award picks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_SCORING_RULES_PATH = "data/raw/company_scoring_rules.csv"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_MIN_EV_RATIO = 0.85
DEFAULT_TARGET_EV_RATIO = 0.90

PROBABILITY_FILES = {
    "champion": ("outputs/champion_probabilities.csv", "champion_probability"),
    "runner_up": ("outputs/runner_up_probabilities.csv", "runner_up_probability"),
    "top_scorer": ("outputs/top_scorer_probabilities.csv", "probability"),
    "mvp": ("outputs/mvp_probabilities.csv", "probability"),
    "golden_glove": ("outputs/golden_glove_probabilities.csv", "probability"),
}


def validate_columns(df: pd.DataFrame, required_columns, dataset_name: str) -> None:
    """Raise a clear error if an award input is missing required columns."""
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )


def load_scoring_rules(path: str | Path = DEFAULT_SCORING_RULES_PATH) -> pd.DataFrame:
    """Load and validate company-game scoring rules."""
    rules_path = Path(path)
    if not rules_path.exists():
        raise FileNotFoundError(f"Scoring rules file not found: {rules_path}")
    rules = pd.read_csv(rules_path)
    validate_columns(rules, ["category", "points"], "company scoring rules")
    rules["category"] = rules["category"].astype(str).str.strip()
    rules["points"] = pd.to_numeric(rules["points"], errors="raise")
    if rules["category"].duplicated().any():
        raise ValueError("company scoring rules contains duplicate categories.")
    return rules


def expected_award_points(probability: float, award: str, scoring_rules: pd.DataFrame | None = None) -> float:
    """Convert one award probability into company-game expected points."""
    rules = load_scoring_rules() if scoring_rules is None else scoring_rules
    points_by_category = rules.set_index("category")["points"].to_dict()
    if award not in points_by_category:
        raise KeyError(f"Unknown award category: {award}")
    return float(probability) * float(points_by_category[award])


def load_probability_table(category: str, path: str | Path, probability_column: str) -> pd.DataFrame:
    """Load one category probability table without mixing independent awards."""
    probability_path = Path(path)
    if not probability_path.exists():
        raise FileNotFoundError(f"{category} probability file not found: {probability_path}")
    probabilities = pd.read_csv(probability_path)
    validate_columns(probabilities, ["team" if category in {"champion", "runner_up"} else "player", probability_column], category)
    if category in {"champion", "runner_up"}:
        probabilities["candidate"] = probabilities["team"]
    else:
        probabilities["candidate"] = probabilities["player"]
    probabilities["probability"] = pd.to_numeric(probabilities[probability_column], errors="raise")
    if (probabilities["probability"] < 0).any():
        raise ValueError(f"{category} probabilities must not be negative.")
    probabilities["category"] = category
    probabilities["source_probability_column"] = probability_column
    keep_columns = ["category", "candidate", "probability", "source_probability_column"]
    if "team" in probabilities.columns:
        keep_columns.insert(2, "team")
    return probabilities[keep_columns]


def calculate_award_expected_points(
    scoring_rules: pd.DataFrame | None = None,
    probability_files: dict[str, tuple[str | Path, str]] | None = None,
) -> pd.DataFrame:
    """Calculate expected points independently for every award candidate."""
    rules = load_scoring_rules() if scoring_rules is None else scoring_rules.copy()
    validate_columns(rules, ["category", "points"], "company scoring rules")
    files = PROBABILITY_FILES if probability_files is None else probability_files
    rows = []
    for category, (path, probability_column) in files.items():
        table = load_probability_table(category, path, probability_column)
        category_points = rules.loc[rules["category"].eq(category), "points"]
        if category_points.empty:
            raise ValueError(f"No scoring rule found for category: {category}")
        table["category_points"] = float(category_points.iloc[0])
        table["expected_points"] = table["probability"] * table["category_points"]
        rows.append(table)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["category", "expected_points", "probability"],
        ascending=[True, False, False],
        kind="stable",
    ).reset_index(drop=True)


def select_safe_pick(category_candidates: pd.DataFrame) -> pd.Series:
    """Return the highest expected-points candidate for one award."""
    validate_columns(category_candidates, ["candidate", "expected_points"], "award candidates")
    if category_candidates.empty:
        raise ValueError("award candidates must not be empty.")
    return category_candidates.sort_values(
        ["expected_points", "probability"],
        ascending=[False, False],
        kind="stable",
    ).iloc[0]


def select_strategic_pick(
    category_candidates: pd.DataFrame,
    min_ev_ratio: float = DEFAULT_MIN_EV_RATIO,
    target_ev_ratio: float = DEFAULT_TARGET_EV_RATIO,
) -> pd.Series:
    """Return a conservative differentiated pick or the safe pick."""
    if not 0 < min_ev_ratio <= target_ev_ratio <= 1:
        raise ValueError("Require 0 < min_ev_ratio <= target_ev_ratio <= 1.")
    safe_pick = select_safe_pick(category_candidates)
    safe_ev = float(safe_pick["expected_points"])
    candidates = category_candidates.copy()
    candidates["ev_ratio"] = 1.0 if safe_ev <= 0 else candidates["expected_points"] / safe_ev
    alternatives = candidates[
        candidates["candidate"].ne(safe_pick["candidate"])
        & candidates["ev_ratio"].ge(min_ev_ratio)
    ].copy()
    if alternatives.empty:
        strategic_pick = safe_pick.copy()
        strategic_pick["ev_ratio"] = 1.0
        return strategic_pick

    preferred = alternatives[alternatives["ev_ratio"].ge(target_ev_ratio)]
    eligible = preferred if not preferred.empty else alternatives
    return eligible.sort_values(
        ["expected_points", "probability"],
        ascending=[False, False],
        kind="stable",
    ).iloc[0]


def build_award_picks(expected_points: pd.DataFrame) -> pd.DataFrame:
    """Build safe and strategic recommendations for every independent award."""
    picks = []
    for category, candidates in expected_points.groupby("category", sort=False):
        safe = select_safe_pick(candidates)
        strategic = select_strategic_pick(candidates)
        safe_ev = float(safe["expected_points"])
        strategic_ev = float(strategic["expected_points"])
        picks.append(
            {
                "category": category,
                "category_points": float(safe["category_points"]),
                "safe_pick": safe["candidate"],
                "safe_probability": float(safe["probability"]),
                "safe_expected_points": safe_ev,
                "strategic_pick": strategic["candidate"],
                "strategic_probability": float(strategic["probability"]),
                "strategic_expected_points": strategic_ev,
                "strategic_ev_ratio": 1.0 if safe_ev <= 0 else strategic_ev / safe_ev,
                "recommendation": "safe" if strategic["candidate"] == safe["candidate"] else "strategic alternative",
            }
        )
    return pd.DataFrame(picks)


def save_award_strategy_outputs(
    expected_points: pd.DataFrame,
    final_picks: pd.DataFrame | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write CSV and Excel award-strategy outputs."""
    picks = build_award_picks(expected_points) if final_picks is None else final_picks.copy()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    expected_points_path = output_path / "award_expected_points.csv"
    picks_path = output_path / "final_award_picks.csv"
    workbook_path = output_path / "award_strategy_recommendations.xlsx"
    expected_points.to_csv(expected_points_path, index=False)
    picks.to_csv(picks_path, index=False)
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        picks.to_excel(writer, sheet_name="recommendations", index=False)
        expected_points.to_excel(writer, sheet_name="expected_points", index=False)
    return {
        "award_expected_points": expected_points_path,
        "final_award_picks": picks_path,
        "award_strategy_recommendations": workbook_path,
    }
