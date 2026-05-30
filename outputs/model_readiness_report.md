# World Cup Predictor Readiness Report

## 1. Current Status

| Area | Status | Notes |
| --- | --- | --- |
| Match pipeline | Working | Generates leakage-safe group-stage match predictions and Excel export. |
| Backtesting | Working with caution | 2014, 2018, and 2022 walk-forward backtests run successfully. |
| Tournament simulation | Working provisionally | Uses the 48-team 2026 format and passes structural checks. |
| Awards pipeline | Working for development | Reads real tournament-path simulations and generates award outputs. |
| Player data | Blocked for final use | Current awards use sample player files rather than complete real squad data. |
| Third-place matrix | Blocked for final use | Official Round-of-32 third-place assignment matrix is still missing. |
| Aggressive picks | Working | Realism filter keeps alternatives nearby and retains at least 95% of safe expected value. |
| Final submission readiness | Not ready | Remaining blockers affect tournament probabilities and award recommendations. |

## 2. Blocking Issues

1. The official 2026 Round-of-32 third-place assignment matrix is missing. Tournament simulations currently use the deterministic development fallback.
2. Real player data is missing. Award predictions are based on sample player files and are not valid for final submission.
3. The optimized expected-points match strategy does not beat the `favorite_1_0` baseline overall in historical backtesting.

## 3. Current Trusted Outputs

| Output | Trust Level | Notes |
| --- | --- | --- |
| `final_predictions.csv` | Provisional match-level output | Suitable for reviewing group-stage safe picks and realistic aggressive alternatives. |
| `final_predictions.xlsx` | Provisional match-level output | Excel version of the match prediction report. |
| `team_path_simulations.csv` | Structurally trusted, provisional probabilities | Uses 20,000 simulations and the 2026 format, but still uses the third-place fallback. |
| `champion_probabilities.csv` | Provisional only | Structurally valid, but rerun after the official third-place matrix is added. |
| `runner_up_probabilities.csv` | Provisional only | Uses actual runner-up outcomes, but rerun after the official matrix is added. |
| `top_scorer_probabilities.csv` | Sample-data test output only | Confirms that the award pipeline works end to end. |
| `mvp_probabilities.csv` | Sample-data test output only | Confirms that the award pipeline works end to end. |
| `golden_glove_probabilities.csv` | Sample-data test output only | Confirms that the award pipeline works end to end. |

## 4. Current Untrusted Outputs

- Award predictions and award expected-points recommendations are untrusted for final submission because they use sample player data.
- Final champion and runner-up probabilities are not submission-ready while the deterministic third-place assignment fallback is active.

## 5. Next Actions

1. Commit or ignore `data/raw/data_manifest.json`.
2. Populate `data/raw/third_place_assignment_matrix.csv` with the official 2026 assignment matrix.
3. Improve the optimized strategy or benchmark final match selections against `favorite_1_0`.
4. Replace sample player, player-stat, goalkeeper, penalty-taker, injury, and squad-status files with real data.
5. Rerun `python scripts/run_final_predictions.py`.
6. Rerun `python scripts/run_awards_prediction.py`.

