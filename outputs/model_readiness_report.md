# World Cup Predictor Readiness Report

## Rules Now Implemented

- Penka match scoring is phase-specific and mutually exclusive.
- Knockout predictions score the 90-minute result only. Extra time and penalties do not count.
- Knockout score recommendations allow draws.
- Group recommendations are refreshable match by match because Penka accepts predictions until 1 minute before kickoff.
- Special prediction points are Champion 25, Runner-up 15, Top Scorer 35, MVP 10, and Golden Glove 10.
- Special predictions lock before the Penka starts.

## Readiness Status

| Area | Status | Notes |
| --- | --- | --- |
| Group match predictions | Development ready, refresh before match | Use `penka_group_predictions.csv` as a baseline. Refresh rankings, lineups, injuries, suspensions, rotation, goalkeeper changes, and incentives before kickoff. |
| Knockout predictions | Script ready, fixtures pending | Populate `data/manual/knockout_round_fixtures.csv` as each confirmed Penka round appears. Advice uses calibrated 90-minute probabilities and phase-specific EV. |
| Awards | Not submission ready | Real player data is still missing. Sample-player outputs only prove that the pipeline runs. |
| Ranking data | Refresh recommended | The bundled ranking source currently ends on `2024-09-19`; refresh rankings before submission and before each match where possible. |
| Tournament simulator | Ready for structural use | Uses the official FIFA World Cup 26 third-place assignment matrix and the 48-team format. |
| Knockout tie resolution | Simulator-only | Tournament paths resolve tied 90-minute knockout simulations stochastically. Penka match predictions remain 90-minute scores and may be draws. |
| Player data validation | Blocked for final awards | Replace sample players, player stats, goalkeeper stats, penalty takers, injuries, and squad status with real files. |

## Warnings

1. The old `favorite_1_0` conclusion was based on an incorrect scoring rule and is invalid until the real Penka backtest reports are reviewed.
2. Old outputs generated under the previous scoring formula should not be submitted.
3. Any award output based on sample player data is marked `NOT SUBMISSION READY - SAMPLE PLAYER DATA`.
4. Pre-tournament group predictions are not final advice. Matchday 3 and confirmed lineup changes especially require a refresh.

## Current Output Trust

| Output | Trust Level |
| --- | --- |
| `penka_group_predictions.csv` / `.xlsx` | Refreshable development recommendation |
| `penka_scoring_strategy_comparison.csv` | Historical diagnostic under real Penka scoring |
| `team_path_simulations.csv` | Structurally trusted tournament simulation |
| `champion_probabilities.csv` | Tournament-model output |
| `runner_up_probabilities.csv` | Actual runner-up flag output, not finalist probability |
| `top_scorer_probabilities.csv` | Sample-data test output only |
| `mvp_probabilities.csv` | Sample-data test output only |
| `golden_glove_probabilities.csv` | Sample-data test output only |
