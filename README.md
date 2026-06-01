# worldcup-predictor

Google Colab-compatible World Cup 2026 prediction project.

This repository will build a transparent, CPU-friendly heuristic model for match scores, match results, goal difference, tournament outcomes, player awards, and company-game expected-points strategy.

The project uses the 2026 format: 48 teams, 12 groups of 4, top 2 from each group plus 8 best third-placed teams, and a Round of 32. It intentionally avoids deep learning, AutoML, heavy grid search, and overfitted award-weight tuning.

Core workflow:

1. Load raw team, player, and historical award data.
2. Build transparent match and player features.
3. Predict scores with a simple Poisson-style model.
4. Simulate the 2026 tournament format.
5. Estimate award probabilities.
6. Optimize picks for expected Colombia Tech Fest Penka points.

Match recommendations use the real phase-specific, mutually exclusive Penka scoring rules. Knockout advice is for the 90-minute score only and allows draws. Build refreshable group-stage recommendations with:

```powershell
python scripts/build_penka_group_predictions.py
```

Populate `data/manual/match_adjustments.csv` with human-reviewed lineup or availability adjustments before kickoff. Once Penka publishes a confirmed knockout round, populate `data/manual/knockout_round_fixtures.csv` and run:

```powershell
python scripts/run_knockout_round.py
```

Pre-tournament group predictions are baselines for review, not final submission advice. Award outputs are also development-only until the sample player files are replaced with real squad data.
