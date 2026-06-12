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

## Market-odds workflow (manual entry, gated candidate strategy)

Betting-market odds are an optional prior. Blended picks are always produced as a
**candidate** comparison; they only replace the validated baseline picks when the
backtest promotion gate (`outputs/market_blend_gate.json`) says they beat
`favorite_2_1` and `favorite_1_0` on historical odds. Matches with blank odds fall
back to the model-only pick and are badged "no odds" in `outputs/pick_sheet.csv`.

Weekly batch:

```powershell
python scripts/generate_odds_sheet.py --from 2026-06-11 --days 7 --bookmakers "Bet365,Pinnacle"
# fill the sheet by hand, paste rows into data/raw/match_odds.csv, then:
python scripts/run_final_predictions.py
```

Per-match update ~2 hours before kickoff (overwrite that match's odds row first):

```powershell
python scripts/reoptimize_match.py --match 37
```

The script stamps `source_timestamp = now()` and computes hours-before-kickoff
itself, so freshness provenance is honest by construction.

Historical odds for the gate (requires `BALLDONTLIE_API_KEY` in the environment or
the git-ignored root `.env` — never commit keys):

```powershell
python scripts/fetch_odds_historical.py
python scripts/run_backtest.py
```

BALLDONTLIE is usable for current 2026 match odds and tournament futures during
the GOAT-tier trial. It is not usable for the 2018/2022 historical promotion gate:
season-wide and per-match probes confirmed on June 11, 2026 that those odds were
not backfilled. Fetch a current advisory snapshot with:

```powershell
python scripts/fetch_2026_odds_snapshot.py
```

`market_blended_optimizer` therefore remains candidate-only. The documented
fallback is manual entry of sourced 2018/2022 closing 1X2 lines into
`data/raw/historical_match_odds.csv`; its existing schema and loader accept those
rows without changing the gate criteria.

Current 2026 odds are advisory live information only. They cannot validate
promotion because match outcomes are not known yet. API-populated
`data/raw/match_odds.csv`, `data/raw/futures_odds.csv`, and
`data/raw/odds_snapshots/` are local data artifacts and must not be committed
unless explicitly reviewed and requested.

Config knobs (environment variables or `.env`): `MARKET_BLEND_ENABLED`,
`MARKET_BLEND_WEIGHT` (default 0.5), `SCORE_PMF_METHOD` (`independent` or
`dixon_coles`), `DEVELOPMENT_MODE`.
