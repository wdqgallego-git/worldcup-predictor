# worldcup-predictor

Google Colab-compatible World Cup 2026 prediction project.

The goal is a transparent, CPU-friendly heuristic model for match predictions, award probabilities, and expected-points strategy in a company prediction game.

## Predictions

- Match scores
- Match result: Team A win / draw / Team B win
- Goal difference
- Champion
- Runner-up
- Top scorer
- MVP / Golden Ball
- Golden Glove

## Company Game Scoring

| Category | Points |
| --- | ---: |
| Champion | 25 |
| Runner-up | 15 |
| Top scorer | 35 |
| MVP / Golden Ball | 10 |
| Golden Glove | 10 |

## Rules

- One pick per award.
- Champion and runner-up are independent picks.
- No partial award points.
- Coworkers do not see each other's picks before the deadline.
- Main objective: maximize expected points.
- Secondary objective: make a strategic pick only when it keeps at least 85-90% of safe expected value.

## Tournament Format

World Cup 2026 uses:

- 48 teams
- 12 groups of 4
- Top 2 teams from each group qualify
- 8 best third-placed teams qualify
- Round of 32 before Round of 16
- Finalists can play 8 matches

This project must not use the old 32-team / 8-group / Round-of-16 format.

## Modeling Philosophy

Transparent heuristic model + limited calibration + historical backtesting + expected-points strategy.

No deep learning, AutoML, heavy grid search, overfitted MVP fitting, or overfitted Golden Glove fitting.

## Simulation Defaults

- Development runs: 1,000 simulations
- Final runs: up to 20,000 simulations

## Project Structure

```text
worldcup-predictor/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 01_colab_quickstart.ipynb
├── outputs/
├── scripts/
│   ├── run_backtest.py
│   └── run_simulation.py
├── src/
│   └── worldcup_predictor/
│       ├── awards.py
│       ├── backtesting.py
│       ├── calibration.py
│       ├── config.py
│       ├── format_2026.py
│       ├── io.py
│       ├── match_model.py
│       ├── ratings.py
│       ├── schemas.py
│       ├── strategy.py
│       └── tournament.py
├── README.md
└── requirements.txt
```

## Colab Quick Start

Open `notebooks/01_colab_quickstart.ipynb` in Google Colab, clone this repository, install `requirements.txt`, and run the skeleton checks before implementing each module.
