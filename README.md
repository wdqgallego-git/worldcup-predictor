# worldcup-predictor

Transparent heuristic model for World Cup match and award predictions.

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
- No partial points for awards.
- Coworkers do not see each other's picks before the deadline.
- Main strategy: maximize expected points.
- Secondary strategy: make a strategic pick only if expected value loss is small.

## Modeling Philosophy

Transparent heuristic model + limited calibration + historical backtesting + expected-points strategy.
