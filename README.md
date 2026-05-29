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
6. Optimize picks for expected company-game points.
