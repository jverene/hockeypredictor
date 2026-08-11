# Hockey Career Trajectory Predictor

This system uses machine learning to predict hockey player performance.
It predicts the Points Per Game (PPG) of NHL skaters for the next season.
It uses only historical box-score statistics.
It has a second model for rookie players.
The rookie model uses NHL Equivalency (NHLe) factors.
NHLe factors translate statistics from other leagues into NHL values.

Refer to `PRD.md` for the full requirements.
Refer to `WRITEUP.md` for the results.

## Quick start

Do these steps to install and run the system:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Get the NHL data. The system caches the data in `data/cache/`:

```bash
python -m src.backtest --quantiles
```

This command also runs the full backtest.
The backtest covers the seasons 1990-91 to 2025-26.
It uses a rolling 5-year training window.

Run these commands for more analysis:

```bash
python -m src.tails             # Evaluate breakout and slump predictions
python -m src.compare_windows   # Compare training window strategies
pytest tests -q                 # Run the 14 unit tests
streamlit run app.py            # Start the demo application
```

## File layout

- `src/data.py` — Gets data from the NHL API. Caches the data on disk.
- `src/features.py` — Makes the model features. Includes the NHLe factors.
- `src/model.py` — Trains the XGBoost models. Supports quantile models.
- `src/backtest.py` — Runs the walk-forward backtest. Prevents data leakage.
- `src/metrics.py` — Calculates the evaluation metrics.
- `src/tails.py` — Evaluates breakout and slump predictions.
- `src/compare_windows.py` — Compares training window strategies.
- `app.py` — Streamlit demo. Has player search and career charts.
- `notebooks/eval.ipynb` — Evaluation notebook. Contains the full analysis.
- `WRITEUP.md` — Results and findings.

## Saved predictions

Every prediction is saved in `data/output/` as CSV:

- `veteran_predictions.csv` — 25,078 rows. One row per predicted player per
  season (1990-91 to 2025-26). Has the mean prediction, the q10/q50/q90
  quantile predictions, the actual result, and the evaluation flag.
- `rookie_predictions.csv` — 3,072 rows. One row per rookie per draft class
  (1990 to 2024). Has the prediction, the actual rookie PPG, and the draft
  position.
- `veteran_season_metrics.csv`, `rookie_class_metrics.csv` — Metrics per
  season and per draft class.
- `tail_predictions.csv`, `tail_metrics_by_season.csv` — Breakout and slump
  evaluation data.
- `window_comparison.csv` — The training-window A/B test results.
