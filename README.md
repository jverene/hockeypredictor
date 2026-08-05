# Hockey Career Trajectory Predictor

A rolling-window ML system that predicts next-season Points Per Game (PPG) for NHL
skaters using only historical box-score stats, with a dedicated rookie-prediction
layer using NHL Equivalency (NHLe) to translate production across junior, college,
AHL, KHL, and European leagues into a common currency.

See `PRD.md` for the full product requirements.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Pull NHL data (cached under data/cache/)
python -m src.data --refresh

# Run the veteran + rookie backtests (1990-present, rolling 5-year window)
python -m src.backtest

# Launch the demo app
streamlit run app.py
```

## Layout

- `src/data.py` — NHL API ingestion + disk caching
- `src/features.py` — veteran & rookie feature engineering (incl. NHLe factors)
- `src/model.py` — XGBoost training wrappers (veteran / rookie modes)
- `src/backtest.py` — rolling walk-forward backtest engine with leakage guards
- `src/metrics.py` — evaluation metrics (MAE, RMSE, R², Spearman ρ, directional accuracy)
- `app.py` — Streamlit demo: player search + career arc visualizations
- `notebooks/eval.ipynb` — model evaluation and analysis
- `WRITEUP.md` — project writeup
