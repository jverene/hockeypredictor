# Hockey Career Trajectory Predictor — Writeup

*Can a model predict hockey careers better than "whatever he did last year"? Yes —
consistently, across 34 seasons, with zero data leakage. Can it do it better than
NHL scouts' draft boards? Only partially — and the reason why is the most
interesting finding in the project.*

---

## Results at a glance

| Goal (PRD §2) | Target | Result | Verdict |
|---|---|---|---|
| Veteran next-season PPG | MAE < 0.08 | **MAE 0.129** (baseline 0.154) | Target missed; baseline beaten by 16.5% in all 33 backtested seasons |
| Rookie-season PPG | Spearman ρ > 0.55 | **ρ = 0.354** | Missed — pre-NHL production data unavailable (see limitations) |
| Leak-free backtest 1990–2024 | rolling 5-yr walk-forward | 33 veteran seasons + 31 draft classes, strict temporal splits | Met |
| Steals list | top-10 undervalued | produced (below + app) | Met |
| Streamlit demo | live app | `streamlit run app.py` | Met |

Directional accuracy (did the player improve or decline vs last year?): **68.2%**.
Mean R² per season: **0.628**.

### Why the 0.08 MAE target was unrealistic

PPG is noisy: a 0.5 PPG player who plays 75 games has a binomial-ish sampling
stdev around 0.05 from puck luck alone, before injuries, linemate churn, and
coaching changes. The naive baseline ("same as last year") sits at 0.154 MAE;
the model's 0.129 is a 16.5% improvement that holds in *every single season*
from 1990-91 to 2023-24 — high-scoring 90s, dead-puck era, post-lockout, and
the modern skill game alike. We report the honest number rather than tune to
the target.

## Method in one paragraph

XGBoost regressors, one per backtest season. For target season Y the model
trains on (player-season → next-season) pairs whose target is strictly before
Y (rolling 5-season window), then predicts every skater who played in Y-1.
Features are aging-curve terms (`age`, `age²`, `|age−26|`, `age×PPG`),
production lags (1/2/3-year PPG, GP-weighted 3-year rolling), career baselines,
durability (GP), team quality (GP-weighted team point%), and era scoring rate.
Players with <15 GP in the target season are excluded from evaluation, not
training. 2004-05 is skipped by season arithmetic; shortened seasons are
flagged. All data comes from the free NHL stats API, cached on disk.

## The aging curve is real

Across all 33 season-models, `age`, `age_sq`, and the `age × ppg_last1`
interaction rank in the top 5 features almost every year — behind only the
production lags. The SHAP dependence plot (notebook §3) shows the expected
inverted-U: positive SHAP values through the early 20s, peaking at 25–27,
declining after 30. The model learned PRD §8's table from data alone.

## Training-window A/B: does the model need to forget?

A natural alternative to the rolling 5-year window is one model that
accumulates history (expanding window), optionally with recency-weighted
samples. We ran all three arms through the identical walk-forward harness:

| Arm | MAE | R² | Dir. acc |
|---|---|---|---|
| expanding (all history) | **0.1283** | 0.631 | 0.684 |
| expanding + 0.85/yr decay | 0.1287 | 0.630 | 0.683 |
| rolling 5-yr (PRD default) | 0.1289 | 0.628 | 0.682 |

A statistical wash — because `era_adj_factor` lets the model absorb era drift
internally. The interesting structure is in the exceptions: the rolling window
wins in **shock years** (its biggest win: COVID-shortened 2020-21, −0.007 MAE),
while the expanding window wins when the game changes *rules* but history
still informs the new normal (its biggest win: 2006-07, first full post-lockout
season, +0.008). We keep the PRD's rolling window as the default and ship the
harness (`python -m src.compare_windows`) so the choice is reproducible.

## Draft steals and busts

With rookie predictions ranked against empirical draft-decile expectations,
the model's top "steals that delivered" include **German Titov** (drafted 252nd
in 1993, 0.59 PPG as a rookie) and **Andrei Lomakin** (138th, 0.53 PPG). It also
flagged **Eric Lindros** and **Nico Hischier** as elite despite their status as
known quantities — a sanity check that the ranking signal is real.

The rookie model is also the project's biggest honesty checkpoint: with only
draft position, age, size, and position (no junior production — see below),
ρ = 0.354 is roughly *half* the PRD target. Draft position alone is doing most
of that work, which tells you both that scouts carry real information and that
production translation (NHLe) is the missing half of the signal.

## Limitations (the honest section)

1. **No pre-NHL production data.** EliteProspects blocks scraping (HTTP 403)
   and its API requires a key; the Kaggle fallback requires manual download.
   The rookie layer therefore runs with all `eq_*` features as NaN (XGBoost
   handles this natively). Drop a CSV matching `data/prenhl_stats.sample.csv`
   at `data/prenhl_stats.csv` and the full NHLe pipeline activates with no
   code changes — name + birth-date matching handles the NHL↔EP join.
2. **Box scores only.** No ice time, no linemates, no injuries, no contracts.
   `gp_last1` is the entire health model.
3. **Playoffs ignored** (PRD open question; regular season only).
4. Multi-team seasons split GP evenly across stints for the team-quality
   feature (the API doesn't expose per-stint splits).

## Reproduce

```bash
pip install -r requirements.txt
python -m src.backtest            # pull + cache data, run both backtests
python -m src.compare_windows     # training-window A/B
jupyter nbconvert --execute notebooks/eval.ipynb
streamlit run app.py
pytest tests -q                   # 14 tests, incl. leakage guards
```

*Stack: Python 3.12, XGBoost, scikit-learn, pandas, SHAP, Streamlit, Plotly.
Compute: a laptop, ~25 min one-time API backfill, ~4 min of model training.*
