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

## Availability sensitivity: would excluding injured players help?

A fair objection to the 0.129 MAE is that it includes players who got hurt.
We tested this two ways on all 17,465 evaluated player-seasons (hindsight
analysis only — you can't know in October who gets hurt in February):

| Filter | n | MAE |
|---|---|---|
| ≥15 GP in target season (current) | 17,465 | 0.1279 |
| ≥41 GP (half season) | 14,793 | 0.1281 |
| ≥70 GP | 8,187 | **0.1352 (worse)** |
| exclude GP collapses (<60% of prior season) | 15,529 | 0.1267 |

Two findings. First, the injury signal is real but small: players whose games
collapsed are predicted 0.010 MAE worse than everyone else, and the model
systematically *over*predicts them by 0.057 PPG — exactly the signature of
invisible injuries. Second, excluding them barely moves the headline number
(−0.001), and harsher GP thresholds actively hurt: they remove low-PPG depth
players, who are the easiest to predict, and leave only volatile stars. The
≥15 GP filter already captures almost all of the available gain.

## Tail predictions: breakouts and slumps

The mean model shrinks extremes (it never predicts above ~1.33 PPG, which is
why it underrates McDavid every year). To say something about the tails we
train q10/q50/q90 pinball-loss quantile models each season alongside the mean
model (`python -m src.backtest --quantiles`). The **upside spread** (q90−q50)
is a breakout-risk score; the **downside spread** (q50−q10) a slump-risk
score. A breakout is a ≥ +0.30 PPG jump vs last season; a slump ≤ −0.30.

Results over all 17,465 evaluated player-seasons:

| Metric | Result | Reading |
|---|---|---|
| 80% interval coverage | **0.797** | near-perfect calibration |
| Upside score → breakout AUC | **0.631** | real signal (0.5 = coin flip) |
| Downside score → slump AUC | **0.690** | slumps are more predictable than breakouts |
| Precision@50 breakouts | 0.128 vs 0.069 base | 1.9× enrichment |
| Precision@50 slumps | 0.149 vs 0.066 base | 2.3× enrichment |

The model genuinely sees the tails coming: among actual breakouts, the highest
upside calls include Lemieux 1992-93 (median 1.36 but **q90 of 2.06** — the
model left the door open for superhuman) and Keith Tkachuk 1993-94 (q50 0.39,
q90 0.90, actual 0.96). And McDavid is the exception that proves the rule: his
actual PPG exceeds even his personal q90 in five of seven seasons — he is, by
the model's own accounting, a >90th-percentile-outcome player every year.

## Archetype predictability ranking

Every evaluated player-season binned by position × age band × production tier
(last-season PPG), ranked by backtest MAE (cells with n ≥ 100):

| # | Archetype | n | MAE | Bias |
|---|---|---|---|---|
| 1 | D, 31+, depth (<0.25) | 688 | **0.062** | −0.00 |
| 2 | F, 31+, depth | 441 | 0.073 | −0.01 |
| 3 | D, 27-30, depth | 941 | 0.076 | +0.01 |
| 4 | D, 23-26, depth | 1,079 | 0.087 | −0.00 |
| 5 | F, 27-30, depth | 821 | 0.093 | +0.02 |
| … | (middle of the table: middle-tier and older top-pair players) | | 0.10–0.16 | |
| 22 | F, ≤22, middle (0.25-0.55) | 402 | 0.180 | +0.05 |
| 23 | F, ≤22, top-6 (0.55-0.85) | 191 | 0.188 | +0.06 |
| 24 | F, 23-26, elite (>0.85) | 424 | 0.193 | +0.05 |
| 25 | F, ≤22, elite (>0.85) | 91 | **0.240** | +0.07 |

Three clean laws emerge:

1. **Defensemen are easier than forwards at every age and tier.** D scoring is
   role-driven (power-play time, team system) and roles are sticky; F scoring
   has more individual variance.
2. **Low production is easy, high production is hard.** The floor is bounded
   (nobody scores below zero), so depth players have almost no downside
   variance. Error scales with the mean — a near-proportional relationship.
3. **Young is hard, and young + good is hardest.** Every ≤22 cell carries a
   *positive* bias (+0.05 to +0.07): the model systematically underpredicts
   developing players because mean reversion can't see the development curve
   arriving. The single hardest cell in hockey — a 20-year-old who is already
   elite (the McDavid/Matthews/Hughes cell) — is 4× harder to predict than an
   aging depth defenseman. This is precisely the cell the quantile layer
   (upside spread) was built for.

## Where the model is right, and where it can't be

**Most accurate predictions** — mid-tier veterans in stable roles; boring is
predictable: Tyler Pitlick 2019-20 (0.317 pred / 0.317 actual), Niklas
Hjalmarsson 2012-13 (0.217 / 0.217), Darren Turcotte 1991-92 (0.747 / 0.746),
Marian Hossa 2012-13 (0.775 / 0.775).

**Biggest busts (predicted high, delivered low):** Curtis Brown 1996-97
(predicted 1.03 after a small-sample 2.0 PPG tease, delivered 0.25), Mike
Green 2011-12 (0.87 → 0.22, injuries after back-to-back 70-point seasons),
Jimmy Carson and Rob Brown 1990-91 (both lost the Lemieux/Gretzky linemate
effect), Steve Yzerman 1994-95 (lockout + knee at 29). The failure modes are
the human ones: linemate effects, aging cliffs arriving a year early, and
small-sample flukes.

**Biggest risers (delivered far above prediction):** Mark Recchi 1990-91
(0.29 → 1.45, age-22 Cup-year breakout), Adam Oates 1990-91 (0.79 → 1.89),
Mario Lemieux 1992-93 (predicted a league-leading 1.73; he returned from
cancer and posted **2.67** — being wrong about Lemieux because he was
superhuman is a good error to have), Tage Thompson 2022-23 (0.29 → 1.21),
Jonathan Cheechoo 2005-06 (0.29 → 1.13, 56 goals, never repeated), Erik
Karlsson 2022-23 (0.45 → 1.23, the first 100-point defenseman in 30 years).
Genuine breakouts are by definition absent from historical box scores — this
list is the honest ceiling of the approach.

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
