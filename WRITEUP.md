# Hockey Career Trajectory Predictor — Results

Can a model predict hockey careers? Yes. It does better than the simple
baseline. The simple baseline is "the same as last season". The model is
better in all 35 backtested seasons. There is no data leakage.

Can it predict rookies better than NHL scouts? Almost. The remaining
difference is the information that scouts have and box scores do not have.

---

## Results summary

| Goal (PRD §2) | Target | Result | Verdict |
|---|---|---|---|
| Veteran next-season PPG | MAE < 0.08 | **MAE 0.129** (baseline 0.154) | Target not met. Model is 16.4% better than the baseline in all 35 seasons |
| Rookie-season PPG | Spearman ρ > 0.55 | **ρ = 0.503** (0.354 without NHLe data) | Target almost met. ρ > 0.55 in 13 of 35 draft classes |
| Leak-free backtest 1990–2026 | rolling 5-year walk-forward | 35 veteran seasons and 35 draft classes | Met |
| Steals list | top-10 undervalued players | produced | Met |
| Streamlit demo | live application | `streamlit run app.py` | Met |

Directional accuracy is **68.3%**. This is the percentage of players where the
model predicted the correct direction of change. Mean R² per season is
**0.632**.

### The 0.08 MAE target was not realistic

PPG has much random variation. A 0.5 PPG player who plays 75 games has a
sampling error of approximately 0.05. This is before injuries, line changes,
and coaching changes. The baseline has an MAE of 0.154. The model has an MAE
of 0.129. This is an improvement of 16.4%. The improvement occurs in every
season from 1990-91 to 2025-26. This includes the high-scoring 1990s, the
dead-puck era, the post-lockout era, and the modern era. This document shows
the honest number. We did not tune the model to the target.

## Method summary

The system uses XGBoost regressors. It trains one model per backtest season.
For target season Y, the model trains on player-season pairs. The target of
each pair is strictly before season Y. The training window is 5 seasons. The
model then predicts season Y for each skater who played in season Y-1.

The features are: aging-curve terms (`age`, `age²`, `|age−26|`, `age×PPG`),
production lags (1, 2, and 3 years), a 3-year rolling average, career
baselines, durability (games played), team quality, and the era scoring rate.

The evaluation excludes players with less than 15 games in the target season.
The training includes these players. The cancelled 2004-05 season is skipped.
Shortened seasons have a flag. All data comes from the free NHL API. The
system caches the data on disk.

## The aging curve is real

In all 35 season-models, `age`, `age_sq`, and `age × ppg_last1` are in the
top 5 features in almost every year. Only the production lags are more
important. The SHAP dependence plot (notebook §3) shows an inverted U-shape.
The effect is positive until the early 20s. The peak is at age 25 to 27.
The decline starts after age 30. The model learned the PRD §8 table from
data alone.

## Training-window A/B test

Question: must the model forget old seasons? An alternative is one model
with all history. A third option adds recency weights. We tested all three
options in the same walk-forward procedure:

| Option | MAE | R² | Directional accuracy |
|---|---|---|---|
| expanding (all history) | **0.1279** | 0.635 | 0.684 |
| expanding + 0.85/year decay | 0.1283 | 0.634 | 0.683 |
| rolling 5-year (PRD default) | 0.1285 | 0.632 | 0.683 |

The difference is very small. The `era_adj_factor` feature lets the model
adjust for era differences internally. The details are interesting. The
rolling window is better in shock years. Its largest win is the COVID season
2020-21 (−0.007 MAE). The expanding window is better after rule changes. Its
largest win is 2006-07 (+0.008). We keep the rolling window as the default.
Run `python -m src.compare_windows` to repeat this test.

## Availability sensitivity

Question: is the MAE better if we exclude injured players? We tested this
on all 18,653 evaluated player-seasons. Note: this is hindsight analysis.
A real system cannot know future injuries.

| Filter | n | MAE |
|---|---|---|
| ≥15 games in target season (current) | 18,653 | 0.1275 |
| ≥41 games | 15,829 | 0.1277 |
| ≥70 games | 8,861 | **0.1345 (worse)** |
| exclude large games-played decreases | 16,604 | 0.1263 |

There are two findings. First, the injury effect is real but small. Players
with a large decrease in games have an MAE worse by 0.010. The model predicts
too much for these players by 0.058 PPG on average. This is the signature of
injuries. Second, the exclusion of these players changes the total MAE by
only 0.001. Stricter filters make the result worse. They remove the
easy-to-predict depth players. Only the volatile stars remain. The 15-game
filter already captures almost all of the possible gain.

## Tail predictions: breakouts and slumps

The mean model shrinks extreme values. It never predicts more than
approximately 1.33 PPG. This is why it predicts too low for McDavid every
year. For the tails, the system trains quantile models (q10, q50, q90) each
season. Use `python -m src.backtest --quantiles`. The **upside spread**
(q90−q50) is the breakout-risk score. The **downside spread** (q50−q10) is
the slump-risk score. A breakout is an increase of ≥ +0.30 PPG from the last
season. A slump is a decrease of ≥ 0.30 PPG.

Results for all 18,653 evaluated player-seasons:

| Metric | Result | Meaning |
|---|---|---|
| 80% interval coverage | **0.797** | the calibration is very good |
| Upside score → breakout AUC | **0.631** | real signal (0.5 = random) |
| Downside score → slump AUC | **0.691** | slumps are easier to predict than breakouts |
| Precision@50 breakouts | 0.126 vs 0.068 base rate | 1.9× better than random |
| Precision@50 slumps | 0.147 vs 0.065 base rate | 2.3× better than random |

The model sees the tails. Two examples of correct upside calls: Lemieux
1992-93 (median 1.36, but **q90 was 2.06**) and Keith Tkachuk 1993-94
(median 0.39, q90 0.90, actual 0.96). McDavid is the exception. His actual
PPG is above his personal q90 in five of seven seasons. He is a
greater-than-90th-percentile player every year.

## Do the best players always beat expectations? Yes.

Question: is there a correlation between player quality and prediction
deviation? Deviation is actual minus predicted PPG. The answer is yes. The
correlation is large and significant.

For 1,441 players with 6 or more backtested seasons: Spearman(career rating,
mean deviation) = **0.589** (p ≈ 3×10⁻¹³⁵). The better the player, the more
the model underpredicts him.

The q90 data shows the structure. The table shows how often each tier beats
its own 90th-percentile prediction:

| Established rating (last-season PPG) | Beats own q90 |
|---|---|
| <0.25 | 9.8% (correctly calibrated) |
| 0.45–0.65 | 13.4% |
| 0.85–1.05 | 19.2% |
| 1.05–1.25 | 21.7% |
| **>1.25 (stars)** | **24.4%** |

Depth players are correctly calibrated. Stars beat their supposed ceiling
2.4 times too often. The career leaderboard of expectation-beaters (mean
deviation per season, q90-beat rate): **Connor McDavid +0.45 (78%)**,
**Nikita Kucherov +0.41 (89%)**, Leon Draisaitl +0.32 (70%), Nathan
MacKinnon +0.30 (64%), Mario Lemieux +0.29 (38%), Sidney Crosby +0.22 (53%),
Jaromir Jagr +0.19 (48%). Auston Matthews is the exception: +0.002. The
model prices him correctly.

The cause is symmetric loss. Mean-squared-error training treats an error of
0.4 for McDavid the same as an error of 0.4 for any other player. There is
one McDavid and thousands of other players. So the optimum moves toward the
average. Mean reversion is correct for 99% of players. Therefore the model
always prices greatness too low. The q90-beat rate is a free "greatness
index": the players who beat their ceiling every year are, by definition,
the great ones.

**The reverse also works: the model's most overrated players.** The same
machinery finds the players who chronically deliver less than predicted.
But the bias is asymmetric. Only 8% of high-rated players (≥0.8 PPG) have a
negative mean deviation. And stars land below their q10 only 8.0% of the
time (target: 10%). The pessimistic side of the model is correctly
calibrated even for stars. The miscalibration is only at the ceiling.

The chronic under-deliverers (6+ seasons): Pat Elynuik (−0.21 mean
deviation), Elias Pettersson (−0.14, below his q10 in 33% of seasons),
Patrik Laine (−0.14), Colby Armstrong (−0.14), Wojtek Wolski (−0.14). These
cases almost always have a cause that box scores do not show. Elynuik:
knee injuries. Pettersson: the model predicted the 100-point version after
the decline started. Laine and Wolski: one-dimensional scorers who lost
their role. The model is wrong about great players upward because they do
things that never happened before. It is wrong about declining players
downward because injuries and role loss are not in the data.

## Archetype predictability ranking

Each player-season has a position, an age band, and a production tier. The
production tier uses the PPG of the last season. This table ranks the
archetypes by MAE (cells with n ≥ 100):

| # | Archetype | n | MAE | Bias |
|---|---|---|---|---|
| 1 | D, 31+, depth (<0.25) | 739 | **0.061** | −0.00 |
| 2 | F, 31+, depth | 476 | 0.074 | −0.01 |
| 3 | D, 27-30, depth | 997 | 0.077 | +0.00 |
| … | (middle: middle-tier and older top-pair players) | | 0.09–0.17 | |
| 23 | F, ≤22, middle (0.25-0.55) | 425 | 0.181 | +0.05 |
| 24 | F, ≤22, top-6 (0.55-0.85) | 203 | 0.186 | +0.06 |
| 25 | F, 23-26, elite (>0.85) | 448 | 0.191 | +0.05 |
| 26 | F, ≤22, elite (>0.85) | 96 | **0.242** | +0.05 |

Three rules are visible:

1. **Defensemen are easier to predict than forwards.** This is true at every
   age and tier. Defense scoring depends on the role. Roles are stable.
   Forward scoring has more individual variance.
2. **Low production is easy. High production is hard.** The floor is zero.
   Depth players have almost no downside variance. The error increases with
   the mean.
3. **Young and good is the hardest.** Every cell with age ≤22 has a positive
   bias (+0.05 to +0.06). The model predicts too low for developing players.
   Mean reversion cannot see the development curve. The hardest cell is a
   player age ≤22 who is already elite (the McDavid/Matthews/Hughes cell).
   This cell is 4 times harder than an aging depth defenseman. The quantile
   layer is built for this cell.

## Correct predictions and incorrect predictions

**The most accurate predictions** are mid-tier veterans in stable roles:
Tyler Pitlick 2019-20 (0.317 predicted, 0.317 actual), Niklas Hjalmarsson
2012-13 (0.217 / 0.217), Darren Turcotte 1991-92 (0.747 / 0.746), Marian
Hossa 2012-13 (0.775 / 0.775).

**The largest misses downward** (predicted high, actual low): Curtis Brown
1996-97 (predicted 1.03 after a small-sample 2.0 PPG season, actual 0.25),
Mike Green 2011-12 (0.87 → 0.22, injuries), Jimmy Carson and Rob Brown
1990-91 (both lost the star-linemate effect), Steve Yzerman 1994-95
(lockout and knee injury at age 29). The failure modes are the human failure
modes: linemate effects, aging cliffs, and small samples.

**The largest misses upward** (actual much above prediction): Mark Recchi
1990-91 (0.29 → 1.45), Adam Oates 1990-91 (0.79 → 1.89), Mario Lemieux
1992-93 (predicted 1.73, actual **2.67** after his cancer return), Tage
Thompson 2022-23 (0.29 → 1.21), Jonathan Cheechoo 2005-06 (0.29 → 1.13),
Erik Karlsson 2022-23 (0.45 → 1.23). Real breakouts are not in the
historical data. This list shows the limit of the box-score approach.

## Draft steals and busts

The rookie layer uses NHLe-translated pre-NHL production. The data comes
from the NHL API player biographies. There are 53,561 junior, college, and
European stints in 12 leagues. EliteProspects was not necessary. Refer to
Limitations for the details.

**The marquee test cases:** the model used only data available before each
draft class. It predicted **Connor Bedard at 0.850 PPG (actual: 0.897)**,
**Auston Matthews at 0.794 (actual: 0.841)**, **Connor McDavid at 0.779
(actual: 1.067)**, and **Macklin Celebrini at 0.719 (actual: 0.900)**.
Without the NHLe features, Bedard and Celebrini both get 0.67. That is the
historical average for a #1-overall pick. Draft position is all the model
can see without NHLe. The 143-point WHL season is the difference between
Bedard and Alexandre Daigle. NHLe is how the model sees this difference.

**Steals that delivered** (predicted much above the draft-position
expectation, and the player delivered): **Miroslav Satan** (111th pick,
predicted 0.68, actual 0.57), **Andreas Dackell** (136th), and **Kai
Nurminen** (193rd).

**Busts the model saw** (top-10 pick, low prediction, low actual): **Nino
Niederreiter** (5th, predicted 0.34, actual 0.02), **Alek Stojanov** (7th,
0.28 → 0.03), **Alexandre Picard** (8th, 0.34 → 0.00), **Griffin Reinhart**
(4th, 0.22 → 0.03).

The rookie model is also the honesty checkpoint of the project. The mean ρ
of 0.503 does not meet the PRD target of 0.55. But 13 of 35 classes meet it.
The difference between the model and scouts is the information that box
scores do not have: skating, hockey sense, medical results, and interviews.
Draft position is a proxy for scout consensus. It stays one of the top
features. This shows that the scout information is real.

## Limitations

1. **The pre-NHL data comes from NHL biographies.** It is not a full league
   scrape. The `seasonTotals` data covers players who played in the NHL.
   It does not cover other players. Two consequences: leagues without a
   literature-backed NHLe factor are excluded (ECHL, second-tier European
   leagues, tournaments). And `age_vs_league_avg` compares against future
   NHL players in that league-season. It is not the true league average.
   Historical league names are mapped with `LEAGUE_ALIASES`. For example,
   "Sweden" maps to SHL. The Czech Extraliga factor (0.40) is an addition
   to the PRD table. A CSV file at `data/prenhl_stats.csv` overrides the
   biography data.
2. **Box scores only.** No ice time. No linemates. No injuries. No
   contracts. `gp_last1` is the complete health model.
3. **Playoffs are excluded.** The model uses the regular season only.
4. Multi-team seasons split the games evenly across teams. The API does not
   give per-team games for these players. This affects the team-quality
   feature.

## Reproduce

```bash
pip install -r requirements.txt
python -m src.backtest --quantiles  # Get data and run both backtests
python -m src.tails                 # Evaluate breakouts and slumps
python -m src.compare_windows       # Training-window A/B test
jupyter nbconvert --execute notebooks/eval.ipynb
streamlit run app.py
pytest tests -q                     # 14 tests, includes leakage tests
```

*Stack: Python 3.12, XGBoost, scikit-learn, pandas, SHAP, Streamlit, Plotly.
Compute: one laptop. Approximately 30 minutes for the one-time data download.
Approximately 5 minutes of model training.*
