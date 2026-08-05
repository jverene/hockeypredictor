# PRD: Hockey Career Trajectory Predictor

## 1. Overview

A rolling-window ML system that predicts next-season Points Per Game (PPG) for NHL skaters using only historical box-score stats. The model backtests from 1990–present with strict temporal rigor (no data leakage) and includes a dedicated rookie-prediction layer using NHL Equivalency (NHLe) to translate production across junior, college, AHL, KHL, and European leagues into a common currency.

**Why this punches above its weight:** It uses domain-native hockey analytics (NHLe, aging curves) rather than generic ML. The backtest proves the model would have identified draft steals and busts in real time.

---

## 2. Goals

| Goal | Priority | Success Metric |
|------|----------|----------------|
| Predict next-season PPG for NHL skaters with ≥1 prior NHL season | P0 | MAE < 0.08 PPG league-wide |
| Predict rookie-season PPG using only pre-NHL data | P0 | Spearman ρ > 0.55 between predicted and actual rookie PPG |
| Backtest 1990–present with zero data leakage | P0 | Rolling 5-year training window, year-by-year walk-forward |
| Identify model "steals" vs actual draft position | P1 | Publishable top-10 undervalued players list |
| Streamlit demo with career arc visualizations | P1 | Live app with player search and era-adjusted trajectories |

---

## 3. Data Sources

| Source | Coverage | Access | Notes |
|--------|----------|--------|-------|
| NHL API (`api-web.nhle.com`) | 1918–present | Free, no auth | Season totals, skater stats, draft data |
| `nhl-api-py` (PyPI) | Same | `pip install nhl-api-py` | Python wrapper, handles pagination & caching |
| EliteProspects / `TopDownHockey_Scraper` | CHL, AHL, KHL, SHL, Liiga, NCAA, USHL | GitHub scraper or API application | Junior/minor league stats for rookie layer |
| Kaggle: "Hockey stats 30 leagues" | 38 seasons | Free download | Fallback / validation for junior data |

---

## 4. Feature Engineering

### 4.1 Veteran Prediction Features (≥1 prior NHL season)

| Feature | Type | Rationale |
|---------|------|-----------|
| `age` | Continuous (float) | Biological age at season start |
| `age_sq` | Continuous | Age squared — enables quadratic peak/decline curve |
| `age_vs_peak` | Continuous | `abs(age - 26)` — distance from theoretical peak |
| `years_since_debut` | Int | NHL seasons played (experience, separate from biological age) |
| `position` | Categorical (F/D) | Defensemen peak later and score less |
| `ppg_last1` | Continuous | Most recent season PPG |
| `ppg_last2` | Continuous | 2-year lag PPG |
| `ppg_last3` | Continuous | 3-year lag PPG |
| `ppg_rolling_3yr` | Continuous | 3-year average (smooths injury noise) |
| `gpg_last1` | Continuous | Goal production (separate from assists) |
| `apg_last1` | Continuous | Assist production |
| `gp_last1` | Int | Games played — durability/health proxy |
| `career_ppg` | Continuous | Baseline talent level across entire career |
| `career_gp` | Int | Total NHL games played |
| `team_pts_pct` | Continuous | Team points percentage that season (quality of teammates/system) |
| `era_adj_factor` | Continuous | League-average scoring rate that season (scoring was higher in the 80s) |
| `age_x_ppg_last1` | Continuous | Interaction: `age × ppg_last1` — same PPG means different things at 22 vs 32 |

**Target:** `ppg_next` — Points / Games in the following season

**Evaluation filter:** Only evaluate on players with ≥15 GP in the target season. This naturally excludes major injuries without requiring historical injury logs.

### 4.2 Rookie Prediction Features (pre-NHL → first NHL season)

**NHLe (NHL Equivalency) Factors:**

| League | NHLe Factor | Meaning |
|--------|-------------|---------|
| NHL | 1.00 | Baseline |
| KHL | 0.55 | Strong European league |
| SHL | 0.58 | Swedish Hockey League |
| Liiga | 0.54 | Finnish league |
| AHL | 0.45 | Primary minor league |
| NLA | 0.43 | Swiss league |
| DEL | 0.39 | German league |
| NCAA | 0.32 | US college hockey |
| OHL / WHL / QMJHL | 0.28 | Canadian major junior |
| USHL | 0.22 | Tier-1 US junior |

**Pre-NHL Season Definition:** The most recent season before the player's first NHL season with ≥15 GP. If they played in multiple leagues that season, aggregate using NHLe weighting:

```
EQ_PPG = Σ(league_gp × league_ppg × nhle_factor) / Σ(league_gp)
```

**Rookie Feature Vector:**

| Feature | Type | Source |
|---------|------|--------|
| `eq_ppg` | float | NHLe-weighted pre-NHL PPG |
| `eq_gpg` | float | NHLe-weighted pre-NHL GPG |
| `eq_apg` | float | NHLe-weighted pre-NHL APG |
| `pre_gp` | int | Total GP in pre-NHL season |
| `top_league_level` | ordinal | Highest league played: 1=USHL/CHL, 2=NCAA, 3=AHL, 4=KHL/SHL, 5=NHL cup-of-coffee |
| `age` | float | Age at start of rookie NHL season |
| `age_sq` | float | Age squared |
| `age_vs_league_avg` | float | Age minus average age of their pre-NHL league |
| `draft_ovr` | int | Overall draft position (undrafted = 300 or separate flag) |
| `draft_year` | int | Year drafted |
| `years_post_draft` | int | 0 = D+0, 1 = D+1, 2 = D+2, etc. |
| `position` | cat | F/D |
| `height` | int | Optional, from EliteProspects |
| `weight` | int | Optional, from EliteProspects |

**Target:** `ppg_rookie` — PPG in first NHL season with ≥15 GP

---

## 5. Model Architecture

### 5.1 Algorithm

| Spec | Choice | Why |
|------|--------|-----|
| Algorithm | **XGBoost** or **LightGBM** | Tabular data, fast training, interpretable feature importance, beats neural nets on small structured datasets |
| Hyperparameters | Default + early stopping (`n_estimators=500`, `early_stopping_rounds=20`) | Dataset is small; overfitting is the primary risk |
| Validation | Time-series split — no random shuffle | Prevents temporal data leakage |
| Baseline | "Last season's PPG" (veteran) / "Draft position expectation" (rookie) | If we can't beat this, the model is useless |

### 5.2 Two-Mode System

| Mode | Training Data | Feature Set | Use Case |
|------|---------------|-------------|----------|
| **Veteran** | NHL-only, rolling 5-year window | 4.1 features | Players with ≥1 prior NHL season |
| **Rookie** | All players' pre-NHL → rookie season pairs | 4.2 features | First-year NHL players |

Both modes share the same XGBoost architecture but are trained separately for cleaner interpretation. A `is_rookie` flag in a unified model is acceptable but harder to diagnose.

---

## 6. Training & Backtest Protocol

### 6.1 Veteran Backtest

```
For each season Y from 1990–91 to 2023–24:
    Training data: All player-seasons from Y-5 to Y-1
    Target: PPG in season Y
    Filter: Players with ≥15 GP in season Y
    Train XGBoost on training data
    Predict on all players who played in Y-1
    Record predictions vs actuals
```

### 6.2 Rookie Backtest

```
For each draft class from 1990 to 2020:
    Training data: All pre-NHL → rookie pairs from earlier draft classes
    Target: Rookie PPG (first NHL season ≥15 GP)
    Train XGBoost on training data
    Predict on current draft class rookies
    Record predictions vs actuals
```

### 6.3 Data Leakage Guards

- **Never** use future-season data in training
- Rolling 5-year window strictly enforced
- Players with <15 GP in target season excluded from evaluation (not from training)
- Lockout seasons handled explicitly:
  - **2004–05:** No NHL season — skip entirely
  - **2012–13:** 48-game season — flag as shortened
  - **2019–20:** COVID cutoff — flag as shortened
  - **2020–21:** 56-game season — flag as shortened

---

## 7. Evaluation

### 7.1 Veteran Metrics

| Metric | Purpose |
|--------|---------|
| **MAE** | Primary — mean absolute error on PPG |
| **RMSE** | Secondary — penalizes large misses |
| **R² per season** | Tracks consistency over time |
| **Directional accuracy** | % of players correctly predicted up/down vs last season |

### 7.2 Rookie Metrics

| Metric | Purpose |
|--------|---------|
| **MAE on rookie PPG** | Primary (will be higher than veteran MAE — this is expected) |
| **Spearman ρ** | Rank correlation between predicted and actual rookie PPG |
| **Steal identification** | Top-N players where `predicted_ppg >> draft_position_expectation` |
| **Bust identification** | High draft picks with low predicted AND low actual PPG |

### 7.3 Backtest Outputs

For each season 1990–2024:
- Predicted vs actual PPG for every qualifying skater
- Feature importance (SHAP values preferred)
- Biggest over-performers and under-performers
- Era-adjusted accuracy trends

---

## 8. Hockey Aging Curve (Domain Knowledge)

The model learns the following curve implicitly through `age` and `age_sq`:

| Age Range | Typical Trajectory |
|-----------|-------------------|
| 18–20 | Steep improvement (especially defensemen) |
| 21–24 | Solid year-over-year gains |
| 25–27 | **Peak production** |
| 28–30 | Plateau or slight decline |
| 31+ | Accelerating decline (steeper for forwards than defensemen) |

**Key insight:** The `age_sq` feature enables the model to learn that 26 is better than both 20 and 34 without hardcoding a peak. The `age × ppg_last1` interaction lets the model understand that 0.80 PPG at age 22 signals future stardom, while 0.80 PPG at age 32 signals decline.

---

## 9. Compute Estimates

| Task | Data Size | Est. Time | Hardware Required |
|------|-----------|-----------|-------------------|
| NHL API pull (all seasons) | ~2 MB raw | 5–10 min | Any CPU |
| Feature engineering | ~40K rows × 30 features | <1 min | Any CPU |
| XGBoost train (one year) | ~5K rows × 30 features | <2 sec | Any CPU |
| **Full veteran backtest (35 years)** | 35 models | **<2 min total** | Any CPU |
| Rookie model train | ~5K players | <1 sec | Any CPU |
| Hyperparameter tuning (optuna) | Same | <5 min | Any CPU |
| Streamlit app | N/A | Instant | Any CPU |

**Verdict: Zero rented compute needed.** This runs on a 2015 MacBook Air. The entire dataset is tiny by ML standards. Free Google Colab is more than sufficient for experimentation.

---

## 10. Deliverables

| Item | Format | Timeline |
|------|--------|----------|
| Data ingestion pipeline (NHL API + EP scraper) | Python module (`src/data.py`) | Day 1–2 |
| Feature engineering (veteran + rookie) | Python module (`src/features.py`) | Day 2–3 |
| Backtest engine (rolling window) | Python script (`src/backtest.py`) | Day 3–4 |
| Model training + evaluation | Jupyter notebook (`notebooks/eval.ipynb`) | Day 4–5 |
| Streamlit app (player search + career arcs) | `streamlit run app.py` | Day 5–7 |
| Project writeup / blog post | Markdown (`WRITEUP.md`) | Day 7 |

---

## 11. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Lockout/shortened seasons skew PPG | High | Medium | Flag shortened seasons; skip 2004–05; normalize per-82 where appropriate |
| Rookies with no structured pre-NHL data | Medium | Medium | Accept as limitation; note in writeup; these players are excluded from rookie evaluation |
| Name matching NHL ↔ EliteProspects | Medium | High | Use name + DOB; manual spot-check edge cases; cache aggressively |
| EliteProspects rate limits / blocks | Low | Medium | Cache all API responses; fall back to Kaggle dataset |
| Position changes (F ↔ D) | Low | Low | Rare; use "primary position" rule (mode across career) |
| European leagues with sparse coverage pre-2000 | Medium | Low | NHLe factors are static and literature-backed; missing data handled gracefully |

---

## 12. Narrative Hooks (For Portfolio / Blog)

1. **"Can a model predict hockey careers better than scouts?"** — The core question that sells the project.
2. **"Every draft steal our model identified"** — Scatter plot of predicted vs actual rookie PPG, colored by draft position.
3. **"The aging curve is real"** — Feature importance showing age and age_sq in top 3; SHAP dependence plot.
4. **"What the 2004 lockout did to every model"** — Anomaly detection in backtest accuracy.
5. **"From the AHL to the NHL: one number"** — NHLe explanation with real player examples.

---

## 13. Open Questions

1. Should the Streamlit app be **publicly hosted** (Streamlit Cloud free tier) or local-only?
2. Do you want to predict **all positions** or split into Forward and Defense models?
3. Should we include **playoff performance** as a feature, or keep it regular-season only?
4. Do you want to model **goal-scoring** and **assist production** separately, or only total points?

---

## 14. Tech Stack

```
Python 3.10+
nhl-api-py          # NHL API wrapper
pandas, numpy       # Data manipulation
xgboost / lightgbm  # Gradient boosting
scikit-learn        # Metrics, preprocessing
shap                # Model interpretability
streamlit           # Demo app
plotly / matplotlib # Visualization
requests + bs4      # EliteProspects scraping (if needed)
```

---

*Last updated: 2026-08-05*
*Status: Draft — ready for implementation*
