"""Streamlit demo: player search, career arcs, backtest explorer (PRD §10).

Run:  streamlit run app.py
Requires the backtest outputs under data/output/ (see `python -m src.backtest`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.constants import SHORTENED_SEASONS, season_label

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "output"

st.set_page_config(page_title="Hockey Career Trajectory Predictor", layout="wide")


@st.cache_data
def load_outputs() -> dict[str, pd.DataFrame]:
    out = {}
    for name in [
        "player_seasons",
        "veteran_features",
        "veteran_predictions",
        "rookie_features",
        "rookie_predictions",
    ]:
        path = OUTPUT_DIR / f"{name}.parquet"
        if path.exists():
            out[name] = pd.read_parquet(path)
    for name in ["veteran_season_metrics", "rookie_class_metrics"]:
        path = OUTPUT_DIR / f"{name}.csv"
        if path.exists():
            out[name] = pd.read_csv(path)
    return out


data = load_outputs()
if "player_seasons" not in data:
    st.error("No backtest outputs found. Run `python -m src.backtest` first.")
    st.stop()

st.title("Hockey Career Trajectory Predictor")
st.caption(
    "Rolling-window XGBoost backtest, 1990-91 → 2023-24. "
    "Predictions are made with data available before each season only."
)

tab_player, tab_backtest, tab_rookies = st.tabs(["Player career arcs", "Backtest accuracy", "Rookie steals & busts"])

# ---------------------------------------------------------------------------
# Tab 1: player search + career arc
# ---------------------------------------------------------------------------
with tab_player:
    seasons_df = data["player_seasons"].copy()
    seasons_df["ppg"] = seasons_df["points"] / seasons_df["gp"]
    seasons_df["season"] = seasons_df["seasonId"].map(season_label)

    query = st.text_input("Search a player", placeholder="e.g. Crosby, Jagr, Makar…")
    matches = seasons_df
    if query:
        matches = seasons_df[seasons_df["skaterFullName"].str.contains(query, case=False, na=False)]
    names = sorted(matches["skaterFullName"].unique())
    if not names:
        st.info("No matching players.")
        st.stop()
    name = st.selectbox("Player", names)

    player = seasons_df[seasons_df["skaterFullName"] == name].sort_values("seasonId")
    preds = data.get("veteran_predictions", pd.DataFrame())
    player_preds = pd.DataFrame()
    if not preds.empty:
        player_preds = preds[preds["skaterFullName"] == name].copy()
        player_preds["season"] = player_preds["pred_season"].map(season_label)

    # Era adjustment: scale each season's PPG to 2023-24 scoring environment.
    ref = seasons_df.loc[seasons_df["seasonId"] == seasons_df["seasonId"].max(), "era_adj_factor"].iloc[0]
    player["ppg_era_adj"] = player["ppg"] * ref / player["era_adj_factor"]

    era_adjust = st.checkbox("Era-adjust to modern scoring environment", value=False)
    y_col = "ppg_era_adj" if era_adjust else "ppg"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=player["season"], y=player[y_col], mode="lines+markers", name="Actual PPG")
    )
    if not player_preds.empty:
        fig.add_trace(
            go.Scatter(
                x=player_preds["season"],
                y=player_preds["pred_ppg"],
                mode="lines+markers",
                name="Model prediction",
                line=dict(dash="dash"),
            )
        )
    for sid, note in SHORTENED_SEASONS.items():
        label = season_label(sid)
        if label in set(player["season"]):
            fig.add_vline(x=label, line_dash="dot", opacity=0.4, annotation_text=note)
    fig.update_layout(yaxis_title="Points per game", xaxis_title="Season", height=480)
    st.plotly_chart(fig, use_container_width=True)

    cols = ["season", "gp", "goals", "assists", "points", "ppg"]
    st.dataframe(player[cols].rename(columns=str.upper), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Tab 2: backtest accuracy over time
# ---------------------------------------------------------------------------
with tab_backtest:
    metrics = data.get("veteran_season_metrics")
    if metrics is None:
        st.info("No veteran metrics found.")
    else:
        m = metrics.dropna(subset=["mae"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean MAE (model)", f"{m['mae'].mean():.4f}")
        c2.metric("Mean MAE (last-season baseline)", f"{m['mae_baseline'].mean():.4f}")
        c3.metric("Mean directional accuracy", f"{m['directional_accuracy'].mean():.1%}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m["season"], y=m["mae"], mode="lines+markers", name="Model"))
        fig.add_trace(
            go.Scatter(x=m["season"], y=m["mae_baseline"], mode="lines+markers", name="Baseline (last season PPG)")
        )
        for _, r in m[m["shortened"].fillna("") != ""].iterrows():
            fig.add_vline(x=r["season"], line_dash="dot", opacity=0.4, annotation_text=r["shortened"])
        fig.update_layout(yaxis_title="MAE (PPG)", xaxis_title="Season", height=420)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(m, x="season", y="r2", markers=True, title="R² per season")
        st.plotly_chart(fig2, use_container_width=True)

        if "top_features" in m.columns:
            st.subheader("Top features (most recent season)")
            st.write(m.iloc[-1]["top_features"])

# ---------------------------------------------------------------------------
# Tab 3: rookie steals & busts
# ---------------------------------------------------------------------------
with tab_rookies:
    rook_preds = data.get("rookie_predictions")
    class_metrics = data.get("rookie_class_metrics")
    if rook_preds is None or class_metrics is None:
        st.info("No rookie backtest outputs found.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Mean rookie MAE", f"{class_metrics['mae'].mean():.4f}")
        c2.metric("Mean Spearman ρ", f"{class_metrics['spearman'].mean():.3f}")

        rp = rook_preds.copy()
        rp["steal_score"] = rp["pred_ppg"] - (0.55 - 0.0015 * rp["draft_ovr"])  # vs draft-slot expectation
        rp["season"] = rp["rookie_season"].map(lambda s: season_label(int(s)))

        fig = px.scatter(
            rp,
            x="pred_ppg",
            y="ppg_rookie",
            color="draft_ovr",
            hover_data=["skaterFullName", "season", "draft_ovr"],
            color_continuous_scale="RdYlGn_r",
            title="Predicted vs actual rookie PPG (color = draft position)",
            labels={"pred_ppg": "Predicted rookie PPG", "ppg_rookie": "Actual rookie PPG"},
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=rp["pred_ppg"].max(), y1=rp["pred_ppg"].max(),
                      line=dict(dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top 10 undervalued (model steals)")
        steals = rp.sort_values("steal_score", ascending=False).head(10)
        st.dataframe(
            steals[["skaterFullName", "season", "draft_ovr", "pred_ppg", "ppg_rookie"]],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Biggest busts (top-30 pick, lowest actual PPG)")
        busts = rp[rp["draft_ovr"] <= 30].sort_values("ppg_rookie").head(10)
        st.dataframe(
            busts[["skaterFullName", "season", "draft_ovr", "pred_ppg", "ppg_rookie"]],
            use_container_width=True,
            hide_index=True,
        )
