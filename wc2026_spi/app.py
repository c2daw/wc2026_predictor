"""
World Cup 2026 SPI Predictor — Streamlit app
Dixon-Coles Poisson model with attack/defense strength per team.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from model import load_data, fit_model, predict_match, top_scorelines, simulate_group, WC2026_GROUPS

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WC 2026 SPI Predictor",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ World Cup 2026 — SPI Match Predictor")
st.caption(
    "Dixon-Coles Poisson model · separate attack & defense strength per team · "
    "time-decayed match history · data via [martj42/international_results](https://github.com/martj42/international_results)"
)

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model settings")
    min_date = st.selectbox(
        "Training data from",
        ["2010-01-01", "2014-01-01", "2016-01-01", "2018-01-01"],
        index=1,
        help="Earlier = more data but older matches. 2014 is a good balance."
    )
    neutral = st.checkbox("Neutral venue", value=True,
                          help="World Cup matches are at neutral venues (no home advantage).")
    st.divider()
    st.markdown("**About**")
    st.markdown(
        "Unlike Elo (which predicts 1-0 or 1-1), this model uses a Poisson "
        "distribution with separate attack & defense ratings, so it produces "
        "realistic scoreline distributions."
    )

# ── Fit model (cached) ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching data & fitting model…", ttl=3600)
def get_model(min_date):
    df = load_data(min_date=min_date)
    ratings, home_adv, rho, valid_teams, params = fit_model(df)
    return ratings, home_adv, rho, valid_teams, params

with st.spinner("Loading…"):
    ratings, home_adv, rho, valid_teams, params = get_model(min_date)

# ── Cached group simulations ─────────────────────────────────────────────────
@st.cache_data(show_spinner="Simulating group stages…", ttl=3600)
def get_group_tables(_valid_teams, _params, _home_adv, _rho):
    """Run Monte-Carlo for all 12 groups. Underscore args bypass cache hashing."""
    return {
        letter: simulate_group(
            teams, _valid_teams, _params, _home_adv, _rho, n_sims=20_000
        )
        for letter, teams in WC2026_GROUPS.items()
    }

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Match Predictor", "🏆 Group Tables", "📊 Team Rankings"])

with tab2:
    st.subheader("WC 2026 Group Stage — Expected Tables")
    st.caption(
        "20,000 Monte-Carlo simulations per group · all matches at neutral venues · "
        "TBD playoff teams use average (50 SPI) ratings."
    )

    group_tables = get_group_tables(valid_teams, params, home_adv, rho)

    cols_per_row = 3
    group_letters = list(WC2026_GROUPS.keys())
    for row_start in range(0, len(group_letters), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, letter in enumerate(group_letters[row_start:row_start + cols_per_row]):
            with cols[col_idx]:
                st.markdown(f"#### Group {letter}")
                df = group_tables[letter]

                # Colour the qualification columns
                styled = df.style.background_gradient(
                    subset=["P(1st)", "P(2nd)"], cmap="Greens", vmin=0, vmax=100
                ).background_gradient(
                    subset=["P(3rd)"], cmap="YlOrBr", vmin=0, vmax=50
                ).background_gradient(
                    subset=["xPts"], cmap="Blues", vmin=0, vmax=9
                ).format({
                    "xPts": "{:.1f}", "xGD": "{:+.1f}", "xGF": "{:.1f}",
                    "P(1st)": "{:.0f}%", "P(2nd)": "{:.0f}%",
                    "P(3rd)": "{:.0f}%", "P(4th)": "{:.0f}%",
                })
                st.dataframe(styled, use_container_width=True, hide_index=False)

with tab3:
    st.subheader("Team Power Rankings")
    st.caption("SPI = composite rating (attack + defense), centred at 50.")

    display_df = ratings[["team", "spi", "attack", "defense"]].copy()
    display_df["spi"] = display_df["spi"].round(1)
    display_df["attack"] = display_df["attack"].round(3)
    display_df["defense"] = display_df["defense"].round(3)
    display_df.columns = ["Team", "SPI", "Attack", "Defense"]

    st.dataframe(
        display_df.style.background_gradient(subset=["SPI"], cmap="RdYlGn"),
        use_container_width=True,
        height=600,
    )

# ── Match predictor tab ───────────────────────────────────────────────────────
with tab1:
    st.subheader("Predict a match")

    teams_sorted = sorted(valid_teams)

    col1, col2 = st.columns(2)
    default_a = next((t for t in ["France", "Spain", teams_sorted[0]] if t in teams_sorted), teams_sorted[0])
    default_b = next((t for t in ["Brazil", "Argentina", teams_sorted[1]] if t in teams_sorted), teams_sorted[1])

    with col1:
        team_a = st.selectbox("Team A (home / left)", teams_sorted,
                              index=teams_sorted.index(default_a))
    with col2:
        team_b = st.selectbox("Team B (away / right)", teams_sorted,
                              index=teams_sorted.index(default_b))

    if team_a == team_b:
        st.warning("Please select two different teams.")
        st.stop()

    result = predict_match(team_a, team_b, neutral=neutral,
                           ratings_df=ratings, home_adv=home_adv, rho=rho,
                           valid_teams=valid_teams, params=params)

    # ── Outcome probabilities ─────────────────────────────────────────────────
    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"{team_a} win", f"{result['win_a']*100:.1f}%")
    c2.metric("Draw", f"{result['draw']*100:.1f}%")
    c3.metric(f"{team_b} win", f"{result['win_b']*100:.1f}%")
    c4.metric(f"xG {team_a}", f"{result['lam']:.2f}")
    c5.metric(f"xG {team_b}", f"{result['mu']:.2f}")

    # SPI for each team
    spi_a = ratings.loc[ratings["team"] == team_a, "spi"].values
    spi_b = ratings.loc[ratings["team"] == team_b, "spi"].values
    if len(spi_a) and len(spi_b):
        st.caption(f"SPI — {team_a}: **{spi_a[0]:.1f}** · {team_b}: **{spi_b[0]:.1f}**")

    st.divider()

    col_left, col_right = st.columns([3, 2])

    # ── Score probability heatmap ─────────────────────────────────────────────
    with col_left:
        st.markdown("#### Score probability heatmap")

        max_g = 5  # 0..4 exact goals; index 5 = "5+" bucket
        full_sm = result["score_matrix"]

        # Aggregate into (max_g+1)×(max_g+1), capping at 5+
        sm_capped = np.zeros((max_g + 1, max_g + 1))
        sm_capped[:max_g, :max_g] = full_sm[:max_g, :max_g]
        sm_capped[max_g, :max_g]  = full_sm[max_g:, :max_g].sum(axis=0)
        sm_capped[:max_g, max_g]  = full_sm[:max_g, max_g:].sum(axis=1)
        sm_capped[max_g, max_g]   = full_sm[max_g:, max_g:].sum()
        sm_pct = sm_capped * 100

        goal_range  = list(range(max_g + 1))
        goal_labels = [str(i) for i in range(max_g)] + ["5+"]

        a_marginal = sm_pct.sum(axis=1)
        b_marginal = sm_pct.sum(axis=0)

        fig = plt.figure(figsize=(8.5, 8.5), facecolor="white")
        gs = fig.add_gridspec(2, 2,
                              width_ratios=[1.8, 5], height_ratios=[1.8, 5],
                              hspace=0.30, wspace=0.35)
        ax_tl   = fig.add_subplot(gs[0, 0])
        ax_top  = fig.add_subplot(gs[0, 1])
        ax_left = fig.add_subplot(gs[1, 0])
        ax_main = fig.add_subplot(gs[1, 1])
        ax_tl.axis("off")

        # Heatmap
        im = ax_main.imshow(sm_pct, cmap="YlOrRd", aspect="auto",
                            origin="upper", vmin=0)
        ax_main.set_xticks(goal_range)
        ax_main.set_yticks(goal_range)
        ax_main.set_xticklabels(goal_labels, fontsize=10)
        ax_main.set_yticklabels(goal_labels, fontsize=10)
        ax_main.xaxis.set_ticks_position("top")
        ax_main.xaxis.set_label_position("top")
        ax_main.set_xlabel(f"{team_b} goals", fontsize=11, labelpad=10)
        ax_main.set_ylabel(f"{team_a} goals", fontsize=11, labelpad=10)
        for r in goal_range:
            for c in goal_range:
                val = sm_pct[r, c]
                txt_col = "white" if val > sm_pct.max() * 0.55 else "#333"
                ax_main.text(c, r, f"{val:.1f}", ha="center", va="center",
                             fontsize=8.5, color=txt_col)

        # Team B goal distribution (top, aligns with heatmap columns)
        ax_top.bar(goal_range, b_marginal, color="#3A86FF", width=0.65,
                   alpha=0.85, edgecolor="white", linewidth=0.8)
        ax_top.set_xlim(-0.5, max_g + 0.5)
        ax_top.set_ylim(0, b_marginal.max() * 1.4)
        ax_top.set_xticks([])
        ax_top.set_title(f"{team_b} — goal probability", fontsize=10,
                         fontweight="bold", pad=5)
        ax_top.set_ylabel("%", fontsize=8)
        ax_top.tick_params(axis="y", labelsize=8)
        for k, v in enumerate(b_marginal):
            ax_top.text(k, v + b_marginal.max() * 0.05, f"{v:.0f}%",
                        ha="center", va="bottom", fontsize=8, color="#444")
        for s in ["top", "right", "bottom"]:
            ax_top.spines[s].set_visible(False)

        # Team A goal distribution (left, bars grow leftward away from heatmap)
        ax_left.barh(goal_range, a_marginal, color="#FF6B6B", height=0.65,
                     alpha=0.85, edgecolor="white", linewidth=0.8)
        ax_left.set_ylim(-0.5, max_g + 0.5)
        ax_left.invert_yaxis()
        ax_left.set_xlim(a_marginal.max() * 1.4, 0)  # 0 at right (heatmap side)
        ax_left.set_yticks([])
        ax_left.set_title(f"{team_a}\ngoal prob.", fontsize=10,
                          fontweight="bold", pad=5)
        ax_left.set_xlabel("%", fontsize=8)
        ax_left.tick_params(axis="x", labelsize=8)
        for k, v in enumerate(a_marginal):
            ax_left.text(v + a_marginal.max() * 0.05, k, f"{v:.0f}%",
                         ha="right", va="center", fontsize=8, color="#444")
        for s in ["top", "left", "bottom"]:
            ax_left.spines[s].set_visible(False)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Top scorelines ────────────────────────────────────────────────────────
    with col_right:
        st.markdown("#### Most likely scorelines")
        top = top_scorelines(result, top_n=10)

        fig2, ax2 = plt.subplots(figsize=(4, 5), facecolor="white")
        norm_val = plt.Normalize(top["prob"].min(), top["prob"].max())
        bar_colors = plt.cm.YlOrRd(norm_val(top["prob"]))
        bars = ax2.barh(range(len(top)), top["prob"] * 100,
                        color=bar_colors, edgecolor="white", linewidth=0.5)
        ax2.set_yticks(range(len(top)))
        ax2.set_yticklabels(top["scoreline"], fontsize=10)
        ax2.invert_yaxis()
        ax2.set_xlabel("Probability (%)", fontsize=10)
        ax2.set_title(f"{team_a} vs {team_b}", fontsize=11,
                      fontweight="bold", pad=8)
        for bar, val in zip(bars, top["prob"] * 100):
            ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center", fontsize=9, color="#333")
        ax2.set_xlim(0, top["prob"].max() * 140)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.tick_params(labelsize=9)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ── Full score table ──────────────────────────────────────────────────────
    with st.expander("All scoreline probabilities (top 20)"):
        top20 = top_scorelines(result, top_n=20)
        top20["prob"] = (top20["prob"] * 100).round(2).astype(str) + "%"
        top20.columns = ["Goals A", "Goals B", "Probability", "Scoreline"]
        st.dataframe(top20[["Scoreline", "Probability"]], use_container_width=True)
