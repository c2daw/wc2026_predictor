# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

The app runs at http://localhost:8501. There is no build step, test suite, or linter configured.

## Running Evaluation Scripts

These are standalone CLI scripts — run them directly with Python:

```bash
python evaluate_wc2022.py          # Backtest vs WC2022 results (tendency/GD/exact accuracy)
python evaluate_boldness.py        # Sweep boldness factor k across WC2022 + Euro 2024
python evaluate_points.py          # Points-scoring backtest
```

Each script fetches data, fits its own model, and prints results to stdout. They share the same model logic as `model.py` but are self-contained (they duplicate constants rather than importing from `model.py`).

## Architecture

Two core files at the repo root:

**`model.py`** — Statistical engine. All data fetching, model fitting, and prediction logic.
- Fetches international match results from `martj42/international_results` on GitHub
- Implements the **Dixon-Coles Poisson model**: separate attack/defense ratings per team, home advantage, and a low-score correction factor (`rho`) for 0-0/0-1/1-0/1-1 scorelines
- Time-decay weighting (3-year half-life) and tournament weighting (World Cup = 1.5×, Friendly = 0.5×)
- Teams with fewer than `MIN_MATCHES=10` appearances are excluded from the model
- `fit_model()` uses `scipy.optimize.minimize` (SLSQP) with the constraint that attack parameters sum to zero (identifiability); also returns an SPI rating (normalized 50 ± 10 composite score)
- `predict_match()` returns an (N×N) scoreline probability matrix
- `best_scoreline()` picks the score that maximises expected points under a tiered scoring system (4/2/1 for exact/GD/tendency)
- `simulate_group()` runs 20,000 Monte-Carlo group stage simulations using score matrices with rho correction
- `simulate_tournament()` runs 8,000 full WC2026 Monte-Carlo simulations (group → R32 → R16 → QF → SF → Final) using faster Poisson sampling without rho correction
- `WC2026_GROUPS` dict (groups A–L) is hardcoded here

**`app.py`** — Streamlit UI. Five tabs:
1. **Match Simulator** — score probability heatmap (tricolor: red=team A, navy=team B, grey=draw), outcome pills, xG bars, most-likely scorelines
2. **Group Stage** — Monte-Carlo group tables with xPts/xGD/xGF and finish-position probabilities
3. **KO Bracket** — most-likely bracket + round-by-round championship probabilities
4. **Rankings** — SPI table with attack/defense parameters
5. **Backtest** — pre-computed Round(xG) results for WC2022 and Euro 2024

All settings are hardcoded (`MIN_DATE = "2014-01-01"`, `NEUTRAL = True`). There is no sidebar. `@st.cache_data` with 1-hour TTL wraps model fitting and all simulations.

## Refreshing the Model Cache

`app.py` loads pre-computed model parameters from `model_cache.json` on startup instead of fetching data and fitting the model live (eliminates the cold-start delay on Streamlit Cloud). Run this after new match results are available:

```bash
python precompute.py
git add model_cache.json && git commit -m "Refresh model cache" && git push
```

Streamlit Cloud redeploys automatically on push. If `model_cache.json` is missing, the app falls back to live fitting.

## Data Flow

```
load_data() → fit_model() → predict_match() / simulate_group() / simulate_tournament() → Streamlit UI
```

Model parameters array layout: `params[:n]` = attack, `params[n:2n]` = defense, `params[2n]` = home_adv, `params[2n+1]` = rho.

## Key Design Constraints

- Unknown teams in group simulation fall back to (0, 0) attack/defense (league average)
- KO bracket in `get_bracket()` uses the most-likely finisher per position from group simulation, not the MC tournament simulation — the two are independent
- `simulate_tournament()` approximates best-3rd-place selection by randomly shuffling third-place teams rather than ranking by points
- The Backtest tab (tab 5) uses hardcoded results — it does not recompute from the live model
