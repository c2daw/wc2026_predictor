# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

The project has two independent layers:

- **Python backend** (`model.py`, `precompute.py`) — statistical model, data fetching, simulation
- **React frontend** (`frontend/`) — static site, deployed to Vercel

The frontend has no server. All predictions run client-side using pre-computed model parameters loaded as JSON.

## Running the Frontend (Dev)

```bash
cd frontend
npm install        # first time only
npm run dev        # starts at http://localhost:5173
```

The dev server loads data from `frontend/public/data/`. Run `precompute.py` first if that directory is empty.

## Refreshing Pre-computed Data

Run this after new match results are available (fetches live data, fits model, runs all simulations):

```bash
python precompute.py
git add model_cache.json frontend/public/data/ && git commit -m "Refresh model cache" && git push
```

Vercel redeploys automatically on push. This takes ~2–3 minutes to run.

`precompute.py` generates:
- `model_cache.json` — model params (Streamlit fallback, kept for compatibility)
- `frontend/public/data/model.json` — team params used by the Match Simulator (JS Poisson math)
- `frontend/public/data/groups.json` — 12 group simulation results (20k sims)
- `frontend/public/data/bracket.html` — pre-rendered KO bracket with popup heatmaps
- `frontend/public/data/tournament.json` — round-by-round probabilities + SPI rankings

## Deploying to Vercel

1. Connect GitHub repo to Vercel
2. Set **Root Directory** to `frontend`
3. Build command: `npm run build`, output: `dist`
4. Done — `vercel.json` in `frontend/` handles the rest

## Running Evaluation Scripts

Standalone CLI scripts — run directly with Python:

```bash
python evaluate_wc2022.py          # Backtest vs WC2022 results (tendency/GD/exact accuracy)
python evaluate_boldness.py        # Sweep boldness factor k across WC2022 + Euro 2024
python evaluate_points.py          # Points-scoring backtest
```

## Frontend Architecture

**`frontend/src/`**
- `App.jsx` — tab shell, fetches all three JSON data files on load
- `tabs/MatchSimulator.jsx` — team selectors + heatmap + outcome pills + xG + scorelines
- `tabs/GroupStage.jsx` — 12 group cards with colored HTML tables
- `tabs/KOBracket.jsx` — bracket iframe + round-by-round table + SPI rankings
- `tabs/Backtest.jsx` — hardcoded backtest table
- `utils/poisson.js` — `predictMatch()` re-implements Dixon-Coles in JS using model params
- `utils/colors.js` — `getCellColor()` matches Plotly color scales for the heatmap
- `utils/teamMeta.js` — flag/code constants

**Match Simulator data flow**: `model.json` → `predictMatch(teamA, teamB, modelData)` (pure JS) → score matrix → heatmap grid (CSS)

## Python Model (`model.py`)

- Fetches international results from `martj42/international_results` on GitHub
- **Dixon-Coles Poisson model**: attack/defense ratings, home advantage, low-score rho correction
- Time-decay (3-year half-life), tournament weighting (World Cup = 1.5×, Friendly = 0.5×)
- Teams with fewer than `MIN_MATCHES=10` excluded
- `fit_model()` uses `scipy.optimize.minimize` (SLSQP); attack params sum-to-zero constraint
- `WC2026_GROUPS` dict (groups A–L) hardcoded here

Model parameters array: `params[:n]` = attack, `params[n:2n]` = defense, `params[2n]` = home_adv, `params[2n+1]` = rho.

## Key Design Constraints

- Match Simulator predictions are **neutral** (no home advantage applied)
- KO bracket uses the **most-likely finisher** per group position (from group sim), not the MC tournament sim — the two are independent
- `simulate_tournament()` approximates best-3rd-place selection by random shuffle (not points-ranked)
- Backtest tab uses hardcoded results — it does not recompute from the live model
- `app.py` (Streamlit) is kept for local reference but is no longer the primary UI
