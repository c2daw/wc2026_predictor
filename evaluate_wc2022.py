"""
Backtest: train on 2014–2022-11-19, predict every WC2022 match,
evaluate exact result / goal difference / tendency vs 33% baseline.
All results are 90-minute only (martj42 data records 90-min scores).
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
from scipy.special import gammaln
import warnings
warnings.filterwarnings("ignore")

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 1.5,
    "FIFA World Cup qualification": 1.2,
    "UEFA Euro": 1.4,
    "UEFA Euro qualification": 1.2,
    "Copa América": 1.4,
    "Africa Cup of Nations": 1.3,
    "Africa Cup of Nations qualification": 1.1,
    "Asian Cup": 1.3,
    "Asian Cup qualification": 1.1,
    "CONCACAF Gold Cup": 1.2,
    "CONCACAF Gold Cup qualification": 1.0,
    "OFC Nations Cup": 1.1,
    "Friendly": 0.5,
    "Confederations Cup": 1.3,
    "Nations League": 1.1,
    "UEFA Nations League": 1.1,
}

EXCLUDED_PATTERNS = [
    "conifa", "island games", "muratti", "inter games",
    "parish", "inter-insular", "nf-board",
]

DECAY_DAYS = 365 * 3
MIN_MATCHES = 10
LAMBDA_REG = 0.02
MAX_GOALS = 8

WC2022_START = pd.Timestamp("2022-11-20")
WC2022_END   = pd.Timestamp("2022-12-18")
TRAIN_START  = "2014-01-01"


def load_and_split():
    df = pd.read_csv(DATA_URL, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].map({"TRUE": True, "FALSE": False}).fillna(False)

    excl = "|".join(EXCLUDED_PATTERNS)
    df = df[~df["tournament"].str.lower().str.contains(excl, na=False)]

    train = df[
        (df["date"] >= TRAIN_START) &
        (df["date"] < WC2022_START)
    ].copy()

    # Time decay anchored to WC2022 kick-off
    train["days_ago"] = (WC2022_START - train["date"]).dt.days
    train["time_weight"] = np.exp(-np.log(2) * train["days_ago"] / DECAY_DAYS)
    train["tourney_weight"] = train["tournament"].map(
        lambda t: next((v for k, v in TOURNAMENT_WEIGHTS.items() if k.lower() in t.lower()), 0.8)
    )
    train["weight"] = train["time_weight"] * train["tourney_weight"]

    wc22 = df[
        (df["tournament"] == "FIFA World Cup") &
        (df["date"] >= WC2022_START) &
        (df["date"] <= WC2022_END)
    ].copy().sort_values("date").reset_index(drop=True)

    return train, wc22


def _rho_correction_vec(x, y, lam, mu, rho):
    rc = np.ones(len(x))
    rc[(x == 0) & (y == 0)] = 1 - lam[(x == 0) & (y == 0)] * mu[(x == 0) & (y == 0)] * rho
    rc[(x == 0) & (y == 1)] = 1 + lam[(x == 0) & (y == 1)] * rho
    rc[(x == 1) & (y == 0)] = 1 + mu[(x == 1) & (y == 0)] * rho
    rc[(x == 1) & (y == 1)] = 1 - rho
    return np.maximum(rc, 1e-10)


def _neg_log_likelihood(params, teams, arrays):
    n = len(teams)
    attack, defense = params[:n], params[n:2*n]
    home_adv, rho = params[2*n], params[2*n + 1]

    hi_idx, ai_idx, x, y, weights, ha_vec = arrays
    lam = np.exp(attack[hi_idx] - defense[ai_idx] + ha_vec * home_adv)
    mu  = np.exp(attack[ai_idx] - defense[hi_idx])

    log_pmf_x = x * np.log(lam) - lam - gammaln(x + 1)
    log_pmf_y = y * np.log(mu)  - mu  - gammaln(y + 1)
    rho_c = _rho_correction_vec(x, y, lam, mu, rho)

    ll = weights * (np.log(rho_c) + log_pmf_x + log_pmf_y)
    reg = LAMBDA_REG * (np.sum(attack**2) + np.sum(defense**2))
    return -ll.sum() + reg


def fit_model(df):
    all_teams = pd.concat([df["home_team"], df["away_team"]])
    valid_teams = sorted(
        all_teams.value_counts()[lambda s: s >= MIN_MATCHES].index.tolist()
    )
    team_idx = {t: i for i, t in enumerate(valid_teams)}

    df_fit = df[df["home_team"].isin(team_idx) & df["away_team"].isin(team_idx)].copy()
    arrays = (
        df_fit["home_team"].map(team_idx).to_numpy(dtype=np.int32),
        df_fit["away_team"].map(team_idx).to_numpy(dtype=np.int32),
        df_fit["home_score"].to_numpy(dtype=np.float64),
        df_fit["away_score"].to_numpy(dtype=np.float64),
        df_fit["weight"].to_numpy(dtype=np.float64),
        (~df_fit["neutral"]).to_numpy(dtype=np.float64),
    )

    n = len(valid_teams)
    x0 = np.zeros(2 * n + 2)
    x0[2 * n] = 0.3
    x0[2 * n + 1] = -0.1
    constraints = [{"type": "eq", "fun": lambda p: np.sum(p[:n])}]

    res = minimize(
        _neg_log_likelihood, x0,
        args=(valid_teams, arrays),
        method="SLSQP",
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-6},
    )
    return valid_teams, res.x


def score_matrix(team_a, team_b, valid_teams, params):
    team_idx = {t: i for i, t in enumerate(valid_teams)}
    n = len(valid_teams)
    attack, defense = params[:n], params[n:2*n]
    rho = params[2*n + 1]

    ai, bi = team_idx[team_a], team_idx[team_b]
    lam = np.exp(attack[ai] - defense[bi])   # neutral venue: no home_adv
    mu  = np.exp(attack[bi] - defense[ai])

    sm = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    for x in range(MAX_GOALS + 1):
        for y in range(MAX_GOALS + 1):
            rc = (
                1 - lam * mu * rho if x == 0 and y == 0 else
                1 + lam * rho      if x == 0 and y == 1 else
                1 + mu * rho       if x == 1 and y == 0 else
                1 - rho            if x == 1 and y == 1 else
                1.0
            )
            sm[x, y] = max(rc * poisson.pmf(x, lam) * poisson.pmf(y, mu), 0)
    sm /= sm.sum()
    return sm, lam, mu


def tendency(h, a):
    return "home" if h > a else ("away" if h < a else "draw")


def main():
    print("Loading data …")
    train, wc22 = load_and_split()
    print(f"  Training rows : {len(train):,}")
    print(f"  WC2022 matches: {len(wc22)}")

    print("Fitting Dixon-Coles model (this takes ~30 s) …")
    valid_teams, params = fit_model(train)
    print(f"  Teams in model: {len(valid_teams)}")

    rows = []
    skipped = []

    for _, m in wc22.iterrows():
        h, a = m["home_team"], m["away_team"]
        ah, aa = int(m["home_score"]), int(m["away_score"])

        if h not in valid_teams or a not in valid_teams:
            skipped.append(f"{h} vs {a}  (missing: {[t for t in [h,a] if t not in valid_teams]})")
            continue

        sm, lam, mu = score_matrix(h, a, valid_teams, params)
        pi, pj = np.unravel_index(np.argmax(sm), sm.shape)

        win_h = float(np.tril(sm, -1).sum())
        drw   = float(np.trace(sm))
        win_a = float(np.triu(sm, 1).sum())
        pred_tend = tendency(win_h, win_a) if max(win_h, win_a) != drw else "draw"
        # Predicted tendency from most-likely scoreline
        pred_tend_ml = tendency(pi, pj)

        rows.append({
            "date":          m["date"].date(),
            "home":          h,
            "away":          a,
            "actual":        f"{ah}-{aa}",
            "predicted":     f"{pi}-{pj}",
            "actual_tend":   tendency(ah, aa),
            "pred_tend":     pred_tend_ml,
            "right_result":  (pi == ah) and (pj == aa),
            "right_gd":      (pi - pj) == (ah - aa),
            "right_tend":    pred_tend_ml == tendency(ah, aa),
            "xG_home":       round(lam, 2),
            "xG_away":       round(mu, 2),
            "p_home":        round(win_h, 3),
            "p_draw":        round(drw, 3),
            "p_away":        round(win_a, 3),
        })

    df = pd.DataFrame(rows)
    n = len(df)

    # ── Per-match table ───────────────────────────────────────────────────────
    col_w = 28
    print(f"\n{'Match':<{col_w}} {'Date':<12} {'Act':>5} {'Pred':>5}  {'R':>2} {'GD':>2} {'T':>2}  {'xG':>9}  {'P(H/D/A)'}")
    print("─" * 100)
    for _, r in df.iterrows():
        xg  = f"{r['xG_home']:.2f}-{r['xG_away']:.2f}"
        prb = f"{r['p_home']:.0%}/{r['p_draw']:.0%}/{r['p_away']:.0%}"
        print(
            f"{r['home']+' vs '+r['away']:<{col_w}} "
            f"{str(r['date']):<12} "
            f"{r['actual']:>5} {r['predicted']:>5}  "
            f"{'✓' if r['right_result'] else '✗':>2} "
            f"{'✓' if r['right_gd'] else '✗':>2} "
            f"{'✓' if r['right_tend'] else '✗':>2}  "
            f"{xg:>9}  {prb}"
        )

    # ── Phase breakdown ───────────────────────────────────────────────────────
    # Identify phase by date (rough WC2022 schedule)
    def phase(d):
        d = pd.Timestamp(d)
        if d <= pd.Timestamp("2022-12-02"): return "Group stage"
        if d <= pd.Timestamp("2022-12-06"): return "Round of 16"
        if d <= pd.Timestamp("2022-12-10"): return "Quarter-finals"
        if d <= pd.Timestamp("2022-12-14"): return "Semi-finals"
        return "Final / 3rd place"

    df["phase"] = df["date"].apply(phase)

    print("\n\n── By phase ─────────────────────────────────────────────────────────────")
    print(f"{'Phase':<20} {'N':>4}  {'Tendency':>9}  {'Goal Diff':>9}  {'Exact':>9}")
    print("─" * 60)
    for ph, g in df.groupby("phase", sort=False):
        ng = len(g)
        print(
            f"{ph:<20} {ng:>4}  "
            f"{g['right_tend'].mean():>8.1%}  "
            f"{g['right_gd'].mean():>9.1%}  "
            f"{g['right_result'].mean():>9.1%}"
        )

    # ── Overall summary ───────────────────────────────────────────────────────
    baseline = 1 / 3

    rt  = df["right_tend"].mean()
    rgd = df["right_gd"].mean()
    rr  = df["right_result"].mean()

    print("\n\n── Overall summary ──────────────────────────────────────────────────────")
    print(f"  Total matches evaluated : {n}")
    if skipped:
        print(f"  Skipped (team not in model): {len(skipped)}")
        for s in skipped:
            print(f"    • {s}")

    print(f"\n  {'Metric':<25} {'Model':>8}  {'Baseline':>8}  {'Δ':>8}")
    print(f"  {'─'*52}")
    print(f"  {'Tendency (W/D/L)':<25} {rt:>8.1%}  {baseline:>8.1%}  {rt-baseline:>+8.1%}")
    print(f"  {'Goal Difference':<25} {rgd:>8.1%}  {'  —':>8}  {'':>8}")
    print(f"  {'Exact Result':<25} {rr:>8.1%}  {'  —':>8}  {'':>8}")

    print(f"\n  Note: Baseline = random 3-way coin flip (33.3% per outcome).")
    print(f"  Right GD ⊆ Right Tendency, Right Result ⊆ Right GD.")

    # ── Actual vs predicted tendency distribution ─────────────────────────────
    print("\n\n── Tendency confusion (actual rows × predicted cols) ────────────────────")
    cats = ["home", "draw", "away"]
    conf = pd.crosstab(df["actual_tend"], df["pred_tend"]).reindex(index=cats, columns=cats, fill_value=0)
    print(conf.to_string())

    # ── Most common wrong calls ───────────────────────────────────────────────
    wrong = df[~df["right_tend"]].copy()
    if not wrong.empty:
        print("\n\n── Wrong tendency predictions ───────────────────────────────────────────")
        for _, r in wrong.iterrows():
            print(f"  {r['home']} vs {r['away']:20s}  actual {r['actual']} ({r['actual_tend']:4s})  "
                  f"pred {r['predicted']} ({r['pred_tend']:4s})  "
                  f"P={r['p_home']:.0%}/{r['p_draw']:.0%}/{r['p_away']:.0%}")


if __name__ == "__main__":
    main()
