"""
Test whether scaling expected goals upward (boldness factor k) improves points.

Two prediction modes:
  - "argmax": most probable scoreline from full score matrix (current behaviour)
  - "round(k*lam) - round(k*mu)": directly scale expected goals before rounding

Sweeps k from 0.8 to 3.0 for both WC2022 and Euro 2024.
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


def load_all():
    df = pd.read_csv(DATA_URL, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].map({"TRUE": True, "FALSE": False}).fillna(False)
    excl = "|".join(EXCLUDED_PATTERNS)
    df = df[~df["tournament"].str.lower().str.contains(excl, na=False)]
    return df


def make_train(df, cutoff):
    t = df[(df["date"] >= "2014-01-01") & (df["date"] < cutoff)].copy()
    t["days_ago"] = (cutoff - t["date"]).dt.days
    t["time_weight"] = np.exp(-np.log(2) * t["days_ago"] / DECAY_DAYS)
    t["tourney_weight"] = t["tournament"].map(
        lambda s: next((v for k, v in TOURNAMENT_WEIGHTS.items() if k.lower() in s.lower()), 0.8)
    )
    t["weight"] = t["time_weight"] * t["tourney_weight"]
    return t


def _rho(x, y, lam, mu, rho):
    if x == 0 and y == 0: return 1 - lam * mu * rho
    if x == 0 and y == 1: return 1 + lam * rho
    if x == 1 and y == 0: return 1 + mu * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0


def _nll(params, teams, arrays):
    n = len(teams)
    atk, dfs = params[:n], params[n:2*n]
    ha, rho = params[2*n], params[2*n+1]
    hi, ai, x, y, w, hv = arrays
    lam = np.exp(atk[hi] - dfs[ai] + hv * ha)
    mu  = np.exp(atk[ai] - dfs[hi])
    lx = x * np.log(lam) - lam - gammaln(x + 1)
    ly = y * np.log(mu)  - mu  - gammaln(y + 1)
    rc = np.ones(len(x))
    rc[(x==0)&(y==0)] = 1 - lam[(x==0)&(y==0)] * mu[(x==0)&(y==0)] * rho
    rc[(x==0)&(y==1)] = 1 + lam[(x==0)&(y==1)] * rho
    rc[(x==1)&(y==0)] = 1 + mu[(x==1)&(y==0)] * rho
    rc[(x==1)&(y==1)] = 1 - rho
    rc = np.maximum(rc, 1e-10)
    ll = w * (np.log(rc) + lx + ly)
    return -ll.sum() + LAMBDA_REG * (np.sum(atk**2) + np.sum(dfs**2))


def fit_model(df):
    all_teams = pd.concat([df["home_team"], df["away_team"]])
    valid = sorted(all_teams.value_counts()[lambda s: s >= MIN_MATCHES].index.tolist())
    idx = {t: i for i, t in enumerate(valid)}
    d = df[df["home_team"].isin(idx) & df["away_team"].isin(idx)].copy()
    arrays = (
        d["home_team"].map(idx).to_numpy(np.int32),
        d["away_team"].map(idx).to_numpy(np.int32),
        d["home_score"].to_numpy(np.float64),
        d["away_score"].to_numpy(np.float64),
        d["weight"].to_numpy(np.float64),
        (~d["neutral"]).to_numpy(np.float64),
    )
    n = len(valid)
    x0 = np.zeros(2*n+2); x0[2*n] = 0.3; x0[2*n+1] = -0.1
    res = minimize(_nll, x0, args=(valid, arrays), method="SLSQP",
                   constraints=[{"type": "eq", "fun": lambda p: np.sum(p[:n])}],
                   options={"maxiter": 500, "ftol": 1e-6})
    return valid, res.x


def get_lam_mu(h, a, valid, params):
    idx = {t: i for i, t in enumerate(valid)}
    n = len(valid)
    atk, dfs, rho_v = params[:n], params[n:2*n], params[2*n+1]
    ai, bi = idx[h], idx[a]
    lam = float(np.exp(atk[ai] - dfs[bi]))
    mu  = float(np.exp(atk[bi] - dfs[ai]))
    return lam, mu, rho_v


def argmax_score(lam, mu, rho_v):
    sm = np.zeros((MAX_GOALS+1, MAX_GOALS+1))
    for x in range(MAX_GOALS+1):
        for y in range(MAX_GOALS+1):
            sm[x, y] = max(_rho(x, y, lam, mu, rho_v) * poisson.pmf(x, lam) * poisson.pmf(y, mu), 0)
    sm /= sm.sum()
    r, c = np.unravel_index(np.argmax(sm), sm.shape)
    return int(r), int(c)


def bold_score(lam, mu, k):
    """Round(k*lam) - Round(k*mu), clipped to [0, MAX_GOALS]."""
    ph = int(min(round(k * lam), MAX_GOALS))
    pa = int(min(round(k * mu), MAX_GOALS))
    return ph, pa


def pts(ph, pa, ah, aa):
    if ph == ah and pa == aa:                           return 4, "exact"
    if (ph - pa) == (ah - aa):                          return 2, "gd"
    if (ph > pa) == (ah > aa) and (ph==pa) == (ah==aa): return 1, "tend"
    return 0, "miss"


def sweep(games, k_values):
    """games: list of (lam, mu, rho_v, actual_h, actual_a)"""
    results = []
    for k in k_values:
        exact = gd = tend = miss = 0
        for lam, mu, rho_v, ah, aa in games:
            if k == 1.0:
                ph, pa = argmax_score(lam, mu, rho_v)   # true argmax at k=1
            else:
                ph, pa = bold_score(lam, mu, k)
            p, cat = pts(ph, pa, ah, aa)
            if cat == "exact": exact += 1
            elif cat == "gd":  gd    += 1
            elif cat == "tend":tend  += 1
            else:              miss  += 1
        total = exact*4 + gd*2 + tend
        results.append({"k": k, "exact": exact, "gd": gd, "tend": tend,
                        "miss": miss, "total": total})
    return pd.DataFrame(results)


def prepare_games(test_df, valid, params):
    games = []
    for _, m in test_df.iterrows():
        h, a = m["home_team"], m["away_team"]
        if h not in valid or a not in valid:
            continue
        lam, mu, rho_v = get_lam_mu(h, a, valid, params)
        games.append((lam, mu, rho_v, int(m["home_score"]), int(m["away_score"])))
    return games


def show_distribution_examples(games_raw):
    """Show a few examples of why argmax is conservative."""
    print("\n  Distribution examples (why argmax is conservative):")
    print(f"  {'xG home':>8}  {'xG away':>8}  {'Argmax':>8}  {'round(xG)':>10}  {'round(1.5×xG)':>14}")
    print(f"  {'─'*55}")
    shown = 0
    for lam, mu, rho_v, ah, aa in games_raw:
        ph_am, pa_am = argmax_score(lam, mu, rho_v)
        ph_r,  pa_r  = bold_score(lam, mu, 1.0)   # round at k=1
        ph_b,  pa_b  = bold_score(lam, mu, 1.5)
        if ph_am != ph_r or ph_r != ph_b:          # only show interesting cases
            print(f"  {lam:>8.2f}  {mu:>8.2f}  {ph_am}-{pa_am}  →  "
                  f"{ph_r}-{pa_r}  →  {ph_b}-{pa_b}   (actual {ah}-{aa})")
            shown += 1
        if shown >= 12:
            break


def run(df_all, tourney_filter, cutoff, start, end, label):
    print(f"\n{'='*68}")
    print(f"  {label}")
    print(f"{'='*68}")

    train = make_train(df_all, cutoff)
    test = df_all[
        df_all["tournament"].str.contains(tourney_filter, case=False, na=False) &
        (df_all["date"] >= start) & (df_all["date"] <= end)
    ].sort_values("date").reset_index(drop=True)
    print(f"  Training rows: {len(train):,}  |  Test matches: {len(test)}")

    print("  Fitting model …")
    valid, params = fit_model(train)

    games = prepare_games(test, valid, params)
    print(f"  Predicted games: {len(games)}")

    show_distribution_examples(games)

    # Sweep k for round(k*xG) strategy, plus pure argmax at k=1
    k_values = [round(k, 1) for k in np.arange(0.8, 3.05, 0.1)]
    df_sweep = sweep(games, k_values)

    best_idx = df_sweep["total"].idxmax()
    best = df_sweep.loc[best_idx]

    print(f"\n  k-sweep results (bold = best total):")
    print(f"  {'k':>5}  {'Exact':>6}  {'GD':>4}  {'Tend':>5}  {'Miss':>5}  {'Total':>6}")
    print(f"  {'─'*40}")
    for _, r in df_sweep.iterrows():
        flag = " ◀ best" if r["k"] == best["k"] else ""
        marker = "**" if r["k"] == best["k"] else "  "
        print(f"  {marker}{r['k']:>4.1f}  {int(r['exact']):>6}  {int(r['gd']):>4}  "
              f"{int(r['tend']):>5}  {int(r['miss']):>5}  {int(r['total']):>6}{flag}")

    # Argmax baseline (k=1.0)
    base = df_sweep[df_sweep["k"] == 1.0].iloc[0]
    print(f"\n  Argmax (k=1.0) : {int(base['total'])} pts  "
          f"({int(base['exact'])} exact / {int(base['gd'])} gd / {int(base['tend'])} tend)")
    print(f"  Best (k={best['k']:.1f})    : {int(best['total'])} pts  "
          f"({int(best['exact'])} exact / {int(best['gd'])} gd / {int(best['tend'])} tend)  "
          f"[{int(best['total'])-int(base['total']):+d} pts vs argmax]")

    return df_sweep, int(base["total"]), int(best["total"]), float(best["k"]), len(games)


def main():
    print("Loading data …")
    df_all = load_all()

    sw_wc, wc_base, wc_best, wc_k, wc_n = run(
        df_all,
        "FIFA World Cup",
        pd.Timestamp("2022-11-20"),
        pd.Timestamp("2022-11-20"),
        pd.Timestamp("2022-12-18"),
        "WC 2022  (train: 2014–2022-11-19)",
    )

    sw_eu, eu_base, eu_best, eu_k, eu_n = run(
        df_all,
        "UEFA Euro$",
        pd.Timestamp("2024-06-14"),
        pd.Timestamp("2024-06-14"),
        pd.Timestamp("2024-07-14"),
        "Euro 2024  (train: 2014–2024-06-13)",
    )

    print(f"\n\n{'='*68}")
    print("  SUMMARY")
    print(f"{'='*68}")
    print(f"  {'':30} {'WC2022':>8}  {'Euro24':>8}")
    print(f"  {'─'*50}")
    print(f"  {'Games':<30} {wc_n:>8}  {eu_n:>8}")
    print(f"  {'Argmax (k=1.0)':<30} {wc_base:>8}  {eu_base:>8}")
    print(f"  {'Best k':<30} {wc_k:>8.1f}  {eu_k:>8.1f}")
    print(f"  {'Best score':<30} {wc_best:>8}  {eu_best:>8}")
    print(f"  {'Improvement':<30} {wc_best-wc_base:>+8}  {eu_best-eu_base:>+8}")
    print(f"  {'% of max possible':<30} {wc_best/(wc_n*4):>8.1%}  {eu_best/(eu_n*4):>8.1%}")

    print(f"\n  Euro 2024 — with best k={eu_k:.1f} vs reference players:")
    ref = [(13,4,9,69),(8,6,9,53),(7,1,13,43),(6,6,12,48)]
    all_scores = sorted([r[3] for r in ref] + [eu_best], reverse=True)
    rank = all_scores.index(eu_best) + 1
    for i, (e,g,t,tot) in enumerate(ref, 1):
        print(f"    #{i} human: {tot} pts")
    print(f"    Model (k={eu_k:.1f}): {eu_best} pts  → rank #{rank}/{len(all_scores)}")


if __name__ == "__main__":
    main()
