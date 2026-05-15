"""
WC2022 backtest — betting comparison.
Train on 2014–2022-11-19, predict every WC2022 match, compare vs bookmaker odds.

Two strategies:
  1. Top-pick: 1 unit on whichever outcome the model rates highest, per game
  2. Value bets: 1 unit on every outcome where model_prob > bookie_implied_prob

Odds source: checkbestodds.com (best available decimal odds per match)
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

# ---------------------------------------------------------------------------
# Decimal odds (home / draw / away) from checkbestodds.com
# These are the best available odds recorded for each match.
# ---------------------------------------------------------------------------
WC2022_ODDS = {
    # Group A
    ("Qatar", "Ecuador"):              (100.00, 13.50,  2.41),
    ("Senegal", "Netherlands"):        (  6.28,  3.75,  2.09),
    ("Qatar", "Senegal"):              ( 18.00,  6.20,  1.77),
    ("Netherlands", "Ecuador"):        (  1.90,  4.70, 15.00),
    ("Netherlands", "Qatar"):          (  1.22, 10.00, 60.00),
    ("Ecuador", "Senegal"):            (  6.80,  4.00,  3.35),
    # Group B
    ("England", "Iran"):               (  1.40, 10.50, 110.00),
    ("United States", "Wales"):        (  2.53,  4.60, 12.50),
    ("Wales", "Iran"):                 (  2.61,  3.26,  4.31),
    ("England", "United States"):      (  1.91,  4.45,  9.08),
    ("Iran", "United States"):         ( 18.00,  5.40,  2.05),
    ("Wales", "England"):              (  9.91,  4.70,  1.56),
    # Group C
    ("Argentina", "Saudi Arabia"):     (  1.14, 17.00, 90.00),
    ("Mexico", "Poland"):              (  2.59,  3.25,  4.50),
    ("Poland", "Saudi Arabia"):        (  1.75,  3.70,  5.80),
    ("Argentina", "Mexico"):           (  2.12,  4.10,  7.97),
    ("Saudi Arabia", "Mexico"):        (  5.10,  4.20,  1.74),
    ("Poland", "Argentina"):           (  9.20,  4.50,  1.49),
    # Group D
    ("Denmark", "Tunisia"):            (  2.00,  4.10,  7.50),
    ("France", "Australia"):           (  1.23, 10.00, 75.00),
    ("Tunisia", "Australia"):          (  6.20,  3.50,  3.90),
    ("France", "Denmark"):             (  2.19,  3.64,  5.10),
    ("Australia", "Denmark"):          (  7.40,  4.50,  1.56),
    ("Tunisia", "France"):             (  9.10,  5.00,  1.43),
    # Group E
    ("Germany", "Japan"):              (  1.48,  8.75, 22.00),
    ("Spain", "Costa Rica"):           (  1.17, 18.50, 40.00),
    ("Japan", "Costa Rica"):           (  1.46,  4.60,  9.50),
    ("Spain", "Germany"):              (  2.44,  3.75,  3.08),
    ("Costa Rica", "Germany"):         (100.00, 14.00,  1.12),
    ("Japan", "Spain"):                ( 23.00,  6.80,  1.53),
    # Group F
    ("Morocco", "Croatia"):            (  3.88,  3.35,  2.88),
    ("Belgium", "Canada"):             (  1.62,  4.70,  7.60),
    ("Belgium", "Morocco"):            (  2.02,  3.42,  4.33),
    ("Croatia", "Canada"):             (  2.23,  3.48,  3.50),
    ("Canada", "Morocco"):             ( 14.50,  5.60,  1.86),
    ("Croatia", "Belgium"):            (  3.22,  3.55,  2.77),
    # Group G
    ("Switzerland", "Cameroon"):       (  2.19,  3.56,  5.20),
    ("Brazil", "Serbia"):              (  1.49,  5.20,  9.50),
    ("Cameroon", "Serbia"):            (  5.35,  3.85,  1.75),
    ("Brazil", "Switzerland"):         (  1.53,  4.70,  7.50),
    ("Cameroon", "Brazil"):            (  8.60,  5.15,  1.58),
    ("Serbia", "Switzerland"):         (  2.70,  3.60,  2.96),
    # Group H
    ("Uruguay", "South Korea"):        (  2.05,  3.64,  5.40),
    ("Portugal", "Ghana"):             (  1.55,  5.70, 11.50),
    ("South Korea", "Ghana"):          (  2.60,  3.15,  3.13),
    ("Portugal", "Uruguay"):           (  2.01,  3.45,  4.25),
    ("South Korea", "Portugal"):       (  4.00,  3.92,  2.00),
    ("Ghana", "Uruguay"):              (  4.44,  3.74,  1.95),
    # Round of 16
    ("Netherlands", "United States"):  (  1.96,  3.55,  4.90),
    ("Argentina", "Australia"):        (  1.30,  6.50, 11.25),
    ("France", "Poland"):              (  1.41,  5.30, 10.00),
    ("England", "Senegal"):            (  1.64,  3.82,  6.95),
    ("Japan", "Croatia"):              (  3.95,  3.34,  2.14),
    ("Brazil", "South Korea"):         (  1.24,  7.20, 17.00),
    ("Morocco", "Spain"):              (  6.25,  3.90,  1.65),
    ("Portugal", "Switzerland"):       (  2.06,  3.72,  4.76),
    # Quarter-finals
    ("Croatia", "Brazil"):             (  9.00,  4.85,  1.43),
    ("Netherlands", "Argentina"):      (  4.14,  3.17,  2.29),
    ("Morocco", "Portugal"):           (  6.75,  4.00,  1.65),
    ("England", "France"):             (  3.17,  3.25,  2.81),
    # Semi-finals
    ("Argentina", "Croatia"):          (  2.09,  3.20,  4.95),
    ("France", "Morocco"):             (  1.66,  4.07,  6.75),
    # 3rd place & Final
    ("Croatia", "Morocco"):            (  2.29,  3.70,  3.20),
    ("Argentina", "France"):           (  2.77,  3.28,  2.96),
}


# ---------------------------------------------------------------------------
# Model (same as evaluate_wc2022.py)
# ---------------------------------------------------------------------------

def load_and_split():
    df = pd.read_csv(DATA_URL, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].map({"TRUE": True, "FALSE": False}).fillna(False)
    excl = "|".join(EXCLUDED_PATTERNS)
    df = df[~df["tournament"].str.lower().str.contains(excl, na=False)]

    train = df[(df["date"] >= TRAIN_START) & (df["date"] < WC2022_START)].copy()
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


def _rho_correction(x, y, lam, mu, rho):
    if x == 0 and y == 0: return 1 - lam * mu * rho
    if x == 0 and y == 1: return 1 + lam * rho
    if x == 1 and y == 0: return 1 + mu * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0


def _neg_log_likelihood(params, teams, arrays):
    n = len(teams)
    attack, defense = params[:n], params[n:2*n]
    home_adv, rho = params[2*n], params[2*n + 1]
    hi_idx, ai_idx, x, y, weights, ha_vec = arrays
    lam = np.exp(attack[hi_idx] - defense[ai_idx] + ha_vec * home_adv)
    mu  = np.exp(attack[ai_idx] - defense[hi_idx])
    log_pmf_x = x * np.log(lam) - lam - gammaln(x + 1)
    log_pmf_y = y * np.log(mu)  - mu  - gammaln(y + 1)
    rho_c = np.ones(len(x))
    rho_c[(x==0)&(y==0)] = 1 - lam[(x==0)&(y==0)] * mu[(x==0)&(y==0)] * rho
    rho_c[(x==0)&(y==1)] = 1 + lam[(x==0)&(y==1)] * rho
    rho_c[(x==1)&(y==0)] = 1 + mu[(x==1)&(y==0)] * rho
    rho_c[(x==1)&(y==1)] = 1 - rho
    rho_c = np.maximum(rho_c, 1e-10)
    ll = weights * (np.log(rho_c) + log_pmf_x + log_pmf_y)
    reg = LAMBDA_REG * (np.sum(attack**2) + np.sum(defense**2))
    return -ll.sum() + reg


def fit_model(df):
    all_teams = pd.concat([df["home_team"], df["away_team"]])
    valid_teams = sorted(all_teams.value_counts()[lambda s: s >= MIN_MATCHES].index.tolist())
    team_idx = {t: i for i, t in enumerate(valid_teams)}
    df_fit = df[df["home_team"].isin(team_idx) & df["away_team"].isin(team_idx)].copy()
    arrays = (
        df_fit["home_team"].map(team_idx).to_numpy(np.int32),
        df_fit["away_team"].map(team_idx).to_numpy(np.int32),
        df_fit["home_score"].to_numpy(np.float64),
        df_fit["away_score"].to_numpy(np.float64),
        df_fit["weight"].to_numpy(np.float64),
        (~df_fit["neutral"]).to_numpy(np.float64),
    )
    n = len(valid_teams)
    x0 = np.zeros(2 * n + 2); x0[2*n] = 0.3; x0[2*n+1] = -0.1
    constraints = [{"type": "eq", "fun": lambda p: np.sum(p[:n])}]
    res = minimize(_neg_log_likelihood, x0, args=(valid_teams, arrays),
                   method="SLSQP", constraints=constraints,
                   options={"maxiter": 500, "ftol": 1e-6})
    return valid_teams, res.x


def get_probs(team_a, team_b, valid_teams, params):
    team_idx = {t: i for i, t in enumerate(valid_teams)}
    n = len(valid_teams)
    attack, defense = params[:n], params[n:2*n]
    rho = params[2*n + 1]
    ai, bi = team_idx[team_a], team_idx[team_b]
    lam = np.exp(attack[ai] - defense[bi])
    mu  = np.exp(attack[bi] - defense[ai])
    sm = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    for x in range(MAX_GOALS + 1):
        for y in range(MAX_GOALS + 1):
            sm[x, y] = max(_rho_correction(x, y, lam, mu, rho)
                           * poisson.pmf(x, lam) * poisson.pmf(y, mu), 0)
    sm /= sm.sum()
    win_h = float(np.tril(sm, -1).sum())
    drw   = float(np.trace(sm))
    win_a = float(np.triu(sm, 1).sum())
    return win_h, drw, win_a


def tendency(h, a):
    return "H" if h > a else ("A" if h < a else "D")


def bookie_implied(oh, od, oa):
    """Normalise raw implied probs to remove overround. Returns (p_h, p_d, p_a)."""
    raw = np.array([1/oh, 1/od, 1/oa])
    return tuple(raw / raw.sum())


def main():
    print("Loading data …")
    train, wc22 = load_and_split()
    print(f"  Training rows : {len(train):,}  |  WC2022 matches: {len(wc22)}")

    print("Fitting model …")
    valid_teams, params = fit_model(train)

    rows = []
    for _, m in wc22.iterrows():
        h, a = m["home_team"], m["away_team"]
        ah, aa = int(m["home_score"]), int(m["away_score"])
        actual_tend = tendency(ah, aa)

        if h not in valid_teams or a not in valid_teams:
            continue
        if (h, a) not in WC2022_ODDS:
            print(f"  WARNING: no odds for {h} vs {a}")
            continue

        mh, md, ma = get_probs(h, a, valid_teams, params)
        oh, od, oa = WC2022_ODDS[(h, a)]
        bh, bd, ba = bookie_implied(oh, od, oa)
        overround = sum(1/x for x in (oh, od, oa))

        # Model top-pick outcome
        model_probs = {"H": mh, "D": md, "A": ma}
        model_pick  = max(model_probs, key=model_probs.get)

        # Bookmaker favourite
        bookie_probs = {"H": bh, "D": bd, "A": ba}
        bookie_fav   = max(bookie_probs, key=bookie_probs.get)

        # Odds for each outcome
        odds_map = {"H": oh, "D": od, "A": oa}

        rows.append({
            "match":         f"{h} vs {a}",
            "date":          m["date"].date(),
            "actual":        f"{ah}-{aa}",
            "result":        actual_tend,
            # model
            "m_H": mh, "m_D": md, "m_A": ma,
            "model_pick":    model_pick,
            "model_correct": model_pick == actual_tend,
            # bookmaker
            "b_H": bh, "b_D": bd, "b_A": ba,
            "bookie_fav":    bookie_fav,
            "bookie_correct": bookie_fav == actual_tend,
            "overround":     overround,
            # odds
            "o_H": oh, "o_D": od, "o_A": oa,
            # value flags (model > bookie implied)
            "value_H": mh > bh,
            "value_D": md > bd,
            "value_A": ma > ba,
        })

    df = pd.DataFrame(rows)
    n = len(df)

    # ── Strategy 1: model top-pick ────────────────────────────────────────────
    # 1 unit staked per game; collect odds if correct, lose 1 if not
    df["tp_stake"]  = 1.0
    df["tp_return"] = df.apply(
        lambda r: r[f"o_{r['model_pick']}"] if r["model_correct"] else 0.0, axis=1
    )
    df["tp_pnl"] = df["tp_return"] - df["tp_stake"]

    # ── Strategy 2: value bets ────────────────────────────────────────────────
    # 1 unit per value outcome (can be 0–3 per game)
    value_rows = []
    for _, r in df.iterrows():
        for side in ["H", "D", "A"]:
            if r[f"value_{side}"]:
                value_rows.append({
                    "match":   r["match"],
                    "date":    r["date"],
                    "side":    side,
                    "model_p": r[f"m_{side}"],
                    "bookie_p":r[f"b_{side}"],
                    "edge":    r[f"m_{side}"] - r[f"b_{side}"],
                    "odds":    r[f"o_{side}"],
                    "won":     r["result"] == side,
                })
    vb = pd.DataFrame(value_rows)
    if not vb.empty:
        vb["stake"]  = 1.0
        vb["return"] = vb.apply(lambda r: r["odds"] if r["won"] else 0.0, axis=1)
        vb["pnl"]    = vb["return"] - vb["stake"]

    # ── Strategy 3: bookmaker favourite (baseline comparison) ─────────────────
    df["bk_return"] = df.apply(
        lambda r: r[f"o_{r['bookie_fav']}"] if r["bookie_correct"] else 0.0, axis=1
    )
    df["bk_pnl"] = df["bk_return"] - 1.0

    # ── Per-match detail table ────────────────────────────────────────────────
    print(f"\n{'Match':<32} {'Act':>4}  {'Model':>5}  {'Bookie':>5}  "
          f"{'mH':>5} {'mD':>5} {'mA':>5}  {'bH':>5} {'bD':>5} {'bA':>5}  "
          f"{'Vig':>5}  {'TP P&L':>7}  {'Val?'}")
    print("─" * 130)
    for _, r in df.iterrows():
        val_flags = "".join(s for s in ["H","D","A"] if r[f"value_{s}"])
        print(
            f"{r['match']:<32} {r['actual']:>4}  "
            f"{'✓' if r['model_correct'] else '✗':>5}  "
            f"{'✓' if r['bookie_correct'] else '✗':>5}  "
            f"{r['m_H']:>5.1%} {r['m_D']:>5.1%} {r['m_A']:>5.1%}  "
            f"{r['b_H']:>5.1%} {r['b_D']:>5.1%} {r['b_A']:>5.1%}  "
            f"{r['overround']:>4.1%}  "
            f"{r['tp_pnl']:>+7.2f}  "
            f"{val_flags}"
        )

    # ── Value bet detail ──────────────────────────────────────────────────────
    if not vb.empty:
        print(f"\n\n── Value bets ({len(vb)} bets across {vb['match'].nunique()} games) ──────────────────")
        print(f"{'Match':<32} {'Side':>4}  {'Model':>6}  {'Bookie':>6}  {'Edge':>6}  {'Odds':>5}  {'W?':>3}  {'P&L':>6}")
        print("─" * 80)
        for _, r in vb.iterrows():
            print(f"{r['match']:<32} {r['side']:>4}  {r['model_p']:>6.1%}  {r['bookie_p']:>6.1%}  "
                  f"{r['edge']:>+6.1%}  {r['odds']:>5.2f}  {'✓' if r['won'] else '✗':>3}  {r['pnl']:>+6.2f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    avg_vig = (df["overround"] - 1).mean()

    print("\n\n" + "═" * 65)
    print("SUMMARY")
    print("═" * 65)
    print(f"\n  Games: {n}   |   Avg bookmaker overround: {avg_vig:.1%}")

    print(f"\n  {'Metric':<35} {'Model':>8}  {'Bookie fav':>10}")
    print(f"  {'─'*55}")
    mc = df["model_correct"].mean()
    bc = df["bookie_correct"].mean()
    print(f"  {'Tendency accuracy':<35} {mc:>8.1%}  {bc:>10.1%}")

    # Top-pick P&L
    tp_staked = df["tp_stake"].sum()
    tp_return = df["tp_return"].sum()
    tp_roi    = (tp_return - tp_staked) / tp_staked
    bk_staked = n * 1.0
    bk_return = df["bk_return"].sum()
    bk_roi    = (bk_return - bk_staked) / bk_staked
    print(f"\n  Strategy 1 — Model top-pick (1 unit/game)")
    print(f"    Bets: {n}  |  Staked: {tp_staked:.0f}u  |  "
          f"Return: {tp_return:.2f}u  |  P&L: {tp_return-tp_staked:+.2f}u  |  ROI: {tp_roi:+.1%}")
    print(f"  (Bookmaker-fav baseline: "
          f"Return: {bk_return:.2f}u  |  P&L: {bk_return-bk_staked:+.2f}u  |  ROI: {bk_roi:+.1%})")

    if not vb.empty:
        vb_staked = vb["stake"].sum()
        vb_return = vb["return"].sum()
        vb_pnl    = vb_return - vb_staked
        vb_roi    = vb_pnl / vb_staked
        vb_hitrate = vb["won"].mean()
        print(f"\n  Strategy 2 — Value bets (model_prob > bookie_implied_prob)")
        print(f"    Bets: {len(vb)}  |  Staked: {vb_staked:.0f}u  |  "
              f"Return: {vb_return:.2f}u  |  P&L: {vb_pnl:+.2f}u  |  ROI: {vb_roi:+.1%}")
        print(f"    Hit rate: {vb_hitrate:.1%}  |  Avg edge: {vb['edge'].mean():+.1%}  |  "
              f"Avg odds: {vb['odds'].mean():.2f}")

        # Breakdown by outcome type
        print(f"\n  Value bet breakdown by outcome:")
        print(f"    {'Side':<6}  {'Bets':>5}  {'Won':>5}  {'Hit%':>7}  {'P&L':>8}  {'ROI':>7}")
        for side in ["H", "D", "A"]:
            sub = vb[vb["side"] == side]
            if sub.empty: continue
            sp = sub["pnl"].sum()
            sr = sp / len(sub)
            print(f"    {side:<6}  {len(sub):>5}  {sub['won'].sum():>5}  "
                  f"{sub['won'].mean():>7.1%}  {sp:>+8.2f}  {sr:>+7.1%}")

    # ── Probability calibration: Brier score ──────────────────────────────────
    # Brier score = mean squared error of probabilities vs outcomes
    brier_model  = []
    brier_bookie = []
    logloss_model  = []
    logloss_bookie = []
    for _, r in df.iterrows():
        for side, key in [("H", "result"), ("D", "result"), ("A", "result")]:
            actual_bin = 1.0 if r["result"] == side else 0.0
            mp = r[f"m_{side}"]
            bp = r[f"b_{side}"]
            brier_model.append((mp - actual_bin) ** 2)
            brier_bookie.append((bp - actual_bin) ** 2)
            logloss_model.append(-actual_bin * np.log(max(mp, 1e-10))
                                  - (1 - actual_bin) * np.log(max(1 - mp, 1e-10)))
            logloss_bookie.append(-actual_bin * np.log(max(bp, 1e-10))
                                   - (1 - actual_bin) * np.log(max(1 - bp, 1e-10)))

    bs_m = np.mean(brier_model)
    bs_b = np.mean(brier_bookie)
    ll_m = np.mean(logloss_model)
    ll_b = np.mean(logloss_bookie)
    print(f"\n  Probability calibration (lower = better):")
    print(f"    {'Metric':<15}  {'Model':>8}  {'Bookmaker':>10}")
    print(f"    {'Brier score':<15}  {bs_m:>8.4f}  {bs_b:>10.4f}")
    print(f"    {'Log-loss':<15}  {ll_m:>8.4f}  {ll_b:>10.4f}")

    print("\n  Note: Bookmaker odds are best-available from checkbestodds.com.")
    print("  Overround removed for probability comparison; raw odds used for P&L.")


if __name__ == "__main__":
    main()
