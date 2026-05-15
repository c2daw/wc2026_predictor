"""
Points scoring evaluation: 4 for exact score, 2 for correct GD only, 1 for correct tendency only.
Compares three prediction strategies using model.py (NB Dixon-Coles):
  - argmax   : most probable scoreline (mode of score matrix)
  - round_xg : round(xG_home) – round(xG_away)
  - ep       : scoreline maximising expected points under the 4/2/1 system

Runs for WC2022 (train 2014–2022-11-19) and Euro 2024 (train 2014–2024-06-13).
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from model import fit_model, predict_match, best_scoreline

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 1.5,         "FIFA World Cup qualification": 1.2,
    "UEFA Euro": 1.4,              "UEFA Euro qualification": 1.2,
    "Copa América": 1.4,           "Africa Cup of Nations": 1.3,
    "Africa Cup of Nations qualification": 1.1,
    "Asian Cup": 1.3,              "Asian Cup qualification": 1.1,
    "CONCACAF Gold Cup": 1.2,      "CONCACAF Gold Cup qualification": 1.0,
    "OFC Nations Cup": 1.1,        "Friendly": 0.5,
    "Confederations Cup": 1.3,     "Nations League": 1.1,
    "UEFA Nations League": 1.1,
}
EXCLUDED_PATTERNS = [
    "conifa", "island games", "muratti", "inter games",
    "parish", "inter-insular", "nf-board",
]
DECAY_DAYS = 365 * 3
MAX_GOALS  = 8


def load_all():
    df = pd.read_csv(DATA_URL, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].map({"TRUE": True, "FALSE": False}).fillna(False)
    excl = "|".join(EXCLUDED_PATTERNS)
    df = df[~df["tournament"].str.lower().str.contains(excl, na=False)]
    return df


def make_train(df, cutoff: pd.Timestamp, start="2014-01-01"):
    """Build a training slice with time/tournament weights anchored to cutoff."""
    t = df[(df["date"] >= start) & (df["date"] < cutoff)].copy()
    t["days_ago"] = (cutoff - t["date"]).dt.days
    t["time_weight"] = np.exp(-np.log(2) * t["days_ago"] / DECAY_DAYS)
    t["tourney_weight"] = t["tournament"].map(
        lambda s: next((v for k, v in TOURNAMENT_WEIGHTS.items()
                        if k.lower() in s.lower()), 0.8)
    )
    t["weight"] = t["time_weight"] * t["tourney_weight"]
    return t


def score_pts(ph, pa, ah, aa):
    if ph == ah and pa == aa:                                return 4, "exact"
    if (ph - pa) == (ah - aa):                               return 2, "gd"
    if (ph > pa) == (ah > aa) and (ph == pa) == (ah == aa):  return 1, "tend"
    return 0, "miss"


def run_tournament(df_all, tourney_filter, cutoff, start, end, label,
                   ref_scores=None):
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")

    train = make_train(df_all, cutoff)
    test = df_all[
        df_all["tournament"].str.contains(tourney_filter, case=False, na=False) &
        (df_all["date"] >= start) & (df_all["date"] <= end)
    ].sort_values("date").reset_index(drop=True)
    print(f"  Training rows: {len(train):,}  |  Tournament matches: {len(test)}")
    print("  Fitting model …")

    # fit_model expects the "weight" column already set
    ratings, home_adv, rho, valid_teams, params = fit_model(train)
    alpha = float(np.exp(params[2 * len(valid_teams) + 2]))
    print(f"  alpha={alpha:.2f}  variance/mean at xG=1.5: {1 + 1.5/alpha:.3f}")

    rows = []
    for _, m in test.iterrows():
        h, a = m["home_team"], m["away_team"]
        ah, aa = int(m["home_score"]), int(m["away_score"])
        if h not in valid_teams or a not in valid_teams:
            print(f"  SKIP: {h} vs {a}")
            continue

        result = predict_match(h, a, neutral=True, ratings_df=ratings,
                               home_adv=home_adv, rho=rho,
                               valid_teams=valid_teams, params=params)
        lam, mu = result["lam"], result["mu"]
        sm = result["score_matrix"]

        # Strategy 1: argmax
        r, c = np.unravel_index(np.argmax(sm), sm.shape)
        ph_am, pa_am = int(r), int(c)

        # Strategy 2: round(xG)
        ph_rx = int(min(round(lam), MAX_GOALS))
        pa_rx = int(min(round(mu),  MAX_GOALS))

        # Strategy 3: expected-points optimal
        ph_ep, pa_ep = best_scoreline(result, weights=(4, 2, 1))

        pts_am, cat_am = score_pts(ph_am, pa_am, ah, aa)
        pts_rx, cat_rx = score_pts(ph_rx, pa_rx, ah, aa)
        pts_ep, cat_ep = score_pts(ph_ep, pa_ep, ah, aa)

        rows.append({
            "match":    f"{h} vs {a}",
            "actual":   f"{ah}-{aa}",
            "xG":       f"{lam:.2f}-{mu:.2f}",
            "argmax":   f"{ph_am}-{pa_am}", "pts_am": pts_am, "cat_am": cat_am,
            "round_xg": f"{ph_rx}-{pa_rx}", "pts_rx": pts_rx, "cat_rx": cat_rx,
            "ep":       f"{ph_ep}-{pa_ep}", "pts_ep": pts_ep, "cat_ep": cat_ep,
        })

    df = pd.DataFrame(rows)
    n  = len(df)

    sym = {"exact": "★", "gd": "◆", "tend": "▸", "miss": "·"}
    print(f"\n  {'Match':<32} {'Actual':>6}  {'xG':>9}  "
          f"{'Argmax':>6} {'P':>2}  {'RoundXG':>7} {'P':>2}  {'EP':>6} {'P':>2}")
    print(f"  {'─'*80}")
    for _, r in df.iterrows():
        diff_rx = "+" if r["pts_rx"] > r["pts_am"] else ("-" if r["pts_rx"] < r["pts_am"] else " ")
        diff_ep = "+" if r["pts_ep"] > r["pts_am"] else ("-" if r["pts_ep"] < r["pts_am"] else " ")
        print(
            f"  {r['match']:<32} {r['actual']:>6}  {r['xG']:>9}  "
            f"{r['argmax']:>6} {sym[r['cat_am']]}{r['pts_am']}  "
            f"{r['round_xg']:>7} {diff_rx}{sym[r['cat_rx']]}{r['pts_rx']}  "
            f"{r['ep']:>6} {diff_ep}{sym[r['cat_ep']]}{r['pts_ep']}"
        )

    def summ(cat_col, pts_col):
        return dict(exact=(df[cat_col]=="exact").sum(), gd=(df[cat_col]=="gd").sum(),
                    tend=(df[cat_col]=="tend").sum(), miss=(df[cat_col]=="miss").sum(),
                    total=int(df[pts_col].sum()))

    s_am = summ("cat_am", "pts_am")
    s_rx = summ("cat_rx", "pts_rx")
    s_ep = summ("cat_ep", "pts_ep")

    print(f"\n  {'Strategy':<14} {'Exact':>6}  {'GD':>4}  {'Tend':>5}  {'Miss':>5}  {'Total':>6}  {'pts/g':>6}")
    print(f"  {'─'*52}")
    for lbl, s in [("Argmax", s_am), ("Round(xG)", s_rx), ("EP-optimal", s_ep)]:
        print(f"  {lbl:<14} {s['exact']:>6}  {s['gd']:>4}  {s['tend']:>5}  "
              f"{s['miss']:>5}  {s['total']:>6}  {s['total']/n:>6.2f}")

    if ref_scores:
        print(f"\n  Reference players:")
        for i, (e, g, t, tot) in enumerate(ref_scores, 1):
            print(f"    #{i}: {tot} pts  ({e} exact / {g} gd / {t} tend)")
        all_scores = sorted([r[3] for r in ref_scores] + [s_ep["total"]], reverse=True)
        rank = all_scores.index(s_ep["total"]) + 1
        print(f"    EP model: {s_ep['total']} pts → rank #{rank}/{len(all_scores)}")

    return s_am, s_rx, s_ep, n


def main():
    print("Loading data …")
    df_all = load_all()
    print(f"Total rows: {len(df_all):,}")

    EURO_REF = [(13,4,9,69),(8,6,9,53),(7,1,13,43),(6,6,12,48)]

    am_wc, rx_wc, ep_wc, n_wc = run_tournament(
        df_all, "FIFA World Cup",
        pd.Timestamp("2022-11-20"), pd.Timestamp("2022-11-20"), pd.Timestamp("2022-12-18"),
        "FIFA World Cup 2022  (NB model, train: 2014–2022-11-19)",
    )

    am_eu, rx_eu, ep_eu, n_eu = run_tournament(
        df_all, "UEFA Euro$",
        pd.Timestamp("2024-06-14"), pd.Timestamp("2024-06-14"), pd.Timestamp("2024-07-14"),
        "UEFA Euro 2024  (NB model, train: 2014–2024-06-13)",
        ref_scores=EURO_REF,
    )

    print(f"\n\n{'='*72}")
    print("  GRAND SUMMARY  (NB vs previous Poisson baseline in brackets)")
    print(f"{'='*72}")
    poisson_baseline = {"argmax": (67,43), "round_xg": (44,68), "ep": (54,45)}
    print(f"\n  {'Strategy':<14}  {'WC2022':>8}  {'Euro2024':>9}  {'Combined':>9}")
    print(f"  {'─'*48}")
    for lbl, wc, eu, key in [
        ("Argmax",     am_wc, am_eu, "argmax"),
        ("Round(xG)",  rx_wc, rx_eu, "round_xg"),
        ("EP-optimal", ep_wc, ep_eu, "ep"),
    ]:
        comb = wc["total"] + eu["total"]
        p_wc, p_eu = poisson_baseline[key]
        p_comb = p_wc + p_eu
        print(f"  {lbl:<14}  "
              f"{wc['total']:>4} ({wc['total']-p_wc:+d})  "
              f"{eu['total']:>5} ({eu['total']-p_eu:+d})  "
              f"{comb:>5} ({comb-p_comb:+d})")

    print(f"\n  (+/−) = change vs Poisson model")


if __name__ == "__main__":
    main()
