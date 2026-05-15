"""
Dixon-Coles Poisson model for international football.
Separate attack/defense ratings per team, home advantage, and a low-score
correction factor (rho) for 0-0/0-1/1-0/1-1 scorelines.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from datetime import date
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

DECAY_DAYS = 365 * 3  # half-life ~3 years
MIN_MATCHES = 10
LAMBDA_REG = 0.02     # L2 regularization on attack/defense params

EXCLUDED_TOURNAMENT_PATTERNS = [
    "conifa", "island games", "muratti", "inter games",
    "parish", "inter-insular", "nf-board",
]


def load_data(min_date="2010-01-01") -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, parse_dates=["date"])
    df = df[df["date"] >= pd.Timestamp(min_date)].copy()
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].map({"TRUE": True, "FALSE": False}).fillna(False)
    df["days_ago"] = (pd.Timestamp(date.today()) - df["date"]).dt.days
    df["time_weight"] = np.exp(-np.log(2) * df["days_ago"] / DECAY_DAYS)
    excl_pattern = "|".join(EXCLUDED_TOURNAMENT_PATTERNS)
    df = df[~df["tournament"].str.lower().str.contains(excl_pattern, na=False)]
    df["tourney_weight"] = df["tournament"].map(
        lambda t: next((v for k, v in TOURNAMENT_WEIGHTS.items() if k.lower() in t.lower()), 0.8)
    )
    df["weight"] = df["time_weight"] * df["tourney_weight"]
    return df


def _rho_correction(x, y, lam, mu, rho):
    """Dixon-Coles low-score correction factor."""
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    elif x == 0 and y == 1:
        return 1 + lam * rho
    elif x == 1 and y == 0:
        return 1 + mu * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def _neg_log_likelihood(params, teams, arrays):
    n = len(teams)
    attack   = params[:n]
    defense  = params[n:2*n]
    home_adv = params[2*n]
    rho      = params[2*n + 1]

    hi_idx, ai_idx, x, y, weights, ha_vec = arrays
    lam = np.exp(attack[hi_idx] - defense[ai_idx] + ha_vec * home_adv)
    mu  = np.exp(attack[ai_idx] - defense[hi_idx])

    log_pmf_x = x * np.log(np.maximum(lam, 1e-10)) - lam - gammaln(x + 1)
    log_pmf_y = y * np.log(np.maximum(mu,  1e-10)) - mu  - gammaln(y + 1)

    rho_c = np.ones(len(x))
    m00 = (x == 0) & (y == 0); rho_c[m00] = 1 - lam[m00] * mu[m00] * rho
    m01 = (x == 0) & (y == 1); rho_c[m01] = 1 + lam[m01] * rho
    m10 = (x == 1) & (y == 0); rho_c[m10] = 1 + mu[m10] * rho
    m11 = (x == 1) & (y == 1); rho_c[m11] = 1 - rho
    rho_c = np.maximum(rho_c, 1e-10)

    ll = weights * (np.log(rho_c) + log_pmf_x + log_pmf_y)
    reg = LAMBDA_REG * (np.sum(attack**2) + np.sum(defense**2))
    return -ll.sum() + reg


def fit_model(df: pd.DataFrame):
    all_teams = pd.concat([df["home_team"], df["away_team"]])
    team_counts = all_teams.value_counts()
    valid_teams = sorted(team_counts[team_counts >= MIN_MATCHES].index.tolist())
    team_idx = {t: i for i, t in enumerate(valid_teams)}

    df_fit = df[df["home_team"].isin(team_idx) & df["away_team"].isin(team_idx)].copy()

    hi_idx  = df_fit["home_team"].map(team_idx).to_numpy(dtype=np.int32)
    ai_idx  = df_fit["away_team"].map(team_idx).to_numpy(dtype=np.int32)
    x       = df_fit["home_score"].to_numpy(dtype=np.float64)
    y       = df_fit["away_score"].to_numpy(dtype=np.float64)
    weights = df_fit["weight"].to_numpy(dtype=np.float64)
    ha_vec  = (~df_fit["neutral"]).to_numpy(dtype=np.float64)

    arrays = (hi_idx, ai_idx, x, y, weights, ha_vec)

    n = len(valid_teams)
    x0 = np.zeros(2 * n + 2)
    x0[2 * n]     = 0.3    # home_adv
    x0[2 * n + 1] = -0.1   # rho

    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(valid_teams, arrays),
        method="SLSQP",
        constraints=[{"type": "eq", "fun": lambda p: np.sum(p[:n])}],
        options={"maxiter": 500, "ftol": 1e-6},
    )

    params = result.x
    attack   = params[:n]
    defense  = params[n:2*n]
    home_adv = params[2*n]
    rho      = params[2*n + 1]

    ratings = pd.DataFrame({"team": valid_teams, "attack": attack, "defense": defense})
    ratings["spi"] = (
        (attack - attack.mean()) / (attack.std() + 1e-9) * 0.5
        + (defense - defense.mean()) / (defense.std() + 1e-9) * 0.5
    )
    ratings["spi"] = 50 + 10 * ratings["spi"]
    ratings = ratings.sort_values("spi", ascending=False).reset_index(drop=True)
    ratings.index += 1

    return ratings, home_adv, rho, valid_teams, params


def predict_match(team_a: str, team_b: str, neutral: bool,
                  ratings_df: pd.DataFrame, home_adv: float, rho: float,
                  valid_teams: list, params: np.ndarray,
                  max_goals: int = 8):
    """Return score probability matrix and match outcome probabilities."""
    team_idx = {t: i for i, t in enumerate(valid_teams)}
    n = len(valid_teams)
    attack  = params[:n]
    defense = params[n:2*n]

    if team_a not in team_idx or team_b not in team_idx:
        missing = [t for t in [team_a, team_b] if t not in team_idx]
        raise ValueError(f"Team(s) not in model: {missing}")

    ai, bi = team_idx[team_a], team_idx[team_b]
    ha = 0.0 if neutral else home_adv

    lam = np.exp(attack[ai] - defense[bi] + ha)
    mu  = np.exp(attack[bi] - defense[ai])

    score_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            rc = _rho_correction(x, y, lam, mu, rho)
            score_matrix[x, y] = rc * np.exp(
                x * np.log(max(lam, 1e-10)) - lam - gammaln(x + 1)
                + y * np.log(max(mu,  1e-10)) - mu  - gammaln(y + 1)
            )

    score_matrix = np.clip(score_matrix, 0, None)
    score_matrix /= score_matrix.sum()

    win_a = np.tril(score_matrix, -1).sum()
    draw  = np.trace(score_matrix)
    win_b = np.triu(score_matrix, 1).sum()

    return {
        "team_a": team_a,
        "team_b": team_b,
        "lam": lam,
        "mu": mu,
        "win_a": win_a,
        "draw": draw,
        "win_b": win_b,
        "score_matrix": score_matrix,
        "max_goals": max_goals,
    }


WC2026_GROUPS = {
    "A": ["Mexico", "South Korea", "South Africa", "Czech Republic"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curaçao"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Panama", "Ghana"],
}


def _sample_score(score_matrix: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    flat = score_matrix.ravel()
    idx = rng.choice(len(flat), p=flat)
    return divmod(idx, score_matrix.shape[1])


def simulate_group(
    group_teams: list[str],
    valid_teams: list,
    params: np.ndarray,
    home_adv: float,
    rho: float,
    n_sims: int = 20_000,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Monte-Carlo simulate a WC group stage (all neutral, round-robin).
    Returns a DataFrame with expected points, GD, GF, and finish-position probabilities.
    Unknown/TBD teams get average (0,0) attack/defense.
    """
    rng = np.random.default_rng(rng_seed)
    n_teams = len(group_teams)
    team_idx_map = {t: i for i, t in enumerate(valid_teams)}
    n_model = len(valid_teams)
    attack_all  = params[:n_model]
    defense_all = params[n_model:2*n_model]

    attack_g  = np.array([attack_all[team_idx_map[t]]  if t in team_idx_map else 0.0 for t in group_teams])
    defense_g = np.array([defense_all[team_idx_map[t]] if t in team_idx_map else 0.0 for t in group_teams])

    MAX_G = 8
    matchups = [(i, j) for i in range(n_teams) for j in range(i+1, n_teams)]
    score_matrices = {}
    for (i, j) in matchups:
        lam = np.exp(attack_g[i] - defense_g[j])
        mu  = np.exp(attack_g[j] - defense_g[i])
        sm  = np.zeros((MAX_G+1, MAX_G+1))
        for x in range(MAX_G+1):
            for y in range(MAX_G+1):
                rc = _rho_correction(x, y, lam, mu, rho)
                sm[x, y] = max(rc * np.exp(
                    x * np.log(max(lam, 1e-10)) - lam - gammaln(x + 1)
                    + y * np.log(max(mu,  1e-10)) - mu  - gammaln(y + 1)
                ), 0)
        sm /= sm.sum()
        score_matrices[(i, j)] = sm

    finish_counts = np.zeros((n_teams, n_teams), dtype=np.int64)
    total_pts = np.zeros(n_teams)
    total_gf  = np.zeros(n_teams)
    total_gd  = np.zeros(n_teams)

    for _ in range(n_sims):
        pts = np.zeros(n_teams, dtype=np.int32)
        gf  = np.zeros(n_teams, dtype=np.int32)
        ga  = np.zeros(n_teams, dtype=np.int32)

        for (i, j) in matchups:
            g_i, g_j = _sample_score(score_matrices[(i, j)], rng)
            gf[i] += g_i; ga[i] += g_j
            gf[j] += g_j; ga[j] += g_i
            if g_i > g_j:   pts[i] += 3
            elif g_i < g_j: pts[j] += 3
            else:            pts[i] += 1; pts[j] += 1

        gd = gf - ga
        noise = rng.random(n_teams) * 0.001
        order = np.lexsort((noise, gf, gd, pts))[::-1]
        for pos, team in enumerate(order):
            finish_counts[team, pos] += 1

        total_pts += pts
        total_gf  += gf
        total_gd  += gd

    rows = []
    for i, team in enumerate(group_teams):
        rows.append({
            "Team":     team,
            "xPts":     round(total_pts[i] / n_sims, 2),
            "xGD":      round(total_gd[i]  / n_sims, 2),
            "xGF":      round(total_gf[i]  / n_sims, 2),
            "P(1st)":   round(finish_counts[i, 0] / n_sims * 100, 1),
            "P(2nd)":   round(finish_counts[i, 1] / n_sims * 100, 1),
            "P(3rd)":   round(finish_counts[i, 2] / n_sims * 100, 1),
            "P(4th)":   round(finish_counts[i, 3] / n_sims * 100, 1),
        })

    df = pd.DataFrame(rows).sort_values("xPts", ascending=False).reset_index(drop=True)
    df.index += 1
    return df


def best_scoreline(result: dict, weights: tuple = (4, 2, 1)) -> tuple:
    """
    Return the scoreline (h, a) that maximises expected points under a tiered
    scoring system where weights = (exact, goal_difference, tendency).

    E[pts | predict (h,a)] =
        w_tend  × P(correct tendency)
      + (w_gd   - w_tend)  × P(GD = h−a)
      + (w_exact - w_gd)   × P(score = h,a)
    """
    w_exact, w_gd, w_tend = weights
    sm = result["score_matrix"]
    n = sm.shape[0]

    tend_h = float(np.tril(sm, -1).sum())
    tend_d = float(np.trace(sm))
    tend_a = float(np.triu(sm, 1).sum())
    gd_prob = {d: float(np.trace(sm, offset=-d)) for d in range(-(n - 1), n)}

    ep = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = i - j
            t = tend_h if d > 0 else (tend_d if d == 0 else tend_a)
            ep[i, j] = (w_tend * t
                        + (w_gd - w_tend) * gd_prob[d]
                        + (w_exact - w_gd) * sm[i, j])

    r, c = np.unravel_index(np.argmax(ep), ep.shape)
    return int(r), int(c)


def _sim_group_once(teams, team_idx_map, attack_all, defense_all, rng):
    """Fast single group simulation using Poisson sampling (no rho correction)."""
    n = len(teams)
    pts = np.zeros(n, dtype=np.int32)
    gf  = np.zeros(n, dtype=np.int32)
    ga  = np.zeros(n, dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            ai = team_idx_map.get(teams[i], -1)
            bi = team_idx_map.get(teams[j], -1)
            atk_i = attack_all[ai]  if ai >= 0 else 0.0
            def_i = defense_all[ai] if ai >= 0 else 0.0
            atk_j = attack_all[bi]  if bi >= 0 else 0.0
            def_j = defense_all[bi] if bi >= 0 else 0.0
            lam = np.exp(atk_i - def_j)
            mu  = np.exp(atk_j - def_i)
            g_i = rng.poisson(lam); g_j = rng.poisson(mu)
            gf[i] += g_i; ga[i] += g_j
            gf[j] += g_j; ga[j] += g_i
            if g_i > g_j:   pts[i] += 3
            elif g_i < g_j: pts[j] += 3
            else:            pts[i] += 1; pts[j] += 1
    gd    = gf - ga
    noise = rng.random(n) * 0.001
    order = np.lexsort((noise, gf.astype(float), gd.astype(float), pts.astype(float)))[::-1]
    return [teams[k] for k in order]


def _sim_ko_match(h, a, team_idx_map, attack_all, defense_all, rng):
    """Sample a KO match winner. Draw → penalty coin-flip."""
    ai = team_idx_map.get(h, -1); bi = team_idx_map.get(a, -1)
    lam = np.exp((attack_all[ai]  if ai >= 0 else 0.0) - (defense_all[bi] if bi >= 0 else 0.0))
    mu  = np.exp((attack_all[bi]  if bi >= 0 else 0.0) - (defense_all[ai] if ai >= 0 else 0.0))
    g_h = rng.poisson(lam); g_a = rng.poisson(mu)
    if g_h > g_a: return h
    if g_h < g_a: return a
    return h if rng.random() < 0.5 else a


def simulate_tournament(
    wc_groups: dict,
    valid_teams: list,
    params: np.ndarray,
    home_adv: float,
    rho: float,
    n_sims: int = 8_000,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Monte-Carlo simulate full WC2026 (group stage → R32 → R16 → QF → SF → Final).
    Top 2 from each group + 8 best 3rd-place advance to R32 (32 teams total).
    KO bracket uses random seeding within halves (approximation).
    Returns DataFrame with columns: Team, P(R32), P(R16), P(QF), P(SF), P(Final), P(Win).
    """
    rng        = np.random.default_rng(rng_seed)
    n_model    = len(valid_teams)
    tidx       = {t: i for i, t in enumerate(valid_teams)}
    atk        = params[:n_model]
    dfn        = params[n_model:2 * n_model]
    letters    = list(wc_groups.keys())
    all_teams  = [t for g in wc_groups.values() for t in g]
    rounds     = ["R32", "R16", "QF", "SF", "Final", "Win"]
    counts     = {t: {r: 0 for r in rounds} for t in all_teams}

    for _ in range(n_sims):
        finishes   = {l: _sim_group_once(wc_groups[l], tidx, atk, dfn, rng) for l in letters}
        group_pts  = {}  # track 3rd-place teams' "pts proxy" for best-3rd ranking
        thirds     = []

        qualifiers = []
        for l in letters:
            qualifiers.append(finishes[l][0])  # 1st
            qualifiers.append(finishes[l][1])  # 2nd
            if len(finishes[l]) > 2:
                thirds.append(finishes[l][2])

        # Best 8 3rd-place teams (use random for approximation; could rank by expected pts)
        rng.shuffle(thirds)
        qualifiers += thirds[:8]  # 24 + 8 = 32

        for t in qualifiers:
            counts[t]["R32"] += 1

        # KO rounds: R32(32→16) R16(16→8) QF(8→4) SF(4→2) Final(2→1)
        pool = qualifiers[:]
        for rnd in ["R16", "QF", "SF", "Final", "Win"]:
            rng.shuffle(pool)
            next_pool = []
            for k in range(0, len(pool) - 1, 2):
                w = _sim_ko_match(pool[k], pool[k + 1], tidx, atk, dfn, rng)
                next_pool.append(w)
            pool = next_pool
            for t in pool:
                counts[t][rnd] += 1

    rows = []
    for t in all_teams:
        rows.append({"Team": t, **{r: counts[t][r] / n_sims for r in rounds}})
    return pd.DataFrame(rows).sort_values("Win", ascending=False).reset_index(drop=True)


def top_scorelines(result: dict, top_n: int = 10) -> pd.DataFrame:
    sm = result["score_matrix"]
    mg = result["max_goals"]
    rows = []
    for x in range(mg + 1):
        for y in range(mg + 1):
            rows.append((x, y, sm[x, y]))
    df = pd.DataFrame(rows, columns=["goals_a", "goals_b", "prob"])
    df["scoreline"] = df["goals_a"].astype(str) + " - " + df["goals_b"].astype(str)
    return df.sort_values("prob", ascending=False).head(top_n).reset_index(drop=True)
