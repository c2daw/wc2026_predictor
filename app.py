"""
World Cup 2026 SPI Predictor — Streamlit app
Dixon-Coles Poisson model · time-decay & tournament weighting
"""
import json
import os
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import gammaln

from model import (
    load_data, fit_model, predict_match, top_scorelines,
    simulate_group, simulate_tournament, WC2026_GROUPS,
)

st.set_page_config(page_title="WC 2026 Predictor", page_icon="⚽", layout="wide")

# ── Hardcoded settings (no sidebar) ───────────────────────────────────────────
MIN_DATE = "2014-01-01"
NEUTRAL  = True

# ── Official WC 2026 color palette ────────────────────────────────────────────
NAVY  = "#19237C"
LIME  = "#AFEA00"
RED   = "#D60000"
BLUE2 = "#304FFF"
GREEN = "#00C651"

# ── Team metadata: flag emoji + 3-letter code ─────────────────────────────────
TEAM_META = {
    "Mexico":                 ("🇲🇽", "MEX"),
    "South Korea":            ("🇰🇷", "KOR"),
    "South Africa":           ("🇿🇦", "RSA"),
    "Czech Republic":         ("🇨🇿", "CZE"),
    "Canada":                 ("🇨🇦", "CAN"),
    "Switzerland":            ("🇨🇭", "SUI"),
    "Qatar":                  ("🇶🇦", "QAT"),
    "Bosnia & H.":            ("🇧🇦", "BIH"),
    "Bosnia and Herzegovina": ("🇧🇦", "BIH"),
    "Brazil":                 ("🇧🇷", "BRA"),
    "Morocco":                ("🇲🇦", "MAR"),
    "Scotland":               ("🏴󠁧󠁢󠁳󠁣󠁴󠁿", "SCO"),
    "Haiti":                  ("🇭🇹", "HAI"),
    "United States":          ("🇺🇸", "USA"),
    "Paraguay":               ("🇵🇾", "PAR"),
    "Australia":              ("🇦🇺", "AUS"),
    "Turkey":                 ("🇹🇷", "TUR"),
    "Germany":                ("🇩🇪", "GER"),
    "Ecuador":                ("🇪🇨", "ECU"),
    "Ivory Coast":            ("🇨🇮", "CIV"),
    "Curaçao":                ("🇨🇼", "CUW"),
    "Netherlands":            ("🇳🇱", "NED"),
    "Japan":                  ("🇯🇵", "JPN"),
    "Tunisia":                ("🇹🇳", "TUN"),
    "Sweden":                 ("🇸🇪", "SWE"),
    "Belgium":                ("🇧🇪", "BEL"),
    "Iran":                   ("🇮🇷", "IRN"),
    "Egypt":                  ("🇪🇬", "EGY"),
    "New Zealand":            ("🇳🇿", "NZL"),
    "Spain":                  ("🇪🇸", "ESP"),
    "Uruguay":                ("🇺🇾", "URU"),
    "Saudi Arabia":           ("🇸🇦", "KSA"),
    "Cape Verde":             ("🇨🇻", "CPV"),
    "France":                 ("🇫🇷", "FRA"),
    "Senegal":                ("🇸🇳", "SEN"),
    "Norway":                 ("🇳🇴", "NOR"),
    "Iraq":                   ("🇮🇶", "IRQ"),
    "Argentina":              ("🇦🇷", "ARG"),
    "Austria":                ("🇦🇹", "AUT"),
    "Algeria":                ("🇩🇿", "ALG"),
    "Jordan":                 ("🇯🇴", "JOR"),
    "Portugal":               ("🇵🇹", "POR"),
    "Colombia":               ("🇨🇴", "COL"),
    "Uzbekistan":             ("🇺🇿", "UZB"),
    "DR Congo":               ("🇨🇩", "COD"),
    "England":                ("🏴󠁧󠁢󠁥󠁮󠁧󠁿", "ENG"),
    "Croatia":                ("🇭🇷", "CRO"),
    "Panama":                 ("🇵🇦", "PAN"),
    "Ghana":                  ("🇬🇭", "GHA"),
}

NAME_SHORT = {
    "Bosnia and Herzegovina": "Bosnia & H.",
}

def tmeta(t):
    return TEAM_META.get(t, ("🌍", t[:3].upper()))

def tshort(t):
    return NAME_SHORT.get(t, t)

def tdisplay(t):
    flag, _ = tmeta(t)
    return f"{flag} {t}"

# ── CSS / Font injection ───────────────────────────────────────────────────────
_FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=Barlow+Condensed:wght@600;700;900&"
    "family=Noto+Sans:wght@400;500;600;700&display=swap"
)

_CSS = """
html,body,[class*="css"]{font-family:'Noto Sans',sans-serif!important}
.block-container{padding-top:3rem!important;padding-bottom:2rem!important;max-width:1440px!important}

/* ── Selectbox ───────────────────────────────────────────────────────────── */
[data-testid="stWidgetLabel"]{
  margin-bottom:0!important;padding-bottom:0!important;line-height:1.1!important}
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] *{
  font-family:'Barlow Condensed',sans-serif!important;font-weight:700!important;
  font-size:9px!important;letter-spacing:1px!important;text-transform:uppercase!important;
  color:#9CA3AF!important;margin-bottom:0!important;padding-bottom:0!important}
[data-testid="stSelectbox"],[data-testid="stSelectbox"]>div{gap:1px!important}
div[data-baseweb="select"]>div{
  min-height:30px!important;border-radius:4px!important;
  border-color:#E5E7EB!important;background:#F9FAFB!important;
  display:flex!important;align-items:center!important}
div[data-baseweb="select"]>div:hover{border-color:#19237C!important}
div[data-baseweb="select"] *{
  font-family:'Barlow Condensed',sans-serif!important;font-weight:600!important;
  font-size:13px!important;color:#1F2937!important}
div[data-baseweb="select"] svg{width:14px!important;height:14px!important;color:#9CA3AF!important}
div[data-baseweb="popover"] li,
div[data-baseweb="popover"] li *{
  font-family:'Barlow Condensed',sans-serif!important;font-size:13px!important;
  font-weight:600!important;padding-top:4px!important;padding-bottom:4px!important}

/* ── Segmented control (Highlight) — default Streamlit styling ────────── */

/* ── Tabs ────────────────────────────────────────────────────────────────── */
[data-testid="stTabBar"]{
  gap:3px!important;padding:4px 4px 0!important;
  background:#19237C!important;border-radius:6px 6px 0 0!important}
button[data-baseweb="tab"],
button[data-baseweb="tab"] *{
  font-family:'Barlow Condensed',sans-serif!important;font-weight:700!important;
  font-size:12px!important;letter-spacing:1px!important;text-transform:uppercase!important}
button[data-baseweb="tab"]{
  background:#19237C!important;border:none!important;outline:none!important;
  color:#AFEA00!important;padding:7px 18px!important;
  border-radius:4px 4px 0 0!important}
button[data-baseweb="tab"]:hover{
  background:#0F1650!important;color:#AFEA00!important}
button[aria-selected="true"][data-baseweb="tab"]{
  background:#AFEA00!important;color:#19237C!important;opacity:1!important}
[data-baseweb="tab-highlight"]{display:none!important}
[data-baseweb="tab-border"]{background:#AFEA00!important;height:2px!important}

/* ── App header ─────────────────────────────────────────────────────────── */
.app-header{background:#19237C;padding:10px 20px;display:flex;align-items:center;
  justify-content:space-between;border-bottom:3px solid #AFEA00;border-radius:6px;margin-bottom:16px}
.app-logo{font-family:'Barlow Condensed',sans-serif;font-weight:900;font-size:20px;
  color:#AFEA00;letter-spacing:2px;text-transform:uppercase}
.app-logo span{color:#fff;font-weight:400}
.app-sub{font-size:11px;color:rgba(255,255,255,.35);font-weight:400}

/* Panels */
.panel{background:white;border:.5px solid #E5E7EB;border-radius:6px;overflow:hidden;margin-bottom:10px}
.ph{background:#19237C;padding:8px 14px;display:flex;align-items:center;justify-content:space-between;
  font-family:'Barlow Condensed',sans-serif!important;font-weight:700;font-size:12px;
  letter-spacing:1.5px;text-transform:uppercase;color:#AFEA00}
.ph .ph-sub{color:rgba(255,255,255,.35);font-size:10px;font-weight:400;letter-spacing:0;text-transform:none}
.pb{padding:12px 14px}

/* Teams bar */
.teams-bar{display:flex;align-items:center;justify-content:space-between;
  margin:8px 0;padding:12px 16px;background:#19237C;border-radius:4px}
.tbadge-home{flex:1;text-align:right}
.tbadge-away{flex:1;text-align:left}
.t-flag-code{font-family:'Barlow Condensed',sans-serif;font-weight:700;
  font-size:32px;color:#fff;letter-spacing:1px}
.vs-pill{flex:0 0 56px;text-align:center;font-family:'Barlow Condensed',sans-serif;
  font-weight:900;font-size:26px;color:#AFEA00}

/* Outcome pills */
.outcome-row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:10px}
.outcome-pill{padding:10px 8px;border-radius:4px;text-align:center;
  font-family:'Barlow Condensed',sans-serif!important}
.outcome-pill .pct{font-size:24px;font-weight:900;display:block}
.outcome-pill .lbl{font-size:10px;font-weight:600;letter-spacing:.8px;
  text-transform:uppercase;display:block;margin-top:2px;opacity:.85}
.pill-a-str{background:#D60000;color:#FFD0D0}
.pill-b-str{background:#19237C;color:#C0D8FF}
.pill-a-fnt{background:rgba(214,0,0,.1);color:#D60000}
.pill-b-fnt{background:rgba(25,35,124,.1);color:#19237C}
.pill-draw{background:rgba(131,39,128,.12);color:#832780}

/* xG stat */
.stat-ttl{font-size:10px;font-weight:600;letter-spacing:.8px;text-transform:uppercase;
  color:#9CA3AF;margin-bottom:8px}
.xg-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:3px}
.xg-lbl{font-size:11px;color:#6B7280;font-weight:500}
.xg-val{font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:18px;color:#19237C}
.xg-track{height:4px;background:#E5E7EB;border-radius:2px;margin:3px 0 8px;position:relative}
.xg-fill{position:absolute;left:0;top:0;height:4px;border-radius:2px}
.sep{height:1px;background:#E5E7EB;margin:10px 0}
.xp-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.xp-box{text-align:center;border-radius:4px;padding:10px 6px}
.xp-box .xp-lbl{font-size:10px;text-transform:uppercase;letter-spacing:.6px;margin-bottom:4px}
.xp-box .xp-val{font-family:'Barlow Condensed',sans-serif;font-weight:900;font-size:28px}
.xp-box .xp-sub{font-size:9px;color:#9CA3AF}

/* Score list */
.score-row{display:flex;justify-content:space-between;align-items:center;
  padding:5px 9px;border-radius:3px;margin-bottom:3px}
.score-lbl{font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:14px;color:#1A1A2E}
.score-pct{font-size:11px;color:#6B7280}

/* Group cards */
.grp-card{background:white;border:.5px solid #E5E7EB;border-bottom:none;border-radius:6px 6px 0 0;overflow:hidden;
  margin-bottom:0;box-sizing:border-box;width:100%}
.grp-hdr{background:#19237C;padding:7px 12px;display:flex;align-items:center;justify-content:space-between;min-height:36px;box-sizing:border-box}
[data-testid="element-container"]:has(.grp-card){
  padding:0!important;margin:0!important}
[data-testid="element-container"]:has(.grp-card)+[data-testid="element-container"]{
  margin-top:-8px!important;padding:0!important}
[data-testid="element-container"]:has(.grp-card)+[data-testid="element-container"] .stPlotlyChart{
  border:.5px solid #E5E7EB;border-top:none;border-radius:0 0 6px 6px;
  overflow:hidden;margin-bottom:10px;box-sizing:border-box;width:100%!important}
.grp-letter{font-family:'Barlow Condensed',sans-serif;font-weight:900;font-size:16px;
  color:#AFEA00;letter-spacing:1px}
.grp-sub{font-size:10px;color:rgba(255,255,255,.4);letter-spacing:.5px;text-transform:uppercase;
  font-family:'Barlow Condensed',sans-serif;font-weight:600}
.grp-legend{display:flex;align-items:center;gap:16px;margin-top:10px;padding:6px 12px;
  background:#F9FAFB;border-radius:4px;font-size:11px;color:#6B7280}
.lq{display:inline-block;width:10px;height:10px;border-left:3px solid #AFEA00;margin-right:4px}
.l3a{display:inline-block;width:10px;height:10px;border-left:3px solid #D1FAE5;margin-right:4px}
.l3{display:inline-block;width:10px;height:10px;border-left:3px solid #9CA3AF;margin-right:4px}

/* Bracket */
.bkt-wrap{overflow-x:auto;padding:4px 0}
.bkt-inner{display:flex;gap:0;align-items:stretch;min-width:1100px}
.rnd-col{display:flex;flex-direction:column;min-width:136px;flex:1}
.rnd-lbl{font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:10px;
  letter-spacing:1.2px;text-transform:uppercase;color:#9CA3AF;text-align:center;
  padding:0 0 7px;border-bottom:.5px solid #E5E7EB;margin:0 4px 7px}
.rnd-matches{flex:1;display:flex;flex-direction:column;justify-content:space-evenly;
  gap:5px;padding:0 4px}
.mc{background:white;border:.5px solid #E5E7EB;border-radius:4px;overflow:hidden}
.mt{display:flex;align-items:center;gap:4px;padding:4px 8px;
  font-size:11px;font-family:'Barlow Condensed',sans-serif;font-weight:600;
  letter-spacing:.3px;color:#4B5563;text-transform:uppercase;
  border-bottom:.5px solid #E5E7EB}
.mt:last-child{border-bottom:none}
.mt-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mt-xg{width:18px;text-align:center;font-size:11px;font-weight:700;flex-shrink:0}
.mt .prob{width:30px;text-align:right;font-size:10px;font-weight:400;color:#9CA3AF;flex-shrink:0}
.mt.win{background:#19237C;color:#AFEA00}
.mt.win .prob{color:rgba(175,234,0,.7)}
.fin-col{display:flex;flex-direction:column;min-width:140px;flex:1;
  justify-content:center;align-items:stretch;padding:0 4px}
.fin-lbl{font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:10px;
  letter-spacing:1.2px;text-transform:uppercase;color:#AFEA00;text-align:center;
  padding:0 0 7px;border-bottom:.5px solid #AFEA00;margin-bottom:7px}
.trophy-ban{background:#19237C;text-align:center;padding:6px;
  font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:10px;
  letter-spacing:1.5px;text-transform:uppercase;color:#AFEA00;border-radius:3px 3px 0 0}
.fin-mc{border:1.5px solid #AFEA00;border-radius:0 0 4px 4px;overflow:hidden;background:white}
.fin-mc .mt{font-size:13px;padding:6px 10px}
.conn{width:8px;flex-shrink:0}
.bkt-legend{display:flex;gap:16px;align-items:center;margin-top:10px;font-size:10px;color:#6B7280}
.bl-item{display:flex;align-items:center;gap:6px}
.mc{cursor:default}
#mx-popup{display:none;position:fixed;z-index:9999;background:#fff;
  border:.5px solid #E5E7EB;border-radius:8px;
  box-shadow:0 6px 24px rgba(0,0,0,.14);padding:13px;pointer-events:none}
#mx-popup .pop-header{font-family:'Barlow Condensed',sans-serif;font-weight:700;
  font-size:12px;letter-spacing:.5px;text-transform:uppercase;color:#19237C;
  margin-bottom:8px;text-align:center}
#mx-popup table{border-collapse:collapse}
#mx-popup td{width:27px;height:22px;text-align:center;
  font-family:'Barlow Condensed',sans-serif;font-size:8px;font-weight:700;
  line-height:22px}
#mx-popup .ax{background:none!important;color:#9CA3AF;font-weight:400;font-size:9px}
#mx-popup .pop-xg{font-family:'Barlow Condensed',sans-serif;font-size:9px;
  color:#9CA3AF;text-align:center;margin-top:6px}
"""

st.markdown(f"""<style>
@import url('{_FONTS}');
{_CSS}
</style>""", unsafe_allow_html=True)

# ── App header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-logo">FIFA <span>World Cup</span> 2026 <span class="app-sub">· Predictor</span></div>
  <div class="app-sub">Dixon-Coles Poisson · time-decay · martj42/international_results</div>
</div>
""", unsafe_allow_html=True)

# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading model…", ttl=3600)
def get_model(min_date):
    cache_path = os.path.join(os.path.dirname(__file__), "model_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            c = json.load(f)
        ratings = pd.DataFrame(c["ratings"]).set_index("rank")
        ratings.index.name = None
        return (
            ratings,
            c["home_adv"],
            c["rho"],
            c["valid_teams"],
            np.array(c["params"]),
        )
    df = load_data(min_date=min_date)
    return fit_model(df)

ratings, home_adv, rho, valid_teams, params = get_model(MIN_DATE)

# ── Group simulations ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Simulating groups…", ttl=3600)
def get_group_tables(_vt, _p, _ha, _rho):
    return {l: simulate_group(t, _vt, _p, _ha, _rho, n_sims=20_000)
            for l, t in WC2026_GROUPS.items()}

# ── Tournament simulation ──────────────────────────────────────────────────────
@st.cache_data(show_spinner="Simulating tournament…", ttl=3600)
def get_tournament(_vt, _p, _ha, _rho):
    return simulate_tournament(WC2026_GROUPS, _vt, _p, _ha, _rho, n_sims=8_000)

# ── KO Bracket ────────────────────────────────────────────────────────────────
def _ko_prob(a, b, _vt, _p):
    n = len(_vt)
    tidx = {t: i for i, t in enumerate(_vt)}
    atk = _p[:n]; dfn = _p[n:2*n]
    ai = tidx.get(a, -1); bi = tidx.get(b, -1)
    lam = np.exp((atk[ai] if ai >= 0 else 0.) - (dfn[bi] if bi >= 0 else 0.))
    mu  = np.exp((atk[bi] if bi >= 0 else 0.) - (dfn[ai] if ai >= 0 else 0.))
    p_a = p_d = 0.
    for x in range(9):
        for y in range(9):
            px = np.exp(x*np.log(max(lam,1e-10)) - lam - gammaln(x+1))
            py = np.exp(y*np.log(max(mu, 1e-10)) - mu  - gammaln(y+1))
            p = px*py
            if x > y:    p_a += p
            elif x == y: p_d += p
    return p_a + 0.5*p_d, lam, mu

@st.cache_data(show_spinner="Building bracket…", ttl=3600)
def get_bracket(_gt, _vt, _p):
    def top(df, col): return df.sort_values(col, ascending=False).iloc[0]["Team"]
    pos = {}
    thirds_raw = []
    for letter, df in _gt.items():
        pos[f"{letter}1"] = top(df, "P(1st)")
        pos[f"{letter}2"] = top(df, "P(2nd)")
        r3 = df.iloc[2]  # 3rd-placed team by xPts
        thirds_raw.append((letter, r3["Team"], float(r3["xPts"]), float(r3["xGD"])))
    thirds_raw.sort(key=lambda x: (x[2], x[3]), reverse=True)
    t3_adv = thirds_raw[:8]  # 8 best 3rd-place teams

    # Official FIFA WC 2026 slot eligibility per R32 match
    # Keys match the match numbers that feature a 3rd-place team
    SLOTS = [
        ("T74", set("ABCDF")),   # 1E vs 3[ABCDF]
        ("T77", set("CDFGH")),   # 1I vs 3[CDFGH]
        ("T79", set("CEFHI")),   # 1A vs 3[CEFHI]
        ("T80", set("EHIJK")),   # 1L vs 3[EHIJK]
        ("T81", set("BEFIJ")),   # 1D vs 3[BEFIJ]
        ("T82", set("AEHIJ")),   # 1G vs 3[AEHIJ]
        ("T85", set("EFGIJ")),   # 1B vs 3[EFGIJ]
        ("T87", set("DEIJL")),   # 1K vs 3[DEIJL]
    ]
    remaining = list(t3_adv)
    slot_team = {}
    for slot_name, eligible in SLOTS:
        candidates = [t for t in remaining if t[0] in eligible]
        chosen = (candidates if candidates else remaining)[0]
        slot_team[slot_name] = chosen[1]
        remaining = [t for t in remaining if t is not chosen]

    def resolve(pairs):
        out = []
        for a, b in pairs:
            p, lam, mu = _ko_prob(a, b, _vt, _p)
            out.append({"a": a, "b": b, "prob_a": p,
                        "winner": a if p >= 0.5 else b,
                        "sa": f"{lam:.1f}", "sb": f"{mu:.1f}",
                        "lam": lam, "mu": mu})
        return out

    def w(m): return m["winner"]

    # ── R32 (official match nos. 73-88) ──────────────────────────────────────
    m73 = resolve([(pos["A2"],  pos["B2"])])[0]             # 2A vs 2B
    m74 = resolve([(pos["E1"],  slot_team["T74"])])[0]      # 1E vs 3[ABCDF]
    m75 = resolve([(pos["F1"],  pos["C2"])])[0]             # 1F vs 2C
    m76 = resolve([(pos["C1"],  pos["F2"])])[0]             # 1C vs 2F
    m77 = resolve([(pos["I1"],  slot_team["T77"])])[0]      # 1I vs 3[CDFGH]
    m78 = resolve([(pos["E2"],  pos["I2"])])[0]             # 2E vs 2I
    m79 = resolve([(pos["A1"],  slot_team["T79"])])[0]      # 1A vs 3[CEFHI]
    m80 = resolve([(pos["L1"],  slot_team["T80"])])[0]      # 1L vs 3[EHIJK]
    m81 = resolve([(pos["D1"],  slot_team["T81"])])[0]      # 1D vs 3[BEFIJ]
    m82 = resolve([(pos["G1"],  slot_team["T82"])])[0]      # 1G vs 3[AEHIJ]
    m83 = resolve([(pos["K2"],  pos["L2"])])[0]             # 2K vs 2L
    m84 = resolve([(pos["H1"],  pos["J2"])])[0]             # 1H vs 2J
    m85 = resolve([(pos["B1"],  slot_team["T85"])])[0]      # 1B vs 3[EFGIJ]
    m86 = resolve([(pos["J1"],  pos["H2"])])[0]             # 1J vs 2H
    m87 = resolve([(pos["K1"],  slot_team["T87"])])[0]      # 1K vs 3[DEIJL]
    m88 = resolve([(pos["D2"],  pos["G2"])])[0]             # 2D vs 2G

    # ── R16 (matches 89-96) ──────────────────────────────────────────────────
    m89 = resolve([(w(m74), w(m77))])[0]   # W74 vs W77
    m90 = resolve([(w(m73), w(m75))])[0]   # W73 vs W75
    m91 = resolve([(w(m76), w(m78))])[0]   # W76 vs W78
    m92 = resolve([(w(m79), w(m80))])[0]   # W79 vs W80
    m93 = resolve([(w(m83), w(m84))])[0]   # W83 vs W84
    m94 = resolve([(w(m81), w(m82))])[0]   # W81 vs W82
    m95 = resolve([(w(m86), w(m88))])[0]   # W86 vs W88
    m96 = resolve([(w(m85), w(m87))])[0]   # W85 vs W87

    # ── QF (matches 97-100) ──────────────────────────────────────────────────
    m97  = resolve([(w(m89), w(m90))])[0]  # W89 vs W90
    m98  = resolve([(w(m93), w(m94))])[0]  # W93 vs W94
    m99  = resolve([(w(m91), w(m92))])[0]  # W91 vs W92
    m100 = resolve([(w(m95), w(m96))])[0]  # W95 vs W96

    # ── SF (matches 101-102) ─────────────────────────────────────────────────
    m101 = resolve([(w(m97),  w(m98))])[0]   # W97 vs W98
    m102 = resolve([(w(m99),  w(m100))])[0]  # W99 vs W100

    # ── Final ─────────────────────────────────────────────────────────────────
    final = resolve([(w(m101), w(m102))])[0]

    # Bracket halves ordered so adjacent pairs visually align with next round
    # Left half feeds → m101 (left SF)
    # Right half feeds → m102 (right SF)
    return {
        "L": {
            "R32": [m74, m77, m73, m75, m83, m84, m81, m82],
            "R16": [m89, m90, m93, m94],
            "QF":  [m97, m98],
            "SF":  [m101],
        },
        "R": {
            "SF":  [m102],
            "QF":  [m99, m100],
            "R16": [m91, m92, m95, m96],
            "R32": [m76, m78, m79, m80, m86, m88, m85, m87],
        },
        "final": [final],
    }


_BRACKET_FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=Barlow+Condensed:wght@600;700;900&display=swap"
)

_BRACKET_CSS_INLINE = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Barlow Condensed',sans-serif;background:transparent;overflow-x:auto}
.bkt-wrap{padding:4px 0}
.bkt-inner{display:flex;gap:0;align-items:stretch;min-width:1100px}
.rnd-col{display:flex;flex-direction:column;min-width:136px;flex:1}
.rnd-lbl{font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:10px;
  letter-spacing:1.2px;text-transform:uppercase;color:#9CA3AF;text-align:center;
  padding:0 0 7px;border-bottom:.5px solid #E5E7EB;margin:0 4px 7px}
.rnd-matches{flex:1;display:flex;flex-direction:column;justify-content:space-evenly;
  gap:4px;padding:0 4px}
.mc{background:white;border:.5px solid #E5E7EB;border-radius:4px;overflow:hidden;cursor:default}
.mt{display:flex;align-items:center;gap:4px;padding:4px 8px;
  font-size:11px;font-family:'Barlow Condensed',sans-serif;font-weight:600;
  letter-spacing:.3px;color:#4B5563;text-transform:uppercase;
  border-bottom:.5px solid #E5E7EB}
.mt:last-child{border-bottom:none}
.mt-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mt-xg{width:18px;text-align:center;font-size:11px;font-weight:700;flex-shrink:0}
.mt .prob{width:30px;text-align:right;font-size:10px;font-weight:400;color:#9CA3AF;flex-shrink:0}
.mt.win{background:#19237C;color:#AFEA00}
.mt.win .prob{color:rgba(175,234,0,.7)}
.fin-col{display:flex;flex-direction:column;min-width:140px;flex:1;
  justify-content:center;align-items:stretch;padding:0 4px}
.fin-lbl{font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:10px;
  letter-spacing:1.2px;text-transform:uppercase;color:#AFEA00;text-align:center;
  padding:0 0 7px;border-bottom:.5px solid #AFEA00;margin-bottom:7px}
.trophy-ban{background:#19237C;text-align:center;padding:6px;
  font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:10px;
  letter-spacing:1.5px;text-transform:uppercase;color:#AFEA00;border-radius:3px 3px 0 0}
.fin-mc{border:1.5px solid #AFEA00;border-radius:0 0 4px 4px;overflow:hidden;background:white}
.fin-mc .mt{font-size:13px;padding:6px 10px}
.conn{width:8px;flex-shrink:0}
.bkt-legend{display:flex;gap:16px;align-items:center;margin-top:10px;
  font-size:10px;color:#6B7280;font-family:'Barlow Condensed',sans-serif}
.bl-item{display:flex;align-items:center;gap:6px}
#mx-popup{display:none;position:fixed;left:0;top:0;z-index:9999;background:#fff;
  border:.5px solid #E5E7EB;border-radius:8px;
  box-shadow:0 6px 24px rgba(0,0,0,.14);padding:13px;pointer-events:none;
  white-space:nowrap;width:auto;height:auto}
#mx-popup .pop-header{font-family:'Barlow Condensed',sans-serif;font-weight:700;
  font-size:12px;letter-spacing:.5px;text-transform:uppercase;color:#19237C;
  margin-bottom:8px;text-align:center}
#mx-popup table{border-collapse:collapse}
#mx-popup td{width:27px;height:22px;text-align:center;line-height:22px}
#mx-popup .ax{background:none!important;color:#9CA3AF;font-weight:400;font-size:9px;
  font-family:'Barlow Condensed',sans-serif}
#mx-popup .pop-xg{font-family:'Barlow Condensed',sans-serif;font-size:9px;
  color:#9CA3AF;text-align:center;margin-top:6px}
"""

_POPUP_JS = """<div id="mx-popup"></div>"""

_BRACKET_HEAD_JS = """<script>
function _poi(k,lam){var lp=k*Math.log(Math.max(lam,1e-10))-lam;for(var i=1;i<=k;i++)lp-=Math.log(i);return Math.exp(lp);}
function _lerp(a,b,t){return a.map(function(v,i){return Math.round(v+(b[i]-v)*t);});}
function _cc(type,t){
  var pal={a:[[243,244,246],[255,68,68],[214,0,0]],b:[[243,244,246],[48,79,255],[13,27,110]],d:[[243,244,246],[155,53,168],[92,14,106]]};
  var p=pal[type],s=Math.min(t,1);
  return s<0.5?_lerp(p[0],p[1],s*2):_lerp(p[1],p[2],(s-0.5)*2);
}
function _buildPop(lam,mu,na,nb){
  var G=5,cells=[],maxP=0;
  for(var r=0;r<=G;r++)for(var c=0;c<=G;c++){var p=_poi(r,lam)*_poi(c,mu);if(p>maxP)maxP=p;cells.push({r:r,c:c,p:p,t:r>c?"a":r<c?"b":"d"});}
  var h='<div class="pop-header">'+na+' vs '+nb+'</div><table>';
  h+='<tr><td class="ax" style="width:16px"></td>';
  for(var c=0;c<=G;c++)h+='<td class="ax">'+c+'</td>';
  h+='</tr>';
  for(var r=0;r<=G;r++){
    h+='<tr><td class="ax" style="text-align:right;padding-right:3px">'+r+'</td>';
    for(var c=0;c<=G;c++){
      var cell=cells[r*(G+1)+c],inten=maxP>0?cell.p/maxP:0,rgb=_cc(cell.t,inten),fg=inten>0.55?"#fff":"#6B7280";
      h+='<td style="background:rgb('+rgb+')"><span style="font-family:Barlow Condensed,sans-serif;font-size:9px;font-weight:700;color:'+fg+'">'+(cell.p*100).toFixed(1)+'</span></td>';
    }
    h+='</tr>';
  }
  h+='</table><div class="pop-xg">xG &nbsp;'+na+' '+lam.toFixed(2)+'&nbsp;&nbsp;'+nb+' '+mu.toFixed(2)+'</div>';
  return h;
}
function showPop(el){
  var pop=document.getElementById('mx-popup');
  if(!pop)return;
  pop.innerHTML=_buildPop(parseFloat(el.dataset.lam),parseFloat(el.dataset.mu),el.dataset.na,el.dataset.nb);
  pop.style.right='auto';pop.style.bottom='auto';
  pop.style.left='-9999px';pop.style.top='0';pop.style.display='block';
  var pw=pop.offsetWidth,ph=pop.offsetHeight;
  var r=el.getBoundingClientRect();
  var left=r.right+8,top=r.top;
  if(left+pw>window.innerWidth-4)left=r.left-pw-8;
  if(top+ph>window.innerHeight-4)top=window.innerHeight-ph-4;
  if(top<0)top=0;
  pop.style.left=left+'px';pop.style.top=top+'px';
}
function hidePop(){
  var pop=document.getElementById('mx-popup');
  if(pop)pop.style.display='none';
}
</script>"""

def _bracket_html(bkt):
    def mc(m):
        ca = "win" if m["winner"] == m["a"] else ""
        cb = "win" if m["winner"] == m["b"] else ""
        na = tmeta(m["a"])[1]; nb = tmeta(m["b"])[1]
        sa = m.get("sa", ""); sb = m.get("sb", "")
        lam = m.get("lam", 1.0); mu = m.get("mu", 1.0)
        return (f'<div class="mc" data-lam="{lam:.4f}" data-mu="{mu:.4f}"'
                f' data-na="{na}" data-nb="{nb}"'
                f' onmouseenter="showPop(this)" onmouseleave="hidePop()">'
                f'<div class="mt {ca}"><span class="mt-name">{na}</span>'
                f'<span class="mt-xg">{sa}</span>'
                f'<span class="prob">{m["prob_a"]*100:.0f}%</span></div>'
                f'<div class="mt {cb}"><span class="mt-name">{nb}</span>'
                f'<span class="mt-xg">{sb}</span>'
                f'<span class="prob">{(1-m["prob_a"])*100:.0f}%</span></div>'
                f'</div>')

    def rc(label, matches):
        return (f'<div class="rnd-col"><div class="rnd-lbl">{label}</div>'
                f'<div class="rnd-matches">{"".join(mc(m) for m in matches)}</div></div>')

    fm = bkt["final"][0]
    ca_f = "win" if fm["winner"] == fm["a"] else ""
    cb_f = "win" if fm["winner"] == fm["b"] else ""
    na_f = tmeta(fm["a"])[1]; nb_f = tmeta(fm["b"])[1]
    sa_f = fm.get("sa", ""); sb_f = fm.get("sb", "")
    lam_f = fm.get("lam", 1.0); mu_f = fm.get("mu", 1.0)
    fin = (f'<div class="fin-col"><div class="fin-lbl">Final</div>'
           f'<div><div class="trophy-ban">🏆 FIFA WORLD CUP 2026</div>'
           f'<div class="fin-mc mc" data-lam="{lam_f:.4f}" data-mu="{mu_f:.4f}"'
           f' data-na="{na_f}" data-nb="{nb_f}"'
           f' onmouseenter="showPop(this)" onmouseleave="hidePop()">'
           f'<div class="mt {ca_f}"><span class="mt-name">{na_f}</span>'
           f'<span class="mt-xg">{sa_f}</span>'
           f'<span class="prob">{fm["prob_a"]*100:.0f}%</span></div>'
           f'<div class="mt {cb_f}"><span class="mt-name">{nb_f}</span>'
           f'<span class="mt-xg">{sb_f}</span>'
           f'<span class="prob">{(1-fm["prob_a"])*100:.0f}%</span></div>'
           f'</div></div></div>')

    sp = '<div class="conn"></div>'
    L, R = bkt["L"], bkt["R"]
    body = (
        f'<div class="bkt-wrap"><div class="bkt-inner">'
        + rc("R32",L["R32"])+sp+rc("R16",L["R16"])+sp+rc("QF",L["QF"])+sp+rc("SF",L["SF"])+sp
        + fin + sp
        + rc("SF",R["SF"])+sp+rc("QF",R["QF"])+sp+rc("R16",R["R16"])+sp+rc("R32",R["R32"])
        + '</div></div>'
    )
    return (
        f'<!DOCTYPE html><html><head>'
        f'<link rel="stylesheet" href="{_BRACKET_FONTS}">'
        f'<style>{_BRACKET_CSS_INLINE}</style>'
        f'{_BRACKET_HEAD_JS}'
        f'</head><body>{_POPUP_JS}{body}</body></html>'
    )


# ── Colour helpers ─────────────────────────────────────────────────────────────
def _rgba(hex_col, a):
    h = hex_col.lstrip("#")
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{a:.2f})"

def _grad(v, vmax, col, lo=0.06, hi=0.80):
    a = lo + (hi-lo)*min(float(v)/max(float(vmax),1e-6),1.0)
    return _rgba(col, a)

# ── Group table figure ─────────────────────────────────────────────────────────
def _group_fig(df, advancing_thirds=None):
    advancing_thirds = advancing_thirds or set()
    n = len(df)
    teams = df["Team"].tolist()
    def _stripe(i, t):
        if i < 2: return LIME
        return "#D1FAE5" if t in advancing_thirds else "#E5E7EB"
    def _stripe_fc(i, t):
        if i < 2: return NAVY
        return "#065F46" if t in advancing_thirds else "#9CA3AF"
    stripe    = [_stripe(i, t)    for i, t in enumerate(teams)]
    stripe_fc = [_stripe_fc(i, t) for i, t in enumerate(teams)]
    p1_fill = [_grad(v,80,RED)   for v in df["P(1st)"]]
    p2_fill = [_grad(v,70,BLUE2) for v in df["P(2nd)"]]
    p3_fill = [_grad(v,55,GREEN) for v in df["P(3rd)"]]
    xp_fill = [_grad(v,9, NAVY)  for v in df["xPts"]]
    p1_fc = ["white" if v>45 else "#8B0000" for v in df["P(1st)"]]
    p2_fc = ["white" if v>40 else "#1A237E" for v in df["P(2nd)"]]
    p3_fc = ["white" if v>30 else "#1B5E20" for v in df["P(3rd)"]]
    xp_fc = ["white" if v>5  else NAVY      for v in df["xPts"]]
    fig = go.Figure(go.Table(
        columnwidth=[0.25,2.2,1.0,0.85,0.85,1.1,1.1,1.0,0.9],
        header=dict(
            values=["","<b>Team</b>","<b>xPts</b>","<b>xGF</b>","<b>xGD</b>",
                    "<b>1st</b>","<b>2nd</b>","<b>3rd</b>","<b>4th</b>"],
            fill_color=NAVY,
            font=dict(color=[LIME]+["white"]*8, size=10,
                      family="Barlow Condensed, sans-serif"),
            align=["center","left","center","center","center","center","center","center","center"],
            height=22, line_color="#2A3A9C",
        ),
        cells=dict(
            values=[
                [""]*n, [tshort(t) for t in teams],
                [f"{v:.1f}" for v in df["xPts"]],
                [f"{v:.1f}" for v in df["xGF"]],
                [f"{v:+.1f}" for v in df["xGD"]],
                [f"{v:.0f}%" for v in df["P(1st)"]],
                [f"{v:.0f}%" for v in df["P(2nd)"]],
                [f"{v:.0f}%" for v in df["P(3rd)"]],
                [f"{v:.0f}%" for v in df["P(4th)"]],
            ],
            fill_color=[stripe,["white"]*n,xp_fill,["white"]*n,["white"]*n,
                        p1_fill,p2_fill,p3_fill,["white"]*n],
            font=dict(
                color=[stripe_fc,[NAVY]*n,xp_fc,[NAVY]*n,[NAVY]*n,
                       p1_fc,p2_fc,p3_fc,["#9CA3AF"]*n],
                size=[10, 8, 10, 10, 10, 10, 10, 10, 10],
                family="Noto Sans, sans-serif",
            ),
            align=["center","left","center","center","center","center","center","center","center"],
            height=22, line_color="#E5E7EB",
        ),
    ))
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=22*(n+1)+10)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Match Simulator", "🏆 Group Stage",
    "🗓 KO Bracket", "📈 Backtest",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MATCH SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    # Group selector restricts team options AND sets defaults
    def _on_group():
        g = st.session_state.get("grp_sel", "—")
        if g and g != "—":
            letter = g.split()[-1]
            gt = [t for t in WC2026_GROUPS.get(letter, []) if t in valid_teams]
            if gt:        st.session_state["team_a"] = gt[0]
            if len(gt)>1: st.session_state["team_b"] = gt[1]

    for key, default in [("team_a","France"),("team_b","Brazil")]:
        if key not in st.session_state:
            st.session_state[key] = default if default in valid_teams else sorted(valid_teams)[0]

    # Determine which teams are selectable
    grp_now = st.session_state.get("grp_sel", "—")
    if grp_now and grp_now != "—":
        ltr = grp_now.split()[-1]
        team_options = [t for t in WC2026_GROUPS.get(ltr, []) if t in valid_teams]
    else:
        team_options = sorted(valid_teams)

    # Guard: ensure session state values are valid for current options
    for key in ("team_a", "team_b"):
        if st.session_state.get(key) not in team_options:
            st.session_state[key] = team_options[0]

    team_opts_display = [tdisplay(t) for t in team_options]
    disp_to_name      = {tdisplay(t): t for t in team_options}

    def _idx(key):
        want = st.session_state.get(key, team_options[0])
        disp = tdisplay(want)
        return team_opts_display.index(disp) if disp in team_opts_display else 0

    _, cc1, cc2, cc3, _ = st.columns([2, 1, 1.4, 1.4, 2])
    with cc1:
        st.selectbox("Filter by group",
                     ["—"] + [f"Group {l}" for l in WC2026_GROUPS],
                     key="grp_sel", on_change=_on_group)
    with cc2:
        sel_a = st.selectbox("Team A", team_opts_display, index=_idx("team_a"), key="_disp_a")
        team_a = disp_to_name[sel_a]
    with cc3:
        sel_b = st.selectbox("Team B", team_opts_display, index=_idx("team_b"), key="_disp_b")
        team_b = disp_to_name[sel_b]

    if team_a == team_b:
        st.warning("Select two different teams.")
        st.stop()

    result = predict_match(team_a, team_b, neutral=NEUTRAL,
                           ratings_df=ratings, home_adv=home_adv, rho=rho,
                           valid_teams=valid_teams, params=params)
    lam, mu = result["lam"], result["mu"]
    win_a, draw, win_b = result["win_a"], result["draw"], result["win_b"]
    xpts_a = 3*win_a + draw
    xpts_b = 3*win_b + draw

    flag_a, code_a = tmeta(team_a)
    flag_b, code_b = tmeta(team_b)

    col_main, col_side = st.columns([3, 1], gap="large")

    # ── Left: heatmap panel ────────────────────────────────────────────────────
    with col_main:
        st.markdown(f"""
        <div class="panel">
          <div class="ph">Score Probability Matrix
            <span class="ph-sub">Dixon-Coles Poisson · rho correction</span></div>
          <div class="pb">
            <div class="teams-bar">
              <div class="tbadge-home">
                <div class="t-flag-code">{flag_a} {code_a}</div>
              </div>
              <div class="vs-pill">VS</div>
              <div class="tbadge-away">
                <div class="t-flag-code">{code_b} {flag_b}</div>
              </div>
            </div>
        """, unsafe_allow_html=True)

        highlight = st.segmented_control(
            "Highlight",
            options=["All", f"Win {code_a}", "Draw", f"Win {code_b}"],
            default="All", key=f"hl_{team_a}_{team_b}",
        )

        # Build score matrix (0–4 + 5+)
        MAX_G = 4
        full_sm = result["score_matrix"]
        sm = np.zeros((MAX_G+2, MAX_G+2))
        sm[:MAX_G+1, :MAX_G+1] = full_sm[:MAX_G+1, :MAX_G+1]
        sm[MAX_G+1, :MAX_G+1]  = full_sm[MAX_G+1:, :MAX_G+1].sum(axis=0)
        sm[:MAX_G+1, MAX_G+1]  = full_sm[:MAX_G+1, MAX_G+1:].sum(axis=1)
        sm[MAX_G+1, MAX_G+1]   = full_sm[MAX_G+1:, MAX_G+1:].sum()
        sm_pct = sm * 100
        nc = MAX_G + 2
        gl = [str(i) for i in range(MAX_G+1)] + ["5+"]
        a_marg = sm_pct.sum(axis=1)
        b_marg = sm_pct.sum(axis=0)
        y_lbl = [f"{gl[i]}  ({a_marg[i]:.0f}%)" for i in range(nc)]
        x_lbl = [f"{gl[i]}<br>({b_marg[i]:.0f}%)" for i in range(nc)]

        # Opacity mask for highlight
        omask = np.ones((nc, nc))
        if highlight == f"Win {code_a}":
            for r in range(nc):
                for c in range(nc):
                    if r <= c: omask[r,c] = 0.07
        elif highlight == "Draw":
            for r in range(nc):
                for c in range(nc):
                    if r != c: omask[r,c] = 0.07
        elif highlight == f"Win {code_b}":
            for r in range(nc):
                for c in range(nc):
                    if r >= c: omask[r,c] = 0.07

        # ── Tricolor z encoding ──────────────────────────────────────────────
        # Positive → team A wins (red), negative → team B wins (navy), 0 → draw (purple shapes)
        vmax = sm_pct.max()

        # Three separate z matrices — NaN masks out irrelevant cells per trace
        z_a = np.full((nc, nc), np.nan)  # team A wins (r > c)
        z_b = np.full((nc, nc), np.nan)  # team B wins (r < c)
        z_d = np.full((nc, nc), np.nan)  # draws       (r == c)
        for r in range(nc):
            for c in range(nc):
                val = sm_pct[r, c] * omask[r, c]
                if   r > c: z_a[r, c] = val
                elif r < c: z_b[r, c] = val
                else:       z_d[r, c] = val

        # All three scales share the same near-white at zero probability
        _cs_red    = [[0.00,"#F3F4F6"],[0.20,"#FFD0D0"],[0.60,"#FF4444"],[1.00,"#D60000"]]
        _cs_blue   = [[0.00,"#F3F4F6"],[0.20,"#D6E4FF"],[0.60,"#304FFF"],[1.00,"#0D1B6E"]]
        _cs_purple = [[0.00,"#F3F4F6"],[0.20,"#EAD5EC"],[0.60,"#9B35A8"],[1.00,"#5C0E6A"]]

        # Annotations
        _BC = "Barlow Condensed, sans-serif"
        anns = []
        for r in range(nc):
            for c in range(nc):
                op = omask[r, c]
                v  = sm_pct[r, c]
                bright = v * op >= vmax * 0.38 and op > 0.5
                col_txt = f"rgba(255,255,255,{op:.2f})" if bright else f"rgba(55,65,81,{op:.2f})"
                anns.append(dict(x=c, y=r, text=f"{v:.1f}%",
                                 showarrow=False,
                                 font=dict(size=14, color=col_txt, family=_BC),
                                 xref="x", yref="y"))

        # Hover text for all cells (used by the transparent overlay trace)
        hover_text = [
            [f"{code_a} {r} – {code_b} {c}<br>{sm_pct[r,c]:.1f}%"
             for c in range(nc)]
            for r in range(nc)
        ]

        # Three colour traces (no hover — topmost trace was capturing all events,
        # showing empty tooltip for non-draw cells)
        _hm_base = dict(zmin=0, zmax=vmax, showscale=False, xgap=3, ygap=3,
                        hoverinfo="skip")
        fig_hm = go.Figure()
        for z, cs in [(z_b, _cs_blue), (z_a, _cs_red), (z_d, _cs_purple)]:
            fig_hm.add_trace(go.Heatmap(z=z, colorscale=cs, **_hm_base))

        # Transparent overlay — sits on top, owns hover for every cell
        fig_hm.add_trace(go.Heatmap(
            z=np.ones((nc, nc)),
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
            showscale=False, xgap=3, ygap=3,
            text=hover_text,
            hovertemplate="<b>%{text}</b><extra></extra>",
        ))
        fig_hm.update_layout(
            annotations=anns, height=460,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family=_BC),
            hoverlabel=dict(font=dict(family=_BC, size=12),
                            bgcolor="white", bordercolor="#E5E7EB"),
            xaxis=dict(tickvals=list(range(nc)), ticktext=x_lbl,
                       tickfont=dict(size=11, family=_BC), side="top",
                       title=dict(text=f"<b>{code_b}</b> goals",
                                  font=dict(size=12, family=_BC), standoff=4),
                       showgrid=False, fixedrange=True),
            yaxis=dict(tickvals=list(range(nc)), ticktext=y_lbl,
                       tickfont=dict(size=11, family=_BC), autorange="reversed",
                       title=dict(text=f"<b>{code_a}</b> goals",
                                  font=dict(size=12, family=_BC), standoff=4),
                       showgrid=False, fixedrange=True),
        )
        st.plotly_chart(fig_hm, use_container_width=True,
                        config={"displayModeBar": False})

        # Outcome pills
        if   win_a > win_b + 0.05: cls_a, cls_b = "pill-a-str", "pill-b-fnt"
        elif win_b > win_a + 0.05: cls_a, cls_b = "pill-a-fnt", "pill-b-str"
        else:                      cls_a, cls_b = "pill-a-fnt", "pill-b-fnt"
        st.markdown(f"""
          <div class="outcome-row">
            <div class="outcome-pill {cls_a}">
              <span class="pct">{win_a*100:.0f}%</span>
              <span class="lbl">WIN {code_a}</span>
            </div>
            <div class="outcome-pill pill-draw">
              <span class="pct">{draw*100:.0f}%</span>
              <span class="lbl">DRAW</span>
            </div>
            <div class="outcome-pill {cls_b}">
              <span class="pct">{win_b*100:.0f}%</span>
              <span class="lbl">WIN {code_b}</span>
            </div>
          </div>
          </div></div>
        """, unsafe_allow_html=True)

    # ── Right: stats panels ────────────────────────────────────────────────────
    with col_side:
        XG_MAX = 3.0  # fixed scale so bars are meaningful in absolute terms
        pct_a  = min(lam / XG_MAX * 100, 100)
        pct_b  = min(mu  / XG_MAX * 100, 100)
        st.markdown(f"""
        <div class="panel">
          <div class="ph">xG Breakdown</div>
          <div class="pb">
            <div class="xg-row">
              <span class="xg-lbl">{flag_a} {code_a}</span>
              <span class="xg-val">{lam:.2f}</span>
            </div>
            <div class="xg-track">
              <div class="xg-fill" style="width:{pct_a:.0f}%;background:{RED}"></div>
            </div>
            <div class="xg-row">
              <span class="xg-lbl">{flag_b} {code_b}</span>
              <span class="xg-val">{mu:.2f}</span>
            </div>
            <div class="xg-track">
              <div class="xg-fill" style="width:{pct_b:.0f}%;background:{BLUE2}"></div>
            </div>
            <div class="sep"></div>
            <div class="xg-row">
              <span class="xg-lbl" style="font-size:10px;color:#9CA3AF">xG diff</span>
              <span style="font-family:'Barlow Condensed',sans-serif;font-weight:700;
                font-size:14px;color:{'#D60000' if lam>=mu else '#304FFF'}">{lam-mu:+.2f}</span>
            </div>
          </div>
        </div>

        <div class="panel">
          <div class="ph">Expected Points</div>
          <div class="pb">
            <div class="xp-grid">
              <div class="xp-box" style="background:rgba(214,0,0,.08)">
                <div class="xp-lbl" style="color:{RED}">{code_a}</div>
                <div class="xp-val" style="color:{RED}">{xpts_a:.2f}</div>
                <div class="xp-sub">xPts</div>
              </div>
              <div class="xp-box" style="background:rgba(25,35,124,.08)">
                <div class="xp-lbl" style="color:{NAVY}">{code_b}</div>
                <div class="xp-val" style="color:{NAVY}">{xpts_b:.2f}</div>
                <div class="xp-sub">xPts</div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        top = top_scorelines(result, top_n=8)
        max_p = float(top["prob"].iloc[0])
        score_rows = ""
        for _, row in top.iterrows():
            h, ag = int(row["goals_a"]), int(row["goals_b"])
            intensity = (row["prob"] / max_p) ** 0.7  # gamma < 1 → stronger gradient
            if h > ag:
                bg = _rgba(RED,   intensity * 0.35)
            elif h < ag:
                bg = _rgba(BLUE2, intensity * 0.30)
            else:
                bg = _rgba("#832780", intensity * 0.28)
            score_rows += (
                f'<div class="score-row" style="background:{bg}">'
                f'<span class="score-lbl">{h} – {ag}</span>'
                f'<span class="score-pct">{row["prob"]*100:.1f}%</span>'
                f'</div>'
            )
        st.markdown(f"""
        <div class="panel">
          <div class="ph">Most Likely Scores</div>
          <div class="pb">{score_rows}</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GROUP STAGE
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="grp-legend">
      <span><span class="lq"></span> Qualified (1st &amp; 2nd)</span>
      <span><span class="l3a"></span> Top-8 3rd place (advancing)</span>
      <span><span class="l3"></span> Eliminated</span>
      <span style="margin-left:auto;font-size:9px;text-transform:uppercase;letter-spacing:.5px">
        xPts · xGF · xGD = expected values across 3 group games &nbsp;·&nbsp;
        1st/2nd/3rd = finish probability
      </span>
    </div>
    """, unsafe_allow_html=True)

    group_tables = get_group_tables(valid_teams, params, home_adv, rho)

    # Top 8 3rd-place teams advance; take row 2 (xPts-sorted) from each group, rank by xPts then xGD
    _thirds = [(df.iloc[2]["Team"], df.iloc[2]["xPts"], df.iloc[2]["xGD"])
               for df in group_tables.values()]
    _thirds.sort(key=lambda x: (x[1], x[2]), reverse=True)
    advancing_thirds = {t[0] for t in _thirds[:8]}

    letters = list(WC2026_GROUPS.keys())

    for row_start in range(0, len(letters), 3):
        cols = st.columns(3, gap="medium")
        for ci, letter in enumerate(letters[row_start:row_start+3]):
            with cols[ci]:
                df = group_tables[letter]
                sub = " · ".join(tmeta(t)[1] for t in WC2026_GROUPS[letter])
                st.markdown(
                    f'<div class="grp-card"><div class="grp-hdr">'
                    f'<span class="grp-letter">GROUP {letter}</span>'
                    f'<span class="grp-sub">{sub}</span>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(_group_fig(df, advancing_thirds), use_container_width=True,
                                config={"displayModeBar": False})
        st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — KO BRACKET
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        "<p style='color:#9CA3AF;font-size:0.82rem;margin-bottom:12px'>"
        "Most-likely bracket based on group simulation. Top 2 per group + 8 best 3rd-place teams advance to R32. "
        "Official FIFA WC 2026 bracket structure. Hover over any fixture to see the full score-probability matrix.</p>",
        unsafe_allow_html=True,
    )
    group_tables = get_group_tables(valid_teams, params, home_adv, rho)
    bracket = get_bracket(group_tables, valid_teams, params)
    components.html(_bracket_html(bracket), height=440, scrolling=True)

    st.markdown("""
    <div class="bkt-legend">
      <div class="bl-item">
        <div style="width:14px;height:9px;background:#19237C;border-radius:2px"></div>
        Predicted winner
      </div>
      <div class="bl-item">
        <div style="width:14px;height:9px;border:1.5px solid #AFEA00;border-radius:2px"></div>
        Final
      </div>
      <div style="margin-left:auto;font-size:9px;text-transform:uppercase;letter-spacing:.5px;color:#6B7280">
        % = win probability &nbsp;·&nbsp; score = xG
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Round-by-round + Rankings ──────────────────────────────────────────────
    tourn_df = get_tournament(valid_teams, params, home_adv, rho)
    tourn_df = tourn_df.sort_values(
        ["Win","R32"], ascending=False
    ).reset_index(drop=True)
    all_teams = tourn_df.copy()
    for c in ["R32","R16","QF","SF","Final","Win"]:
        all_teams[c] = (all_teams[c]*100).round(1)

    c_rnd, c_rank = st.columns([3, 2], gap="large")

    with c_rnd:
        st.markdown(
            '<div class="grp-card"><div class="grp-hdr">'
            '<span class="grp-letter">Round-by-Round</span>'
            '<span class="grp-sub">All 48 teams · sorted by Win %</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        win_fill = [_grad(v, all_teams["Win"].max(),  RED,   0.05, 0.82) for v in all_teams["Win"]]
        fin_fill = [_grad(v, all_teams["Final"].max(),BLUE2, 0.04, 0.65) for v in all_teams["Final"]]
        sf_fill  = [_grad(v, all_teams["SF"].max(),   GREEN, 0.03, 0.50) for v in all_teams["SF"]]
        fig_rnd = go.Figure(go.Table(
            columnwidth=[2.4, 0.75, 0.75, 0.75, 0.85, 0.95, 1.05],
            header=dict(
                values=["<b>Team</b>","<b>R32</b>","<b>R16</b>","<b>QF</b>",
                        "<b>SF</b>","<b>Final</b>","<b>🏆 Win</b>"],
                fill_color=NAVY,
                font=dict(color=[LIME]+["white"]*6, size=11,
                          family="Barlow Condensed, sans-serif"),
                align=["left"]+["center"]*6,
                height=24, line_color="#2A3A9C",
            ),
            cells=dict(
                values=[
                    all_teams["Team"].tolist(),
                    [f"{v:.0f}%" for v in all_teams["R32"]],
                    [f"{v:.0f}%" for v in all_teams["R16"]],
                    [f"{v:.0f}%" for v in all_teams["QF"]],
                    [f"{v:.1f}%" for v in all_teams["SF"]],
                    [f"{v:.1f}%" for v in all_teams["Final"]],
                    [f"{v:.1f}%" for v in all_teams["Win"]],
                ],
                fill_color=[["white"]*len(all_teams)]*4 + [sf_fill, fin_fill, win_fill],
                font=dict(color="#374151", size=10, family="Noto Sans, sans-serif"),
                align=["left"]+["center"]*6,
                height=22, line_color="#E5E7EB",
            ),
        ))
        fig_rnd.update_layout(margin=dict(l=0,r=0,t=0,b=0),
                              height=24 + 22*len(all_teams) + 10)
        st.plotly_chart(fig_rnd, use_container_width=True,
                        config={"displayModeBar": False})

    with c_rank:
        st.markdown(
            '<div class="grp-card"><div class="grp-hdr">'
            '<span class="grp-letter">SPI Rankings</span>'
            '<span class="grp-sub">WC 2026 teams</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        wc_teams = {t for grp in WC2026_GROUPS.values() for t in grp}
        disp_ko = ratings[ratings["team"].isin(wc_teams)].copy()
        disp_ko = disp_ko.sort_values("spi", ascending=False).reset_index(drop=True)
        spi_max = float(disp_ko["spi"].max())
        def _spi_fill(v):
            if v <= 50:
                return "#F3F4F6"
            t = min((v - 50) / max(spi_max - 50, 1e-6), 1.0)
            return _rgba(RED, 0.06 + 0.76 * t)
        spi_fill_ko = [_spi_fill(v) for v in disp_ko["spi"]]
        spi_fc = ["white" if (v - 50) / max(spi_max - 50, 1e-6) > 0.55 else NAVY
                  for v in disp_ko["spi"]]
        fig_rank = go.Figure(go.Table(
            columnwidth=[3, 1.1, 1.1, 1.1],
            header=dict(
                values=["<b>Team</b>","<b>SPI</b>","<b>Attack</b>","<b>Defense</b>"],
                fill_color=NAVY,
                font=dict(color=[LIME]+["white"]*3, size=11,
                          family="Barlow Condensed, sans-serif"),
                align=["left","center","center","center"],
                height=24, line_color="#2A3A9C",
            ),
            cells=dict(
                values=[
                    disp_ko["team"].tolist(),
                    [f"{v:.1f}" for v in disp_ko["spi"]],
                    [f"{v:+.3f}" for v in disp_ko["attack"]],
                    [f"{v:+.3f}" for v in disp_ko["defense"]],
                ],
                fill_color=[spi_fill_ko, spi_fill_ko,
                            ["white"]*len(disp_ko), ["white"]*len(disp_ko)],
                font=dict(color=[spi_fc, spi_fc, ["#374151"]*len(disp_ko), ["#374151"]*len(disp_ko)],
                          size=10, family="Noto Sans, sans-serif"),
                align=["left","center","center","center"],
                height=22, line_color="#E5E7EB",
            ),
        ))
        fig_rank.update_layout(margin=dict(l=0,r=0,t=0,b=0),
                               height=24 + 22*len(disp_ko) + 10)
        st.plotly_chart(fig_rank, use_container_width=True,
                        config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BACKTEST  (Round(xG) · pre-computed)
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    DATA = {
        "WC 2022":   {"n": 64, "exact": 3,  "gd": 9, "tend": 14, "miss": 38},
        "Euro 2024": {"n": 51, "exact": 12, "gd": 6, "tend": 8,  "miss": 25},
    }

    tourneys = list(DATA.keys())
    ns     = [d["n"]     for d in DATA.values()]
    exacts = [d["exact"] for d in DATA.values()]
    gds    = [d["gd"]    for d in DATA.values()]
    tends  = [d["tend"]  for d in DATA.values()]
    misses = [d["miss"]  for d in DATA.values()]
    totals = [4*d["exact"] + 2*d["gd"] + d["tend"] for d in DATA.values()]
    ppgs   = [t / n for t, n in zip(totals, ns)]

    exact_fill = [_grad(v, max(exacts), LIME,  0.12, 0.55) for v in exacts]
    gd_fill    = [_grad(v, max(gds),   BLUE2, 0.08, 0.40) for v in gds]
    tend_fill  = [_grad(v, max(tends), GREEN, 0.08, 0.40) for v in tends]
    total_fill = [_grad(v, max(totals), RED,  0.08, 0.60) for v in totals]
    ppg_fill   = [_grad(v, max(ppgs),  RED,  0.08, 0.60) for v in ppgs]

    nrows = len(tourneys)
    fig_bt = go.Figure(go.Table(
        columnwidth=[1.8, 0.6, 0.9, 0.9, 0.9, 0.9, 0.9, 1.0],
        header=dict(
            values=["<b>Tournament</b>", "<b>Games</b>",
                    "<b>Exact</b>", "<b>Goal Difference</b>", "<b>Tendency</b>", "<b>Miss</b>",
                    "<b>Total Pts</b>", "<b>Pts / Game</b>"],
            fill_color=NAVY,
            font=dict(color=[LIME] + ["white"] * 7, size=13,
                      family="Barlow Condensed, sans-serif"),
            align=["left"] + ["center"] * 7,
            height=28, line_color="#2A3A9C",
        ),
        cells=dict(
            values=[
                tourneys, ns, exacts, gds, tends, misses,
                totals, [f"{v:.2f}" for v in ppgs],
            ],
            fill_color=[
                ["white"] * nrows,
                ["white"] * nrows,
                exact_fill, gd_fill, tend_fill,
                ["#F3F4F6"] * nrows,
                total_fill, ppg_fill,
            ],
            font=dict(
                color=[[NAVY] * nrows] * 8,
                size=11, family="Noto Sans, sans-serif",
            ),
            align=["left"] + ["center"] * 7,
            height=26, line_color="#E5E7EB",
        ),
    ))
    fig_bt.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=28 + 26 * nrows + 10)

    st.markdown(
        '<div class="grp-card"><div class="grp-hdr">'
        '<span class="grp-letter">BACKTEST</span>'
        '<span class="grp-sub">Round(xG) · 4 pts exact · 2 pts GD · 1 pt tendency</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig_bt, use_container_width=True, config={"displayModeBar": False})
