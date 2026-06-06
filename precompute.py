"""
Run this locally to regenerate all pre-computed data files.
Saves model_cache.json (for Streamlit fallback) and frontend/public/data/*.
"""
import json
import os
import numpy as np
from datetime import datetime
from scipy.special import gammaln
from model import load_data, fit_model, simulate_group, simulate_tournament, WC2026_GROUPS

MIN_DATE = "2014-01-01"

# ── Team metadata (mirrors app.py) ────────────────────────────────────────────
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
NAME_SHORT = {"Bosnia and Herzegovina": "Bosnia & H."}

def tmeta(t):
    return TEAM_META.get(t, ("🌍", t[:3].upper()))

def tshort(t):
    return NAME_SHORT.get(t, t)


# ── Bracket helpers (adapted from app.py) ─────────────────────────────────────
def _ko_prob(a, b, vt, p):
    n = len(vt)
    tidx = {t: i for i, t in enumerate(vt)}
    atk = p[:n]; dfn = p[n:2*n]
    ai = tidx.get(a, -1); bi = tidx.get(b, -1)
    lam = np.exp((atk[ai] if ai >= 0 else 0.) - (dfn[bi] if bi >= 0 else 0.))
    mu  = np.exp((atk[bi] if bi >= 0 else 0.) - (dfn[ai] if ai >= 0 else 0.))
    p_a = p_d = 0.
    for x in range(9):
        for y in range(9):
            px = np.exp(x*np.log(max(lam,1e-10)) - lam - gammaln(x+1))
            py = np.exp(y*np.log(max(mu, 1e-10)) - mu  - gammaln(y+1))
            prob = px * py
            if x > y:    p_a += prob
            elif x == y: p_d += prob
    return p_a + 0.5*p_d, lam, mu


def get_bracket(group_tables, vt, p):
    pos = {}
    thirds_raw = []
    for letter, df in group_tables.items():
        p1 = df.sort_values("P(1st)", ascending=False).iloc[0]["Team"]
        p2 = df[df["Team"] != p1].sort_values("P(2nd)", ascending=False).iloc[0]["Team"]
        pos[f"{letter}1"] = p1
        pos[f"{letter}2"] = p2
        # 3rd placer: highest xPts among teams not already used as 1st or 2nd
        r3 = df[~df["Team"].isin([p1, p2])].iloc[0]
        thirds_raw.append((letter, r3["Team"], float(r3["xPts"]), float(r3["xGD"])))
    thirds_raw.sort(key=lambda x: (x[2], x[3]), reverse=True)
    t3_adv = thirds_raw[:8]

    SLOTS = [
        ("T74", set("ABCDF")),
        ("T77", set("CDFGH")),
        ("T79", set("CEFHI")),
        ("T80", set("EHIJK")),
        ("T81", set("BEFIJ")),
        ("T82", set("AEHIJ")),
        ("T85", set("EFGIJ")),
        ("T87", set("DEIJL")),
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
            prob, lam, mu = _ko_prob(a, b, vt, p)
            out.append({"a": a, "b": b, "prob_a": round(prob, 4),
                        "winner": a if prob >= 0.5 else b,
                        "sa": f"{lam:.1f}", "sb": f"{mu:.1f}",
                        "lam": round(float(lam), 4), "mu": round(float(mu), 4)})
        return out

    def w(m): return m["winner"]

    m73 = resolve([(pos["A2"],  pos["B2"])])[0]
    m74 = resolve([(pos["E1"],  slot_team["T74"])])[0]
    m75 = resolve([(pos["F1"],  pos["C2"])])[0]
    m76 = resolve([(pos["C1"],  pos["F2"])])[0]
    m77 = resolve([(pos["I1"],  slot_team["T77"])])[0]
    m78 = resolve([(pos["E2"],  pos["I2"])])[0]
    m79 = resolve([(pos["A1"],  slot_team["T79"])])[0]
    m80 = resolve([(pos["L1"],  slot_team["T80"])])[0]
    m81 = resolve([(pos["D1"],  slot_team["T81"])])[0]
    m82 = resolve([(pos["G1"],  slot_team["T82"])])[0]
    m83 = resolve([(pos["K2"],  pos["L2"])])[0]
    m84 = resolve([(pos["H1"],  pos["J2"])])[0]
    m85 = resolve([(pos["B1"],  slot_team["T85"])])[0]
    m86 = resolve([(pos["J1"],  pos["H2"])])[0]
    m87 = resolve([(pos["K1"],  slot_team["T87"])])[0]
    m88 = resolve([(pos["D2"],  pos["G2"])])[0]

    m89  = resolve([(w(m74), w(m77))])[0]
    m90  = resolve([(w(m73), w(m75))])[0]
    m91  = resolve([(w(m76), w(m78))])[0]
    m92  = resolve([(w(m79), w(m80))])[0]
    m93  = resolve([(w(m83), w(m84))])[0]
    m94  = resolve([(w(m81), w(m82))])[0]
    m95  = resolve([(w(m86), w(m88))])[0]
    m96  = resolve([(w(m85), w(m87))])[0]

    m97  = resolve([(w(m89), w(m90))])[0]
    m98  = resolve([(w(m93), w(m94))])[0]
    m99  = resolve([(w(m91), w(m92))])[0]
    m100 = resolve([(w(m95), w(m96))])[0]

    m101 = resolve([(w(m97),  w(m98))])[0]
    m102 = resolve([(w(m99),  w(m100))])[0]
    final = resolve([(w(m101), w(m102))])[0]

    return {
        "L": {"R32": [m74,m77,m73,m75,m83,m84,m81,m82],
              "R16": [m89,m90,m93,m94], "QF": [m97,m98], "SF": [m101]},
        "R": {"SF": [m102], "QF": [m99,m100],
              "R16": [m91,m92,m95,m96], "R32": [m76,m78,m79,m80,m86,m88,m85,m87]},
        "final": [final],
    }


# ── Bracket HTML renderer (adapted from app.py) ───────────────────────────────
_FONTS = ("https://fonts.googleapis.com/css2?"
          "family=Barlow+Condensed:wght@600;700;900&display=swap")

_CSS = """
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
#mx-popup{display:none;position:fixed;left:0;top:0;z-index:9999;background:#fff;
  border:.5px solid #E5E7EB;border-radius:8px;
  box-shadow:0 6px 24px rgba(0,0,0,.14);padding:13px;pointer-events:none;
  white-space:nowrap}
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

_JS = """<script>
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
  pop.style.display='block';pop.style.left='-9999px';pop.style.top='0';
  var pw=pop.offsetWidth,ph=pop.offsetHeight;
  var r=el.getBoundingClientRect();
  var left=r.right+8,top=r.top;
  if(left+pw>window.innerWidth-4)left=r.left-pw-8;
  if(top+ph>window.innerHeight-4)top=window.innerHeight-ph-4;
  if(top<0)top=0;
  pop.style.left=left+'px';pop.style.top=top+'px';
}
function hidePop(){var pop=document.getElementById('mx-popup');if(pop)pop.style.display='none';}
</script>"""


def _bracket_html(bkt):
    def mc(m):
        ca = "win" if m["winner"] == m["a"] else ""
        cb = "win" if m["winner"] == m["b"] else ""
        na = tmeta(m["a"])[1]; nb = tmeta(m["b"])[1]
        lam = m["lam"]; mu = m["mu"]
        return (f'<div class="mc" data-lam="{lam}" data-mu="{mu}"'
                f' data-na="{na}" data-nb="{nb}"'
                f' onmouseenter="showPop(this)" onmouseleave="hidePop()">'
                f'<div class="mt {ca}"><span class="mt-name">{na}</span>'
                f'<span class="mt-xg">{m["sa"]}</span>'
                f'<span class="prob">{m["prob_a"]*100:.0f}%</span></div>'
                f'<div class="mt {cb}"><span class="mt-name">{nb}</span>'
                f'<span class="mt-xg">{m["sb"]}</span>'
                f'<span class="prob">{(1-m["prob_a"])*100:.0f}%</span></div>'
                f'</div>')

    def rc(label, matches):
        return (f'<div class="rnd-col"><div class="rnd-lbl">{label}</div>'
                f'<div class="rnd-matches">{"".join(mc(m) for m in matches)}</div></div>')

    fm = bkt["final"][0]
    ca_f = "win" if fm["winner"] == fm["a"] else ""
    cb_f = "win" if fm["winner"] == fm["b"] else ""
    na_f = tmeta(fm["a"])[1]; nb_f = tmeta(fm["b"])[1]
    fin = (f'<div class="fin-col"><div class="fin-lbl">Final</div>'
           f'<div><div class="trophy-ban">🏆 FIFA WORLD CUP 2026</div>'
           f'<div class="fin-mc mc" data-lam="{fm["lam"]}" data-mu="{fm["mu"]}"'
           f' data-na="{na_f}" data-nb="{nb_f}"'
           f' onmouseenter="showPop(this)" onmouseleave="hidePop()">'
           f'<div class="mt {ca_f}"><span class="mt-name">{na_f}</span>'
           f'<span class="mt-xg">{fm["sa"]}</span>'
           f'<span class="prob">{fm["prob_a"]*100:.0f}%</span></div>'
           f'<div class="mt {cb_f}"><span class="mt-name">{nb_f}</span>'
           f'<span class="mt-xg">{fm["sb"]}</span>'
           f'<span class="prob">{(1-fm["prob_a"])*100:.0f}%</span></div>'
           f'</div></div></div>')

    sp = '<div class="conn"></div>'
    L, R = bkt["L"], bkt["R"]
    body = (
        '<div class="bkt-wrap"><div class="bkt-inner">'
        + rc("R32", L["R32"]) + sp + rc("R16", L["R16"]) + sp
        + rc("QF", L["QF"])   + sp + rc("SF",  L["SF"])  + sp
        + fin + sp
        + rc("SF", R["SF"])   + sp + rc("QF",  R["QF"])  + sp
        + rc("R16", R["R16"]) + sp + rc("R32", R["R32"])
        + '</div></div>'
    )
    return (f'<!DOCTYPE html><html><head>'
            f'<link rel="stylesheet" href="{_FONTS}">'
            f'<style>{_CSS}</style>{_JS}'
            f'</head><body><div id="mx-popup"></div>{body}</body></html>')


# ── Main ──────────────────────────────────────────────────────────────────────
print("Fetching data and fitting model...")
df = load_data(min_date=MIN_DATE)
ratings, home_adv, rho, valid_teams, params = fit_model(df)

# ── model_cache.json (Streamlit fallback) ─────────────────────────────────────
cache = {
    "generated_at": datetime.utcnow().isoformat(),
    "home_adv": float(home_adv),
    "rho": float(rho),
    "valid_teams": valid_teams,
    "params": params.tolist(),
    "ratings": ratings.reset_index().rename(columns={"index": "rank"}).to_dict(orient="records"),
}
with open("model_cache.json", "w") as f:
    json.dump(cache, f)
print(f"  model_cache.json — {len(valid_teams)} teams")

# ── frontend/public/data/ ─────────────────────────────────────────────────────
os.makedirs("frontend/public/data", exist_ok=True)

# model.json
model_data = {
    "home_adv": float(home_adv),
    "rho": float(rho),
    "valid_teams": valid_teams,
    "params": params.tolist(),
    "wc_groups": WC2026_GROUPS,
}
with open("frontend/public/data/model.json", "w") as f:
    json.dump(model_data, f)
print("  frontend/public/data/model.json")

# groups.json
print("Simulating groups (20k sims × 12)...")
group_tables = {
    l: simulate_group(t, valid_teams, params, home_adv, rho, n_sims=20_000)
    for l, t in WC2026_GROUPS.items()
}
thirds = [
    (df.iloc[2]["Team"], df.iloc[2]["xPts"], df.iloc[2]["xGD"])
    for df in group_tables.values()
]
thirds.sort(key=lambda x: (x[1], x[2]), reverse=True)
advancing_thirds = [t[0] for t in thirds[:8]]

groups_data = {
    "groups": {
        letter: [
            {"team": row["Team"], "xPts": row["xPts"], "xGD": row["xGD"],
             "xGF": row["xGF"], "p1": row["P(1st)"], "p2": row["P(2nd)"],
             "p3": row["P(3rd)"], "p4": row["P(4th)"]}
            for _, row in df.iterrows()
        ]
        for letter, df in group_tables.items()
    },
    "advancing_thirds": advancing_thirds,
}
with open("frontend/public/data/groups.json", "w") as f:
    json.dump(groups_data, f)
print("  frontend/public/data/groups.json")

# bracket.html
print("Building bracket...")
bracket = get_bracket(group_tables, valid_teams, params)
with open("frontend/public/data/bracket.html", "w", encoding="utf-8") as f:
    f.write(_bracket_html(bracket))
print("  frontend/public/data/bracket.html")

# tournament.json
print("Simulating tournament (8k sims)...")
tourn_df = simulate_tournament(WC2026_GROUPS, valid_teams, params, home_adv, rho, n_sims=8_000)
tourn_df = tourn_df.sort_values(["Win", "R32"], ascending=False).reset_index(drop=True)
for c in ["R32", "R16", "QF", "SF", "Final", "Win"]:
    tourn_df[c] = (tourn_df[c] * 100).round(1)

wc_teams = {t for grp in WC2026_GROUPS.values() for t in grp}
ratings_wc = ratings[ratings["team"].isin(wc_teams)].copy()
ratings_wc = ratings_wc.sort_values("spi", ascending=False).reset_index(drop=True)

tournament_data = {
    "tournament": [
        {"team": row["Team"], "r32": row["R32"], "r16": row["R16"],
         "qf": row["QF"], "sf": row["SF"], "final": row["Final"], "win": row["Win"]}
        for _, row in tourn_df.iterrows()
    ],
    "rankings": [
        {"team": row["team"], "spi": round(float(row["spi"]), 1),
         "attack": round(float(row["attack"]), 4), "defense": round(float(row["defense"]), 4)}
        for _, row in ratings_wc.iterrows()
    ],
}
with open("frontend/public/data/tournament.json", "w") as f:
    json.dump(tournament_data, f)
print("  frontend/public/data/tournament.json")

print(f"\nDone — {len(valid_teams)} teams, {len(advancing_thirds)} advancing thirds: {advancing_thirds}")
