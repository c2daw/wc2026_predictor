import { useState, useMemo } from 'react'
import { predictMatch, collapseMatrix, topScorelines } from '../utils/poisson.js'
import { getCellColor, gradFill, NAVY, RED, BLUE2 } from '../utils/colors.js'
import { tmeta, tshort } from '../utils/teamMeta.js'

const STRENGTH_RANGE = 0.9  // display scale: ±0.9 covers nearly all teams

function StrengthBar({ value, color }) {
  const pct = Math.min(Math.max(Math.abs(value) / STRENGTH_RANGE * 50, 0), 50)
  return (
    <div style={{position:'relative', height:4, background:'#E5E7EB', borderRadius:2, margin:'2px 0 5px'}}>
      {value >= 0
        ? <div style={{position:'absolute', left:'50%', width:`${pct}%`, height:'100%', background:color, borderRadius:2}} />
        : <div style={{position:'absolute', right:'50%', width:`${pct}%`, height:'100%', background:color, borderRadius:2}} />
      }
      <div style={{position:'absolute', left:'calc(50% - 0.5px)', top:-1, width:1, height:6, background:'#D1D5DB'}} />
    </div>
  )
}

function StrengthPanel({ teamA, teamB, modelData }) {
  const n = modelData.valid_teams.length
  const ai = modelData.valid_teams.indexOf(teamA)
  const bi = modelData.valid_teams.indexOf(teamB)
  const atkA = modelData.params[ai],       defA = modelData.params[n + ai]
  const atkB = modelData.params[bi],       defB = modelData.params[n + bi]
  const [flagA, codeA] = tmeta(teamA)
  const [flagB, codeB] = tmeta(teamB)

  function Row({ label, value, color }) {
    const pos = value >= 0
    return (
      <div>
        <div style={{display:'flex', justifyContent:'space-between', alignItems:'baseline', fontSize:9, marginBottom:1}}>
          <span style={{color:'#9CA3AF', fontFamily:"'Barlow Condensed',sans-serif", letterSpacing:.5, textTransform:'uppercase'}}>{label}</span>
          <span style={{fontFamily:"'Barlow Condensed',sans-serif", fontWeight:700, fontSize:11, color: pos ? color : '#9CA3AF'}}>
            {pos ? '+' : ''}{value.toFixed(3)}
          </span>
        </div>
        <StrengthBar value={value} color={color} />
      </div>
    )
  }

  function Team({ flag, code, atk, def }) {
    return (
      <div>
        <div style={{fontFamily:"'Barlow Condensed',sans-serif", fontWeight:700, fontSize:12, marginBottom:5, color:'#374151'}}>
          {flag} {code}
        </div>
        <Row label="Attack"  value={atk} color={RED}   />
        <Row label="Defense" value={def} color={BLUE2} />
      </div>
    )
  }

  return (
    <div className="panel">
      <div className="ph">Team Strength</div>
      <div className="pb">
        <Team flag={flagA} code={codeA} atk={atkA} def={defA} />
        <div className="sep" />
        <Team flag={flagB} code={codeB} atk={atkB} def={defB} />
        <div style={{fontSize:9, color:'#C4C9D4', marginTop:6, fontStyle:'italic'}}>0 = league avg · bars show ±0.9</div>
      </div>
    </div>
  )
}

function HeatmapGrid({ sm, teamA, teamB, highlight }) {
  const nc = 6
  const labels = ['0', '1', '2', '3', '4', '5+']
  const [, codeA] = tmeta(teamA)
  const [, codeB] = tmeta(teamB)

  const rowMarg = sm.map(row => row.reduce((a, b) => a + b, 0))
  const colMarg = labels.map((_, c) => sm.reduce((acc, row) => acc + row[c], 0))
  const vmax = Math.max(...sm.flat())

  function opacity(r, c) {
    if (highlight === 'all') return 1
    if (highlight === 'a') return r > c ? 1 : 0.07
    if (highlight === 'd') return r === c ? 1 : 0.07
    if (highlight === 'b') return r < c ? 1 : 0.07
    return 1
  }

  function cellType(r, c) {
    if (r > c) return 'a'
    if (r < c) return 'b'
    return 'd'
  }

  return (
    <div className="heatmap-wrap">
      <div className="heatmap-b-title">{codeB} goals →</div>
      <div className="heatmap-main">
        <div className="heatmap-a-title">← {codeA} goals</div>
        <div className="heatmap-grid-area">
          <div className="heatmap-col-header-row">
            <div className="heatmap-corner" />
            {labels.map((lbl, c) => (
              <div key={c} className="heatmap-col-header">
                <span>{lbl}</span>
                <span className="heatmap-marg">({colMarg[c].toFixed(0)}%)</span>
              </div>
            ))}
          </div>
          {sm.map((row, r) => (
            <div key={r} className="heatmap-data-row">
              <div className="heatmap-row-header">
                <span>{labels[r]}</span>
                <span className="heatmap-marg">({rowMarg[r].toFixed(0)}%)</span>
              </div>
              {row.map((val, c) => {
                const op = opacity(r, c)
                const type = cellType(r, c)
                const intensity = (val * op) / vmax
                const bg = getCellColor(type, intensity)
                const light = (val * op) >= (vmax * 0.38) && op > 0.5
                return (
                  <div
                    key={c}
                    className="heatmap-cell"
                    style={{
                      background: bg,
                      color: light ? `rgba(255,255,255,${op})` : `rgba(55,65,81,${op})`,
                    }}
                    title={`${codeA} ${r} – ${codeB} ${c}: ${val.toFixed(1)}%`}
                  >
                    {val.toFixed(1)}%
                  </div>
                )
              })}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default function MatchSimulator({ modelData }) {
  const { valid_teams, wc_groups } = modelData
  const [groupFilter, setGroupFilter] = useState('—')
  const [teamA, setTeamA] = useState('France')
  const [teamB, setTeamB] = useState('Brazil')
  const [highlight, setHighlight] = useState('all')

  const teamOptions = useMemo(() => {
    if (groupFilter === '—') return [...valid_teams].sort()
    const letter = groupFilter.split(' ')[1]
    return (wc_groups[letter] || []).filter(t => valid_teams.includes(t))
  }, [groupFilter, valid_teams, wc_groups])

  // Guard selected teams against option list changes
  const safeA = teamOptions.includes(teamA) ? teamA : teamOptions[0] || ''
  const safeB = teamOptions.includes(teamB) ? teamB : (teamOptions[1] || teamOptions[0] || '')

  const pred = useMemo(() => {
    if (!safeA || !safeB || safeA === safeB) return null
    return predictMatch(safeA, safeB, modelData)
  }, [safeA, safeB, modelData])

  const displayMatrix = useMemo(() => pred ? collapseMatrix(pred.matrix) : null, [pred])
  const scores = useMemo(() => pred ? topScorelines(pred.matrix, 8) : [], [pred])

  if (!pred || !displayMatrix) {
    return (
      <div>
        <div className="controls-bar">
          <TeamControls
            groupFilter={groupFilter} setGroupFilter={setGroupFilter}
            teamA={safeA} setTeamA={setTeamA}
            teamB={safeB} setTeamB={setTeamB}
            teamOptions={teamOptions} wc_groups={wc_groups}
          />
        </div>
        <div className="loading">Select two different teams</div>
      </div>
    )
  }

  const { lam, mu, winA, draw, winB } = pred
  const xptsA = 3 * winA + draw
  const xptsB = 3 * winB + draw
  const [flagA, codeA] = tmeta(safeA)
  const [flagB, codeB] = tmeta(safeB)

  const clsA = winA > winB + 0.05 ? 'pill-a-str' : 'pill-a-fnt'
  const clsB = winB > winA + 0.05 ? 'pill-b-str' : 'pill-b-fnt'

  const XG_MAX = 3.0
  const pctA = Math.min(lam / XG_MAX * 100, 100)
  const pctB = Math.min(mu  / XG_MAX * 100, 100)
  const maxP = scores[0]?.p || 1

  return (
    <div>
      <div className="controls-bar">
        <TeamControls
          groupFilter={groupFilter} setGroupFilter={setGroupFilter}
          teamA={safeA} setTeamA={setTeamA}
          teamB={safeB} setTeamB={setTeamB}
          teamOptions={teamOptions} wc_groups={wc_groups}
        />
      </div>

      <div className="match-layout">
        {/* Left: heatmap */}
        <div>
          <div className="panel">
            <div className="ph">
              Score Probability Matrix
              <span className="ph-sub">Dixon-Coles Poisson · rho correction</span>
            </div>
            <div className="pb">
              <div className="teams-bar">
                <div className="tbadge-home">
                  <div className="t-flag-code">{flagA} {codeA}</div>
                </div>
                <div className="vs-pill">VS</div>
                <div className="tbadge-away">
                  <div className="t-flag-code">{codeB} {flagB}</div>
                </div>
              </div>

              <div className="hl-row">
                {[
                  { id: 'all', label: 'All' },
                  { id: 'a',   label: `Win ${codeA}` },
                  { id: 'd',   label: 'Draw' },
                  { id: 'b',   label: `Win ${codeB}` },
                ].map(h => (
                  <button
                    key={h.id}
                    className={`hl-btn${highlight === h.id ? ' active' : ''}`}
                    onClick={() => setHighlight(h.id)}
                  >
                    {h.label}
                  </button>
                ))}
              </div>

              <HeatmapGrid sm={displayMatrix} teamA={safeA} teamB={safeB} highlight={highlight} />

              <div className="outcome-row" style={{marginTop: 12}}>
                <div className={`outcome-pill ${clsA}`}>
                  <span className="pct">{(winA * 100).toFixed(0)}%</span>
                  <span className="lbl">WIN {codeA}</span>
                </div>
                <div className="outcome-pill pill-draw">
                  <span className="pct">{(draw * 100).toFixed(0)}%</span>
                  <span className="lbl">DRAW</span>
                </div>
                <div className={`outcome-pill ${clsB}`}>
                  <span className="pct">{(winB * 100).toFixed(0)}%</span>
                  <span className="lbl">WIN {codeB}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right: stats */}
        <div>
          <StrengthPanel teamA={safeA} teamB={safeB} modelData={modelData} />

          <div className="panel">
            <div className="ph">xG Breakdown</div>
            <div className="pb">
              <div className="xg-row">
                <span className="xg-lbl">{flagA} {codeA}</span>
                <span className="xg-val">{lam.toFixed(2)}</span>
              </div>
              <div className="xg-track">
                <div className="xg-fill" style={{width: `${pctA.toFixed(0)}%`, background: RED}} />
              </div>
              <div className="xg-row">
                <span className="xg-lbl">{flagB} {codeB}</span>
                <span className="xg-val">{mu.toFixed(2)}</span>
              </div>
              <div className="xg-track">
                <div className="xg-fill" style={{width: `${pctB.toFixed(0)}%`, background: BLUE2}} />
              </div>
              <div className="sep" />
              <div className="xg-row">
                <span className="xg-lbl" style={{fontSize:10,color:'#9CA3AF'}}>xG diff</span>
                <span style={{fontFamily:"'Barlow Condensed',sans-serif",fontWeight:700,fontSize:14,color: lam >= mu ? RED : BLUE2}}>
                  {(lam - mu) > 0 ? '+' : ''}{(lam - mu).toFixed(2)}
                </span>
              </div>
            </div>
          </div>

          <div className="panel">
            <div className="ph">Expected Points</div>
            <div className="pb">
              <div className="xp-grid">
                <div className="xp-box" style={{background:'rgba(214,0,0,.08)'}}>
                  <div className="xp-lbl" style={{color:RED}}>{codeA}</div>
                  <div className="xp-val" style={{color:RED}}>{xptsA.toFixed(2)}</div>
                  <div className="xp-sub">xPts</div>
                </div>
                <div className="xp-box" style={{background:'rgba(25,35,124,.08)'}}>
                  <div className="xp-lbl" style={{color:NAVY}}>{codeB}</div>
                  <div className="xp-val" style={{color:NAVY}}>{xptsB.toFixed(2)}</div>
                  <div className="xp-sub">xPts</div>
                </div>
              </div>
            </div>
          </div>

          <div className="panel">
            <div className="ph">Most Likely Scores</div>
            <div className="pb">
              {scores.map(({ x, y, p }, i) => {
                const intensity = (p / maxP) ** 0.7
                let bg
                if (x > y) bg = `rgba(214,0,0,${(intensity * 0.35).toFixed(2)})`
                else if (x < y) bg = `rgba(48,79,255,${(intensity * 0.30).toFixed(2)})`
                else bg = `rgba(131,39,128,${(intensity * 0.28).toFixed(2)})`
                return (
                  <div key={i} className="score-row" style={{background: bg}}>
                    <span className="score-lbl">{x} – {y}</span>
                    <span className="score-pct">{(p * 100).toFixed(1)}%</span>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function TeamControls({ groupFilter, setGroupFilter, teamA, setTeamA, teamB, setTeamB, teamOptions, wc_groups }) {
  const groupKeys = Object.keys(wc_groups)
  return (
    <>
      <div className="ctrl-group">
        <label className="ctrl-label">Filter by group</label>
        <select className="ctrl-select" value={groupFilter} onChange={e => setGroupFilter(e.target.value)}>
          <option value="—">— All teams</option>
          {groupKeys.map(l => <option key={l} value={`Group ${l}`}>Group {l}</option>)}
        </select>
      </div>
      <div className="ctrl-group">
        <label className="ctrl-label">Team A</label>
        <select className="ctrl-select" value={teamA} onChange={e => setTeamA(e.target.value)}>
          {teamOptions.map(t => <option key={t} value={t}>{t}</option>)}
        </select>
      </div>
      <div className="ctrl-group">
        <label className="ctrl-label">Team B</label>
        <select className="ctrl-select" value={teamB} onChange={e => setTeamB(e.target.value)}>
          {teamOptions.map(t => <option key={t} value={t}>{t}</option>)}
        </select>
      </div>
    </>
  )
}
