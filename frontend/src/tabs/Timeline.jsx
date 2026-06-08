import { useState, useRef } from 'react'
import { tmeta, tshort } from '../utils/teamMeta.js'

const PALETTE = [
  '#D60000','#304FFF','#059669','#D97706',
  '#7C3AED','#0891B2','#DB2777','#65A30D',
  '#92400E','#1D4ED8',
]

const DEFAULT_TEAMS = ['Austria', 'Norway', 'France', 'England', 'Germany', 'Brazil']

const METRICS = [
  { id: 'spi',     label: 'SPI' },
  { id: 'attack',  label: 'Attack' },
  { id: 'defense', label: 'Defense' },
]

function fmt(v, metric) {
  if (v === null || v === undefined) return '—'
  if (metric === 'spi') return v.toFixed(1)
  return (v >= 0 ? '+' : '') + v.toFixed(3)
}

function LineChart({ historyData, selectedTeams, metric }) {
  const [hoverIdx, setHoverIdx] = useState(null)
  const svgRef = useRef()

  const { snapshots, teams } = historyData
  const W = 900, H = 280
  const PAD = { l: 48, r: 16, t: 16, b: 34 }
  const pw = W - PAD.l - PAD.r
  const ph = H - PAD.t - PAD.b

  const allVals = selectedTeams
    .flatMap(t => (teams[t]?.[metric] || []).filter(v => v !== null))
  if (allVals.length === 0) return (
    <div style={{textAlign:'center', color:'#9CA3AF', padding:'60px 0', fontFamily:"'Barlow Condensed',sans-serif", fontSize:13, letterSpacing:.5}}>
      NO DATA
    </div>
  )

  const yMin = Math.min(...allVals)
  const yMax = Math.max(...allVals)
  const yPad = Math.max((yMax - yMin) * 0.15, 0.05)
  const yLo = yMin - yPad, yHi = yMax + yPad

  function xPos(i) { return PAD.l + (i / Math.max(snapshots.length - 1, 1)) * pw }
  function yPos(v) { return PAD.t + ph - ((v - yLo) / (yHi - yLo)) * ph }

  function teamPath(team) {
    const vals = teams[team]?.[metric]
    if (!vals) return ''
    let d = '', pen = true
    for (let i = 0; i < snapshots.length; i++) {
      if (vals[i] === null) { pen = true; continue }
      const x = xPos(i).toFixed(1), y = yPos(vals[i]).toFixed(1)
      d += pen ? `M${x},${y}` : `L${x},${y}`
      pen = false
    }
    return d
  }

  // Y-axis: 5 evenly spaced ticks
  const yTicks = Array.from({ length: 5 }, (_, i) => yLo + (yHi - yLo) * i / 4)

  // X-axis: Jan-01 year markers only
  const xMarks = snapshots
    .map((s, i) => ({ i, label: s.slice(0, 4), isJan: s.slice(5) === '01-01' }))
    .filter(m => m.isJan || m.i === 0)

  function handleMouseMove(e) {
    const rect = svgRef.current.getBoundingClientRect()
    const svgX = ((e.clientX - rect.left) / rect.width) * W
    const idx = Math.round((svgX - PAD.l) / pw * (snapshots.length - 1))
    setHoverIdx(Math.max(0, Math.min(snapshots.length - 1, idx)))
  }

  return (
    <div style={{ position: 'relative' }}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: '100%', height: 'auto', display: 'block' }}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoverIdx(null)}
      >
        {/* Horizontal grid */}
        {yTicks.map((v, i) => (
          <line key={i} x1={PAD.l} x2={PAD.l + pw} y1={yPos(v)} y2={yPos(v)}
            stroke="#F3F4F6" strokeWidth="1" />
        ))}

        {/* Y-axis labels */}
        {yTicks.map((v, i) => (
          <text key={i} x={PAD.l - 5} y={yPos(v)} textAnchor="end" dominantBaseline="middle"
            fontSize="9" fill="#9CA3AF" fontFamily="'Barlow Condensed', sans-serif">
            {fmt(v, metric)}
          </text>
        ))}

        {/* X-axis year labels */}
        {xMarks.map(({ i, label }) => (
          <g key={i}>
            <line x1={xPos(i)} x2={xPos(i)} y1={PAD.t + ph} y2={PAD.t + ph + 4} stroke="#E5E7EB" />
            <text x={xPos(i)} y={H - PAD.b + 14} textAnchor="middle"
              fontSize="9" fill="#9CA3AF" fontFamily="'Barlow Condensed', sans-serif">
              {label}
            </text>
          </g>
        ))}

        {/* Y-axis baseline */}
        <line x1={PAD.l} x2={PAD.l} y1={PAD.t} y2={PAD.t + ph} stroke="#E5E7EB" />
        <line x1={PAD.l} x2={PAD.l + pw} y1={PAD.t + ph} y2={PAD.t + ph} stroke="#E5E7EB" />

        {/* Zero line for attack/defense */}
        {metric !== 'spi' && yLo < 0 && yHi > 0 && (
          <line x1={PAD.l} x2={PAD.l + pw} y1={yPos(0)} y2={yPos(0)}
            stroke="#D1D5DB" strokeWidth="1" strokeDasharray="4,3" />
        )}

        {/* Team lines */}
        {selectedTeams.map((team, ti) => (
          <path key={team} d={teamPath(team)} fill="none"
            stroke={PALETTE[ti % PALETTE.length]}
            strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
        ))}

        {/* Hover vertical line */}
        {hoverIdx !== null && (
          <line x1={xPos(hoverIdx)} x2={xPos(hoverIdx)} y1={PAD.t} y2={PAD.t + ph}
            stroke="#9CA3AF" strokeWidth="1" strokeDasharray="3,3" />
        )}

        {/* Hover dots */}
        {hoverIdx !== null && selectedTeams.map((team, ti) => {
          const v = teams[team]?.[metric]?.[hoverIdx]
          if (v === null || v === undefined) return null
          return (
            <circle key={team} cx={xPos(hoverIdx)} cy={yPos(v)} r="3.5"
              fill={PALETTE[ti % PALETTE.length]} />
          )
        })}
      </svg>

      {/* Hover tooltip */}
      {hoverIdx !== null && (
        <div style={{
          position: 'absolute', top: 8, right: 8,
          background: 'white', border: '.5px solid #E5E7EB', borderRadius: 4,
          padding: '8px 10px', pointerEvents: 'none',
          boxShadow: '0 2px 8px rgba(0,0,0,.08)',
          minWidth: 140,
        }}>
          <div style={{ fontFamily: "'Barlow Condensed', sans-serif", fontSize: 10, color: '#9CA3AF', marginBottom: 6, letterSpacing: .5 }}>
            {snapshots[hoverIdx]}
          </div>
          {selectedTeams.map((team, ti) => {
            const v = teams[team]?.[metric]?.[hoverIdx]
            const [flag] = tmeta(team)
            return (
              <div key={team} style={{ display: 'flex', justifyContent: 'space-between', gap: 10, marginBottom: 3 }}>
                <span style={{ fontFamily: "'Barlow Condensed', sans-serif", fontSize: 11, color: '#374151' }}>
                  <span style={{ color: PALETTE[ti % PALETTE.length] }}>●</span> {flag} {tshort(team)}
                </span>
                <span style={{ fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 700, fontSize: 11, color: PALETTE[ti % PALETTE.length] }}>
                  {fmt(v, metric)}
                </span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

export default function Timeline({ historyData }) {
  const [selected, setSelected] = useState(DEFAULT_TEAMS)
  const [metric, setMetric] = useState('spi')

  if (!historyData) return (
    <div className="loading">
      Run <code>python precompute.py</code> to generate timeline data
    </div>
  )

  const allTeams = Object.keys(historyData.teams).sort()
  const available = allTeams.filter(t => !selected.includes(t))

  function addTeam(t) { if (t && !selected.includes(t)) setSelected([...selected, t]) }
  function removeTeam(t) { setSelected(selected.filter(s => s !== t)) }

  return (
    <div>
      {/* Controls */}
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 12, flexWrap: 'wrap' }}>
        <div className="hl-row" style={{ margin: 0 }}>
          {METRICS.map(m => (
            <button key={m.id} className={`hl-btn${metric === m.id ? ' active' : ''}`}
              onClick={() => setMetric(m.id)}>
              {m.label}
            </button>
          ))}
        </div>

        <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
          {selected.map((t, i) => {
            const [flag] = tmeta(t)
            const color = PALETTE[i % PALETTE.length]
            return (
              <span key={t} style={{
                display: 'inline-flex', alignItems: 'center', gap: 4,
                background: color + '18',
                border: `1px solid ${color}55`,
                borderRadius: 3, padding: '3px 8px',
                fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 600, fontSize: 11,
                color,
              }}>
                {flag} {tshort(t)}
                <button onClick={() => removeTeam(t)} style={{
                  background: 'none', border: 'none', cursor: 'pointer', padding: 0,
                  color, fontWeight: 900, fontSize: 13, lineHeight: 1, marginLeft: 2,
                }}>×</button>
              </span>
            )
          })}

          <select
            className="ctrl-select"
            value=""
            onChange={e => addTeam(e.target.value)}
            style={{ fontSize: 11, padding: '3px 8px', minHeight: 26 }}
          >
            <option value="">+ Add team</option>
            {available.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>
      </div>

      {/* Chart */}
      <div className="panel">
        <div className="ph">
          {METRICS.find(m => m.id === metric)?.label} Rating over Time
          <span className="ph-sub">semi-annual snapshots · 2016–present</span>
        </div>
        <div className="pb" style={{ paddingTop: 8 }}>
          {selected.length === 0
            ? <div style={{ textAlign: 'center', color: '#9CA3AF', padding: '48px 0', fontFamily: "'Barlow Condensed',sans-serif", fontSize: 13, letterSpacing: .5 }}>ADD TEAMS TO COMPARE</div>
            : <LineChart historyData={historyData} selectedTeams={selected} metric={metric} />
          }
        </div>
      </div>

      <div style={{ fontSize: 10, color: '#9CA3AF', marginTop: 6 }}>
        Model re-fitted independently at each snapshot using all available data up to that date · 0 = league average for attack &amp; defense
      </div>
    </div>
  )
}
