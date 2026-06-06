import { gradFill, NAVY, RED, BLUE2, GREEN, LIME } from '../utils/colors.js'

const DATA = [
  { tournament: 'WC 2022',   n: 64, exact: 3,  gd: 9, tend: 14, miss: 38 },
  { tournament: 'Euro 2024', n: 51, exact: 12, gd: 6, tend: 8,  miss: 25 },
]

DATA.forEach(d => { d.total = 4 * d.exact + 2 * d.gd + d.tend; d.ppg = d.total / d.n })

export default function Backtest() {
  const exMax = Math.max(...DATA.map(d => d.exact))
  const gdMax = Math.max(...DATA.map(d => d.gd))
  const teMax = Math.max(...DATA.map(d => d.tend))
  const toMax = Math.max(...DATA.map(d => d.total))
  const pgMax = Math.max(...DATA.map(d => d.ppg))

  return (
    <div>
      <div className="grp-card" style={{marginBottom: 0}}>
        <div className="grp-hdr">
          <span className="grp-letter">Backtest</span>
          <span className="grp-sub">Round(xG) · 4 pts exact · 2 pts GD · 1 pt tendency</span>
        </div>
      </div>
      <table className="data-table" style={{borderTop: 'none'}}>
        <thead>
          <tr>
            <th style={{textAlign:'left', color: LIME}}>Tournament</th>
            <th>Games</th>
            <th>Exact</th>
            <th>Goal Diff</th>
            <th>Tendency</th>
            <th>Miss</th>
            <th>Total Pts</th>
            <th>Pts / Game</th>
          </tr>
        </thead>
        <tbody>
          {DATA.map(d => (
            <tr key={d.tournament}>
              <td style={{textAlign:'left', fontWeight: 600}}>{d.tournament}</td>
              <td>{d.n}</td>
              <td style={{background: gradFill(d.exact, exMax, LIME, 0.12, 0.55)}}>{d.exact}</td>
              <td style={{background: gradFill(d.gd,    gdMax, BLUE2,0.08, 0.40)}}>{d.gd}</td>
              <td style={{background: gradFill(d.tend,  teMax, GREEN,0.08, 0.40)}}>{d.tend}</td>
              <td style={{background: '#F3F4F6', color: '#9CA3AF'}}>{d.miss}</td>
              <td style={{background: gradFill(d.total, toMax, RED,  0.08, 0.60), fontWeight: 700}}>{d.total}</td>
              <td style={{background: gradFill(d.ppg,   pgMax, RED,  0.08, 0.60)}}>{d.ppg.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
