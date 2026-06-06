import { gradFill, NAVY, RED, BLUE2, GREEN, LIME } from '../utils/colors.js'

export default function KOBracket({ tournamentData }) {
  if (!tournamentData) return <div className="loading">Loading bracket…</div>

  const { tournament, rankings } = tournamentData

  const winMax   = Math.max(...tournament.map(r => r.win), 1)
  const finalMax = Math.max(...tournament.map(r => r.final), 1)
  const sfMax    = Math.max(...tournament.map(r => r.sf), 1)
  const spiMax   = Math.max(...rankings.map(r => r.spi), 1)

  function spiFill(v) {
    if (v <= 50) return '#F3F4F6'
    const t = Math.min((v - 50) / Math.max(spiMax - 50, 1e-6), 1)
    const a = 0.06 + 0.76 * t
    return `rgba(214,0,0,${a.toFixed(2)})`
  }
  function spiFc(v) {
    return (v - 50) / Math.max(spiMax - 50, 1e-6) > 0.55 ? 'white' : NAVY
  }

  return (
    <div>
      <p style={{color:'#9CA3AF',fontSize:'0.82rem',marginBottom:12}}>
        Most-likely bracket based on group simulation. Top 2 per group + 8 best 3rd-place teams advance to R32.
        Official FIFA WC 2026 bracket structure. Hover over any fixture in the bracket to see the score matrix.
      </p>

      <iframe
        src="/data/bracket.html"
        className="bracket-frame"
        height={440}
        title="KO Bracket"
      />

      <div className="bkt-legend">
        <div className="bl-item">
          <div style={{width:14,height:9,background:'#19237C',borderRadius:2}} />
          Predicted winner
        </div>
        <div className="bl-item">
          <div style={{width:14,height:9,border:'1.5px solid #AFEA00',borderRadius:2}} />
          Final
        </div>
        <div style={{marginLeft:'auto',fontSize:9,textTransform:'uppercase',letterSpacing:.5,color:'#6B7280'}}>
          % = win probability &nbsp;·&nbsp; score = xG
        </div>
      </div>

      <hr style={{border:'none',borderTop:'1px solid #E5E7EB',margin:'16px 0'}} />

      <div className="ko-grid">
        <div>
          <div className="grp-card">
            <div className="grp-hdr">
              <span className="grp-letter">Round-by-Round</span>
              <span className="grp-sub">All 48 teams · sorted by Win %</span>
            </div>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Team</th>
                <th>R32</th>
                <th>R16</th>
                <th>QF</th>
                <th>SF</th>
                <th>Final</th>
                <th>🏆 Win</th>
              </tr>
            </thead>
            <tbody>
              {tournament.map((row, i) => (
                <tr key={row.team}>
                  <td>{row.team}</td>
                  <td>{row.r32.toFixed(0)}%</td>
                  <td>{row.r16.toFixed(0)}%</td>
                  <td>{row.qf.toFixed(0)}%</td>
                  <td style={{background: gradFill(row.sf,    sfMax,    GREEN, 0.03, 0.50)}}>
                    {row.sf.toFixed(1)}%
                  </td>
                  <td style={{background: gradFill(row.final, finalMax, BLUE2, 0.04, 0.65)}}>
                    {row.final.toFixed(1)}%
                  </td>
                  <td style={{background: gradFill(row.win,   winMax,   RED,   0.05, 0.82),
                               color: row.win > winMax * 0.55 ? 'white' : '#374151',
                               fontWeight: 700}}>
                    {row.win.toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div>
          <div className="grp-card">
            <div className="grp-hdr">
              <span className="grp-letter">SPI Rankings</span>
              <span className="grp-sub">WC 2026 teams</span>
            </div>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>#</th>
                <th style={{textAlign:'left', color: LIME}}>Team</th>
                <th>SPI</th>
                <th>Attack</th>
                <th>Defense</th>
              </tr>
            </thead>
            <tbody>
              {rankings.map((row, i) => (
                <tr key={row.team}>
                  <td style={{color:'#9CA3AF'}}>{i + 1}</td>
                  <td style={{textAlign:'left'}}>{row.team}</td>
                  <td style={{background: spiFill(row.spi), color: spiFc(row.spi), fontWeight: 700}}>
                    {row.spi.toFixed(1)}
                  </td>
                  <td style={{color: row.attack >= 0 ? '#065F46' : '#991B1B'}}>
                    {row.attack > 0 ? '+' : ''}{row.attack.toFixed(3)}
                  </td>
                  <td style={{color: row.defense >= 0 ? '#065F46' : '#991B1B'}}>
                    {row.defense > 0 ? '+' : ''}{row.defense.toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
