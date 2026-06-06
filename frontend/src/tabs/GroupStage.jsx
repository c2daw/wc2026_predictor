import { gradFill, NAVY, RED, BLUE2, GREEN, LIME } from '../utils/colors.js'
import { tmeta, tshort } from '../utils/teamMeta.js'

function GroupTable({ letter, rows, advancingThirds }) {
  const teams = rows.map(r => r.team)
  const sub = teams.map(t => tmeta(t)[1]).join(' · ')

  const p1max = Math.max(...rows.map(r => r.p1), 1)
  const p2max = Math.max(...rows.map(r => r.p2), 1)
  const p3max = Math.max(...rows.map(r => r.p3), 1)
  const xpmax = Math.max(...rows.map(r => r.xPts), 1)

  function stripe(i, t) {
    if (i < 2) return LIME
    return advancingThirds.includes(t) ? '#D1FAE5' : '#E5E7EB'
  }
  function stripeFc(i, t) {
    if (i < 2) return NAVY
    return advancingThirds.includes(t) ? '#065F46' : '#9CA3AF'
  }

  return (
    <div className="grp-card">
      <div className="grp-hdr">
        <span className="grp-letter">GROUP {letter}</span>
        <span className="grp-sub">{sub}</span>
      </div>
      <table className="grp-table">
        <thead>
          <tr>
            <th></th>
            <th>Team</th>
            <th>xPts</th>
            <th>xGF</th>
            <th>xGD</th>
            <th>1st</th>
            <th>2nd</th>
            <th>3rd</th>
            <th>4th</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={row.team}>
              <td style={{background: stripe(i, row.team), padding: 0, width: 4}} />
              <td style={{color: stripeFc(i, row.team), fontWeight: i < 2 ? 700 : 400}}>
                {tshort(row.team)}
              </td>
              <td style={{background: gradFill(row.xPts, xpmax, NAVY), color: row.xPts > xpmax * 0.6 ? 'white' : NAVY}}>
                {row.xPts.toFixed(1)}
              </td>
              <td>{row.xGF.toFixed(1)}</td>
              <td style={{color: row.xGD >= 0 ? '#065F46' : '#991B1B'}}>
                {row.xGD > 0 ? '+' : ''}{row.xGD.toFixed(1)}
              </td>
              <td style={{background: gradFill(row.p1, p1max, RED), color: row.p1 > p1max * 0.6 ? 'white' : '#8B0000'}}>
                {row.p1.toFixed(0)}%
              </td>
              <td style={{background: gradFill(row.p2, p2max, BLUE2), color: row.p2 > p2max * 0.6 ? 'white' : '#1A237E'}}>
                {row.p2.toFixed(0)}%
              </td>
              <td style={{background: gradFill(row.p3, p3max, GREEN), color: row.p3 > p3max * 0.6 ? 'white' : '#1B5E20'}}>
                {row.p3.toFixed(0)}%
              </td>
              <td style={{color: '#9CA3AF'}}>{row.p4.toFixed(0)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function GroupStage({ groupsData }) {
  if (!groupsData) return <div className="loading">Loading groups…</div>

  const { groups, advancing_thirds: advancingThirds } = groupsData
  const letters = Object.keys(groups)

  return (
    <div>
      <div className="grp-legend">
        <span><span className="lq" />&nbsp;Qualified (1st &amp; 2nd)</span>
        <span><span className="l3a" />&nbsp;Top-8 3rd place (advancing)</span>
        <span><span className="l3" />&nbsp;Eliminated</span>
        <span style={{marginLeft: 'auto', fontSize: 9, textTransform: 'uppercase', letterSpacing: .5, color: '#9CA3AF'}}>
          xPts · xGF · xGD = expected values across 3 group games &nbsp;·&nbsp; 1st/2nd/3rd = finish probability
        </span>
      </div>
      <div className="groups-grid">
        {letters.map(letter => (
          <GroupTable
            key={letter}
            letter={letter}
            rows={groups[letter]}
            advancingThirds={advancingThirds}
          />
        ))}
      </div>
    </div>
  )
}
