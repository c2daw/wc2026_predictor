import { useState, useEffect } from 'react'
import MatchSimulator from './tabs/MatchSimulator.jsx'
import GroupStage from './tabs/GroupStage.jsx'
import KOBracket from './tabs/KOBracket.jsx'
import Backtest from './tabs/Backtest.jsx'

const TABS = [
  { id: 'match',  label: '🔮 Match Simulator' },
  { id: 'groups', label: '🏆 Group Stage' },
  { id: 'ko',     label: '🗓 KO Bracket' },
  { id: 'bt',     label: '📈 Backtest' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('match')
  const [modelData, setModelData] = useState(null)
  const [groupsData, setGroupsData] = useState(null)
  const [tournamentData, setTournamentData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      fetch('/data/model.json').then(r => r.json()),
      fetch('/data/groups.json').then(r => r.json()),
      fetch('/data/tournament.json').then(r => r.json()),
    ]).then(([model, groups, tournament]) => {
      setModelData(model)
      setGroupsData(groups)
      setTournamentData(tournament)
    }).catch(e => setError(e.message))
  }, [])

  return (
    <div>
      <header className="app-header">
        <div className="app-logo">
          FIFA <span>World Cup</span> 2026 <span style={{fontSize:11,color:'rgba(255,255,255,.35)',fontWeight:400,letterSpacing:0}}>· Predictor</span>
        </div>
        <div className="app-sub">Dixon-Coles Poisson · time-decay · martj42/international_results</div>
      </header>

      <div className="tab-bar">
        {TABS.map(t => (
          <button
            key={t.id}
            className={`tab-btn${activeTab === t.id ? ' active' : ''}`}
            onClick={() => setActiveTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="tab-content">
        {error && <div className="loading" style={{color:'#D60000'}}>Error loading data: {error}</div>}
        {!error && !modelData && <div className="loading">Loading model…</div>}
        {!error && modelData && (
          <>
            {activeTab === 'match'  && <MatchSimulator modelData={modelData} />}
            {activeTab === 'groups' && <GroupStage groupsData={groupsData} modelData={modelData} />}
            {activeTab === 'ko'     && <KOBracket tournamentData={tournamentData} />}
            {activeTab === 'bt'     && <Backtest />}
          </>
        )}
      </div>
    </div>
  )
}
