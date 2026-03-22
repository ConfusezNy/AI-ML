import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import { useState, useEffect } from 'react'
import Dashboard from './pages/Dashboard'
import Compare from './pages/Compare'
import AIPredict from './pages/AIPredict'
import Alignment from './pages/Alignment'
import './App.css'

function App() {
  const [data, setData] = useState(null)
  const [features, setFeatures] = useState(null)
  const [totalSeats, setTotalSeats] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      fetch('/data/model_results.json').then(r => r.json()),
      fetch('/data/ml_features.json').then(r => r.json()),
      fetch('/data/total_seats.json').then(r => r.json()),
    ]).then(([modelData, featData, seatsData]) => {
      setData(modelData)
      setFeatures(featData)
      setTotalSeats(seatsData)
      setLoading(false)
    }).catch(err => {
      console.error('Error loading data:', err)
      setLoading(false)
    })
  }, [])

  if (loading) {
    return (
      <div className="loading">
        <div className="loading-spinner"></div>
        กำลังโหลดข้อมูล...
      </div>
    )
  }

  return (
    <BrowserRouter>
      <div className="app">
        <aside className="sidebar">
          <div className="sidebar-logo">
            <div className="icon">🗳️</div>
            <div>
              <h1>Election AI</h1>
              <span className="subtitle">ทำนายผลเลือกตั้งไทย</span>
            </div>
          </div>
          <nav className="nav-links">
            <NavLink to="/" end className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>
              <span className="nav-icon">📊</span>
              <span>Dashboard</span>
            </NavLink>
            <NavLink to="/compare" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>
              <span className="nav-icon">⚖️</span>
              <span>เปรียบเทียบ</span>
            </NavLink>
            <NavLink to="/predict" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>
              <span className="nav-icon">🤖</span>
              <span>AI Prediction</span>
            </NavLink>
            <NavLink to="/alignment" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>
              <span className="nav-icon">🏛️</span>
              <span>แนวทางการเมือง</span>
            </NavLink>
          </nav>
          <div style={{marginTop:'auto', padding:'12px', fontSize:'11px', color:'var(--text-muted)'}}>
            <div>Data: 2562, 2566, 2569</div>
            <div>AI: LSTM · BiLSTM · Transformer</div>
          </div>
        </aside>
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard data={data} features={features} totalSeats={totalSeats} />} />
            <Route path="/compare" element={<Compare features={features} totalSeats={totalSeats} />} />
            <Route path="/predict" element={<AIPredict data={data} features={features} />} />
            <Route path="/alignment" element={<Alignment features={features} totalSeats={totalSeats} />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App
