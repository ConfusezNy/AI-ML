import { useState, useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  LineChart, Line, CartesianGrid, Legend
} from 'recharts'

const COLORS = {
  progressive: '#FF6B00',
  populist: '#E3232C',
  conservative: '#3B82F6',
  others: '#888888',
}

const LABELS = {
  progressive: 'ก้าวหน้า',
  populist: 'ประชานิยม',
  conservative: 'อนุรักษ์นิยม',
  others: 'อื่นๆ',
}

export default function AIPredict({ data, features }) {
  const [selectedModel, setSelectedModel] = useState('Transformer')
  
  const modelMetrics = data?.model_comparison
  const predictions = data?.predictions_2573
  const history = data?.training_history

  const predData = useMemo(() => {
    if (!predictions?.[selectedModel]) return []
    return predictions[selectedModel].map(p => ({
      ...p,
      name: p.province_name.replace('กรุงเทพมหานคร', 'กทม.'),
    }))
  }, [predictions, selectedModel])

  const summaryData = useMemo(() => {
    if (!predData.length || !features?.province_features?.['2569']) return {}
    const pf2569 = features.province_features['2569']
    const counts = { progressive: 0, populist: 0, conservative: 0, others: 0 }
    const seats = { progressive: 0, populist: 0, conservative: 0, others: 0 }
    predData.forEach(p => {
      const align = p.predicted_winner || 'others'
      counts[align] = (counts[align] || 0) + 1
      const zones = pf2569[p.province_id]?.num_zones || 0
      seats[align] = (seats[align] || 0) + zones
    })
    return { counts, seats, totalSeats: Object.values(seats).reduce((a,b) => a+b, 0) }
  }, [predData, features])

  const lossData = useMemo(() => {
    if (!history) return []
    const h = history[selectedModel]
    if (!h) return []
    return h.train_loss.map((tl, i) => ({
      epoch: i + 1,
      train: +tl.toFixed(3),
      val: +h.val_loss[i].toFixed(3),
    })).filter((_, i) => i % 2 === 0)
  }, [history, selectedModel])

  const radarData = useMemo(() => {
    if (!modelMetrics) return []
    return Object.entries(modelMetrics).map(([name, m]) => ({
      model: name,
      Accuracy: m.accuracy,
      'R²': m.r2 * 100,
      'MAE⁻¹': Math.max(0, 100 - m.mae * 10),
    }))
  }, [modelMetrics])

  if (!modelMetrics) return <div className="loading"><div className="loading-spinner"></div>Loading...</div>

  const metricsTable = Object.entries(modelMetrics).map(([name, m]) => ({
    name,
    accuracy: m.accuracy.toFixed(1),
    mae: m.mae.toFixed(2),
    rmse: m.rmse.toFixed(2),
    r2: m.r2.toFixed(3),
    isBest: name === data.best_model,
  }))

  const pieData = summaryData.seats ? Object.entries(summaryData.seats).filter(([,v]) => v > 0).map(([k, v]) => ({
    name: LABELS[k], value: v, fill: COLORS[k]
  })) : []

  return (
    <div>
      <div className="page-header">
        <h2>🤖 AI Prediction — ทำนายผลเลือกตั้ง 2573</h2>
        <p>เปรียบเทียบ 3 Models: LSTM, BiLSTM, Transformer</p>
      </div>

      {/* Model Comparison Table */}
      <div className="card" style={{marginBottom: 24}}>
        <div className="card-title">📊 ผลเปรียบเทียบ 3 Models</div>
        <table className="data-table">
          <thead>
            <tr><th>Model</th><th>Accuracy</th><th>MAE</th><th>RMSE</th><th>R²</th><th></th></tr>
          </thead>
          <tbody>
            {metricsTable.map(m => (
              <tr key={m.name} style={m.isBest ? {background:'rgba(102,126,234,0.1)'} : {}}>
                <td style={{fontWeight:700}}>{m.name}</td>
                <td style={{color: m.isBest ? '#22c55e' : undefined}}>{m.accuracy}%</td>
                <td>{m.mae}</td>
                <td>{m.rmse}</td>
                <td>{m.r2}</td>
                <td>{m.isBest ? '🏆 Best' : ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Model Selector */}
      <div className="model-selector">
        {['LSTM', 'BiLSTM', 'Transformer'].map(m => (
          <button key={m} className={`model-btn ${selectedModel === m ? 'active' : ''}`}
                  onClick={() => setSelectedModel(m)}>
            {m} {m === data.best_model ? '🏆' : ''}
          </button>
        ))}
      </div>

      {/* Predicted Seats */}
      {summaryData.seats && (
        <>
        <div className="card" style={{marginBottom: 24, padding: '20px 24px'}}>
          <div className="card-title">🔮 ทำนายที่นั่ง 2573 ({selectedModel}) — {summaryData.totalSeats} เขต</div>
          <div style={{display:'flex', gap:16, flexWrap:'wrap', marginTop: 16}}>
            {Object.entries(summaryData.seats).filter(([,v]) => v > 0).map(([align, seatCount]) => (
              <div key={align} style={{flex:1, minWidth: 160, background:'rgba(255,255,255,0.03)', borderRadius:12, padding:'16px 20px', borderLeft:`4px solid ${COLORS[align]}`}}>
                <div style={{fontSize:12, color:'var(--text-muted)', marginBottom:4}}>{LABELS[align]}</div>
                <div style={{fontSize:36, fontWeight:800, color:COLORS[align]}}>{seatCount}</div>
                <div style={{fontSize:12, color:'var(--text-muted)'}}>ที่นั่งเขต ({summaryData.counts[align]} จังหวัด)</div>
              </div>
            ))}
          </div>
        </div>

        <div className="stats-grid">
          {Object.entries(summaryData.counts).filter(([,v]) => v > 0).map(([align, count]) => (
            <div className={`stat-card ${align}`} key={align}>
              <div className="stat-label">{LABELS[align]}</div>
              <div className="stat-value" style={{color: COLORS[align]}}>{count}</div>
              <div className="stat-change">จังหวัด ({(count/77*100).toFixed(0)}%)</div>
            </div>
          ))}
        </div>
        </>
      )}

      <div className="charts-grid">
        <div className="card">
          <div className="card-title">Training Loss ({selectedModel})</div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={lossData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2d3652" />
              <XAxis dataKey="epoch" stroke="#64748b" fontSize={11} />
              <YAxis stroke="#64748b" fontSize={11} />
              <Tooltip contentStyle={{background:'#1a2035',border:'1px solid #2d3652',borderRadius:'8px',color:'#f0f4ff'}} />
              <Legend />
              <Line type="monotone" dataKey="train" name="Train Loss" stroke="#818cf8" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="val" name="Val Loss" stroke="#f472b6" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="card-title">Model Performance Radar</div>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#2d3652" />
              <PolarAngleAxis dataKey="model" stroke="#94a3b8" fontSize={12} />
              <PolarRadiusAxis domain={[0, 100]} stroke="#2d3652" fontSize={10} />
              <Radar name="Accuracy" dataKey="Accuracy" stroke="#818cf8" fill="#818cf8" fillOpacity={0.2} />
              <Radar name="R²" dataKey="R²" stroke="#22c55e" fill="#22c55e" fillOpacity={0.2} />
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Province Predictions */}
      <div className="card">
        <div className="card-title">🔮 ผลทำนาย 77 จังหวัด ({selectedModel})</div>
        <ResponsiveContainer width="100%" height={Math.max(400, predData.length * 22)}>
          <BarChart data={predData} layout="vertical" margin={{left: 10}}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2d3652" />
            <XAxis type="number" domain={[0, 100]} stroke="#64748b" fontSize={10} unit="%" />
            <YAxis dataKey="name" type="category" width={100} stroke="#94a3b8" fontSize={10} interval={0} />
            <Tooltip contentStyle={{background:'#1a2035',border:'1px solid #2d3652',borderRadius:'8px',color:'#f0f4ff'}}
                     formatter={(v) => `${v}%`} />
            <Legend />
            <Bar dataKey="predicted_progressive" name="ก้าวหน้า" fill={COLORS.progressive} stackId="a" />
            <Bar dataKey="predicted_populist" name="ประชานิยม" fill={COLORS.populist} stackId="a" />
            <Bar dataKey="predicted_conservative" name="อนุรักษ์" fill={COLORS.conservative} stackId="a" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
