import { useMemo, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
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

export default function AIPredict({ data, features, totalSeats }) {
  // Support both old multi-model format and new single-model format
  const metrics = data?.metrics || data?.model_comparison?.Transformer
  const predictions = useMemo(() => {
    if (!data) return []
    // New format: predictions_2573 is a flat array
    if (Array.isArray(data.predictions_2573)) return data.predictions_2573
    // Old format: predictions_2573 is an object keyed by model name
    return data.predictions_2573?.Transformer || []
  }, [data])

  const history = useMemo(() => {
    if (!data?.training_history) return []
    const h = Array.isArray(data.training_history?.train_loss)
      ? data.training_history
      : data.training_history?.Transformer
    if (!h) return []
    return h.train_loss.map((tl, i) => ({
      epoch: i + 1,
      train: +tl.toFixed(3),
      val: +h.val_loss[i].toFixed(3),
    })).filter((_, i) => i % 2 === 0)
  }, [data])

  const summaryData = useMemo(() => {
    if (!predictions.length || !features?.province_features?.['2569']) return {}
    const pf2569 = features.province_features['2569']
    const counts = { progressive: 0, populist: 0, conservative: 0, others: 0 }
    const seats  = { progressive: 0, populist: 0, conservative: 0, others: 0 }

    // Weighted vote shares for partylist estimation (weighted by num_zones ≈ population)
    const wShares = { progressive: 0, populist: 0, conservative: 0, others: 0 }
    let totalZones = 0

    predictions.forEach(p => {
      const align = p.predicted_winner || 'others'
      counts[align] = (counts[align] || 0) + 1
      const zones = pf2569[p.province_id]?.num_zones || 0
      seats[align] = (seats[align] || 0) + zones

      // accumulate weighted shares
      const prog = p.predicted_progressive || 0
      const pop  = p.predicted_populist    || 0
      const cons = p.predicted_conservative || 0
      const oth  = Math.max(0, 100 - prog - pop - cons)
      wShares.progressive  += prog * zones
      wShares.populist     += pop  * zones
      wShares.conservative += cons * zones
      wShares.others       += oth  * zones
      totalZones += zones
    })

    // Distribute 100 partylist seats by national weighted vote share
    const PARTY_LIST_SEATS = 100
    const pctProg = totalZones > 0 ? wShares.progressive  / totalZones : 0
    const pctPop  = totalZones > 0 ? wShares.populist     / totalZones : 0
    const pctCons = totalZones > 0 ? wShares.conservative / totalZones : 0
    const pctOth  = totalZones > 0 ? wShares.others       / totalZones : 0
    const totalPct = pctProg + pctPop + pctCons + pctOth || 1
    const partylist = {
      progressive:  Math.round(pctProg / totalPct * PARTY_LIST_SEATS),
      populist:     Math.round(pctPop  / totalPct * PARTY_LIST_SEATS),
      conservative: Math.round(pctCons / totalPct * PARTY_LIST_SEATS),
      others:       Math.round(pctOth  / totalPct * PARTY_LIST_SEATS),
    }

    return { counts, seats, partylist, totalSeats: Object.values(seats).reduce((a, b) => a + b, 0) }
  }, [predictions, features])

  // Unique regions from predictions
  const regions = useMemo(() => {
    const set = new Set(predictions.map(p => p.region || 'ไม่ระบุ'))
    return ['ทั้งหมด', ...Array.from(set).sort()]
  }, [predictions])

  const [selectedRegion, setSelectedRegion] = useState('ทั้งหมด')

  const predData = useMemo(() => {
    const filtered = selectedRegion === 'ทั้งหมด'
      ? predictions
      : predictions.filter(p => (p.region || 'ไม่ระบุ') === selectedRegion)
    return filtered.map(p => ({
      ...p,
      name: p.province_name.replace('กรุงเทพมหานคร', 'กทม.'),
    }))
  }, [predictions, selectedRegion])

  // Filtered province counts (for stat cards when region is filtered)
  const filteredCounts = useMemo(() => {
    const counts = { progressive: 0, populist: 0, conservative: 0, others: 0 }
    predData.forEach(p => {
      const align = p.predicted_winner || 'others'
      counts[align] = (counts[align] || 0) + 1
    })
    return counts
  }, [predData])

  if (!metrics) return <div className="loading"><div className="loading-spinner"></div>Loading...</div>

  return (
    <div>
      <div className="page-header">
        <h2>🤖 AI Prediction — ทำนายผลเลือกตั้ง 2573</h2>
        <p>โมเดล: <strong>Transformer</strong> (Self-Attention · d_model=64 · 4 heads · 2 layers)</p>
      </div>

      {/* Metrics Card */}
      <div className="stats-grid" style={{ marginBottom: 24 }}>
        <div className="stat-card accent">
          <div className="stat-label">Accuracy</div>
          <div className="stat-value" style={{ color: '#22c55e' }}>{metrics.accuracy?.toFixed(1)}%</div>
          <div className="stat-change">Classification</div>
        </div>
        <div className="stat-card accent">
          <div className="stat-label">MAE</div>
          <div className="stat-value">{metrics.mae?.toFixed(2)}</div>
          <div className="stat-change">Regression error</div>
        </div>
        <div className="stat-card accent">
          <div className="stat-label">RMSE</div>
          <div className="stat-value">{metrics.rmse?.toFixed(2)}</div>
          <div className="stat-change">Regression error</div>
        </div>
        <div className="stat-card accent">
          <div className="stat-label">R²</div>
          <div className="stat-value">{metrics.r2?.toFixed(3)}</div>
          <div className="stat-change">Fit quality</div>
        </div>
      </div>

      {/* Per-class accuracy */}
      {metrics.per_class_accuracy && (
        <div className="card" style={{ marginBottom: 24 }}>
          <div className="card-title">📊 Per-Class Accuracy</div>
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginTop: 12 }}>
            {Object.entries(metrics.per_class_accuracy).map(([cls, acc]) => (
              <div key={cls} style={{
                flex: 1, minWidth: 140, background: 'rgba(255,255,255,0.03)',
                borderRadius: 12, padding: '14px 18px',
                borderLeft: `4px solid ${COLORS[cls] || '#888'}`
              }}>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>{LABELS[cls]}</div>
                <div style={{ fontSize: 28, fontWeight: 800, color: COLORS[cls] || '#888' }}>{acc.toFixed(1)}%</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Training Loss */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-title">📉 Training Loss (Transformer)</div>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={history}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2d3652" />
            <XAxis dataKey="epoch" stroke="#64748b" fontSize={11} />
            <YAxis stroke="#64748b" fontSize={11} />
            <Tooltip contentStyle={{ background: '#1a2035', border: '1px solid #2d3652', borderRadius: '8px', color: '#f0f4ff' }} />
            <Legend />
            <Line type="monotone" dataKey="train" name="Train Loss" stroke="#818cf8" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="val" name="Val Loss" stroke="#f472b6" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Predicted Seats */}
      {summaryData.seats && (
        <>
          <div className="card" style={{ marginBottom: 24, padding: '20px 24px' }}>
            <div className="card-title">🔮 ทำนายที่นั่ง 2573</div>
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginTop: 16 }}>
              {Object.entries(summaryData.seats).filter(([, v]) => v > 0).map(([align, seatCount]) => {
                const pl = summaryData.partylist[align] || 0
                return (
                  <div key={align} style={{
                    flex: 1, minWidth: 160, background: 'rgba(255,255,255,0.03)',
                    borderRadius: 12, padding: '16px 20px',
                    borderLeft: `4px solid ${COLORS[align]}`
                  }}>
                    <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 6 }}>{LABELS[align]}</div>
                    <div style={{ fontSize: 32, fontWeight: 800, color: COLORS[align] }}>
                      {seatCount + pl}
                    </div>
                    <div style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 4 }}>
                      ที่นั่ง (เขต {seatCount} + บัญชี {pl}*)
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>{summaryData.counts[align]} จังหวัด</div>
                  </div>
                )
              })}
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 12 }}>* บัญชีรายชื่อประมาณจากสัดส่วนคะแนนเฉลี่ยระดับชาติ (100 ที่นั่ง)</div>
          </div>

          <div className="stats-grid">
            {Object.entries(filteredCounts).filter(([, v]) => v > 0).map(([align, count]) => (
              <div className={`stat-card ${align}`} key={align}>
                <div className="stat-label">{LABELS[align]}</div>
                <div className="stat-value" style={{ color: COLORS[align] }}>{count}</div>
                <div className="stat-change">
                  จังหวัดที่ชนะ{selectedRegion !== 'ทั้งหมด' ? ` (${selectedRegion})` : ' (ทั้งประเทศ)'}
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {/* Province Predictions Bar Chart */}
      <div className="card" style={{ marginTop: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16, flexWrap: 'wrap', gap: 8 }}>
          <div className="card-title" style={{ marginBottom: 0 }}>🔮 ผลทำนายรายจังหวัด ({predData.length} จังหวัด)</div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {regions.map(r => (
              <button
                key={r}
                onClick={() => setSelectedRegion(r)}
                style={{
                  padding: '5px 12px', borderRadius: 20, fontSize: 12, fontWeight: 600,
                  cursor: 'pointer', fontFamily: 'inherit', transition: 'all 0.15s',
                  border: selectedRegion === r ? 'none' : '1px solid var(--border)',
                  background: selectedRegion === r ? 'var(--accent-gradient)' : 'transparent',
                  color: selectedRegion === r ? 'white' : 'var(--text-secondary)',
                }}
              >{r.replace('ภาค', '')}
              </button>
            ))}
          </div>
        </div>
        <ResponsiveContainer width="100%" height={Math.max(400, predData.length * 22)}>
          <BarChart data={predData} layout="vertical" margin={{ left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2d3652" />
            <XAxis type="number" domain={[0, 100]} stroke="#64748b" fontSize={10} unit="%" />
            <YAxis dataKey="name" type="category" width={100} stroke="#94a3b8" fontSize={10} interval={0} />
            <Tooltip
              contentStyle={{ background: '#1a2035', border: '1px solid #2d3652', borderRadius: '8px', color: '#f0f4ff' }}
              formatter={(v) => `${v}%`}
            />
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
