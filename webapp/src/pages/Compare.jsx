import { useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  LineChart, Line, CartesianGrid, Legend
} from 'recharts'

const COLORS = {
  progressive: '#FF6B00',
  populist: '#E3232C',
  conservative: '#3B82F6',
}

export default function Compare({ features, totalSeats }) {
  const comparisonData = useMemo(() => {
    if (!features?.province_features) return null
    const years = ['2562', '2566', '2569']
    
    // National totals per year — use totalSeats for official seat counts (เขต + บัญชีรายชื่อ)
    const national = years.map(y => {
      const pf = features.province_features[y]
      if (!pf) return null
      const provs = Object.values(pf)
      let progV = 0, popV = 0, consV = 0, totalV = 0
      provs.forEach(p => {
        progV += p.progressive_share * p.total_votes / 100
        popV += p.populist_share * p.total_votes / 100
        consV += p.conservative_share * p.total_votes / 100
        totalV += p.total_votes
      })

      // Official seat counts from totalSeats.json
      const ts = totalSeats?.[y]?.alignment_totals
      const progS  = ts?.progressive?.total  || 0
      const popS   = ts?.populist?.total     || 0
      const consS  = ts?.conservative?.total || 0
      const progC  = ts?.progressive?.constituency  || 0
      const popC   = ts?.populist?.constituency     || 0
      const consC  = ts?.conservative?.constituency || 0
      const progPL = ts?.progressive?.partylist  || 0
      const popPL  = ts?.populist?.partylist     || 0
      const consPL = ts?.conservative?.partylist || 0

      return {
        year: `ปี ${y}`,
        yearNum: y,
        ก้าวหน้า_seats: progS,
        ประชานิยม_seats: popS,
        อนุรักษ์_seats: consS,
        ก้าวหน้า_pct: totalV > 0 ? +(progV/totalV*100).toFixed(1) : 0,
        ประชานิยม_pct: totalV > 0 ? +(popV/totalV*100).toFixed(1) : 0,
        อนุรักษ์_pct: totalV > 0 ? +(consV/totalV*100).toFixed(1) : 0,
        totalVotes: totalV,
        provinces: provs,
        // breakdown for table
        progC, popC, consC, progPL, popPL, consPL,
      }
    }).filter(Boolean)
    
    // Per-region comparison
    const regionCompare = {}
    years.forEach(y => {
      const pf = features.province_features[y]
      if (!pf) return
      Object.values(pf).forEach(p => {
        const rKey = p.region || 'ไม่ระบุ'
        if (!regionCompare[rKey]) regionCompare[rKey] = {}
        if (!regionCompare[rKey][y]) regionCompare[rKey][y] = { prog: 0, pop: 0, cons: 0 }
        regionCompare[rKey][y].prog += p.progressive_seats || 0
        regionCompare[rKey][y].pop += p.populist_seats || 0
        regionCompare[rKey][y].cons += p.conservative_seats || 0
      })
    })

    // Province-level swing (2566 → 2569)
    const swing = []
    const pf66 = features.province_features['2566']
    const pf69 = features.province_features['2569']
    if (pf66 && pf69) {
      Object.keys(pf69).forEach(pid => {
        if (pf66[pid] && pf69[pid]) {
          const diff = pf69[pid].conservative_share - pf66[pid].conservative_share
          swing.push({
            name: pf69[pid].province_name.replace('กรุงเทพมหานคร','กทม.'),
            swing: +diff.toFixed(1),
            fill: diff > 0 ? COLORS.conservative : COLORS.progressive,
          })
        }
      })
      swing.sort((a, b) => b.swing - a.swing)
    }
    
    return { national, regionCompare, swing }
  }, [features])

  if (!comparisonData) return <div className="loading"><div className="loading-spinner"></div>Loading...</div>
  const { national, swing } = comparisonData

  const top10Swing = swing.slice(0, 10)
  const bottom10Swing = swing.slice(-10).reverse()

  return (
    <div>
      <div className="page-header">
        <h2>เปรียบเทียบ 3 การเลือกตั้ง</h2>
        <p>วิเคราะห์แนวโน้มจากปี 2562 → 2566 → 2569</p>
      </div>

      <div className="charts-grid">
        <div className="card">
          <div className="card-title">ที่นั่งรายฝ่าย (3 ปี)</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={national}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2d3652" />
              <XAxis dataKey="year" stroke="#94a3b8" fontSize={13} />
              <YAxis stroke="#64748b" fontSize={11} />
              <Tooltip contentStyle={{background:'#1a2035',border:'1px solid #2d3652',borderRadius:'8px',color:'#f0f4ff'}} />
              <Legend />
              <Bar dataKey="ก้าวหน้า_seats" name="ก้าวหน้า" fill={COLORS.progressive} radius={[4,4,0,0]} />
              <Bar dataKey="ประชานิยม_seats" name="ประชานิยม" fill={COLORS.populist} radius={[4,4,0,0]} />
              <Bar dataKey="อนุรักษ์_seats" name="อนุรักษ์" fill={COLORS.conservative} radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="card-title">แนวโน้ม % คะแนน (3 ปี)</div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={national}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2d3652" />
              <XAxis dataKey="year" stroke="#94a3b8" fontSize={13} />
              <YAxis stroke="#64748b" fontSize={11} unit="%" />
              <Tooltip contentStyle={{background:'#1a2035',border:'1px solid #2d3652',borderRadius:'8px',color:'#f0f4ff'}}
                       formatter={(v) => `${v}%`} />
              <Legend />
              <Line type="monotone" dataKey="ก้าวหน้า_pct" name="ก้าวหน้า" stroke={COLORS.progressive} strokeWidth={3} dot={{r:6}} />
              <Line type="monotone" dataKey="ประชานิยม_pct" name="ประชานิยม" stroke={COLORS.populist} strokeWidth={3} dot={{r:6}} />
              <Line type="monotone" dataKey="อนุรักษ์_pct" name="อนุรักษ์" stroke={COLORS.conservative} strokeWidth={3} dot={{r:6}} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="charts-grid">
        <div className="card">
          <div className="card-title">🔵 Top 10 จังหวัดที่อนุรักษ์เพิ่มมากสุด (2566→2569)</div>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={top10Swing} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#2d3652" />
              <XAxis type="number" stroke="#64748b" fontSize={11} unit="%" />
              <YAxis dataKey="name" type="category" width={90} stroke="#94a3b8" fontSize={12} />
              <Tooltip contentStyle={{background:'#1a2035',border:'1px solid #2d3652',borderRadius:'8px',color:'#f0f4ff'}}
                       formatter={(v) => `${v > 0 ? '+' : ''}${v}%`} />
              <Bar dataKey="swing" name="Swing %" radius={[0,4,4,0]}>
                {top10Swing.map((e, i) => <rect key={i} fill={e.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="card-title">🟠 Top 10 จังหวัดที่ก้าวหน้าเพิ่มมากสุด (2566→2569)</div>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={bottom10Swing} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#2d3652" />
              <XAxis type="number" stroke="#64748b" fontSize={11} unit="%" />
              <YAxis dataKey="name" type="category" width={90} stroke="#94a3b8" fontSize={12} />
              <Tooltip contentStyle={{background:'#1a2035',border:'1px solid #2d3652',borderRadius:'8px',color:'#f0f4ff'}}
                       formatter={(v) => `${v > 0 ? '+' : ''}${v}%`} />
              <Bar dataKey="swing" name="Swing %" fill={COLORS.progressive} radius={[4,0,0,4]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Summary Table */}
      <div className="card">
        <div className="card-title">สรุปตัวเลข 3 ปี (เขต + บัญชีรายชื่อ)</div>
        <table className="data-table">
          <thead>
            <tr>
              <th>ปี</th>
              <th>ก้าวหน้า</th>
              <th>ประชานิยม</th>
              <th>อนุรักษ์นิยม</th>
              <th>คะแนนรวม</th>
            </tr>
          </thead>
          <tbody>
            {national.map(n => (
              <tr key={n.yearNum}>
                <td style={{fontWeight:700}}>{n.year}</td>
                <td>
                  <span className="badge progressive">{n.ก้าวหน้า_seats} ที่นั่ง</span>
                  <div style={{fontSize:11,color:'var(--text-muted)',marginTop:3}}>เขต {n.progC} + บัญชี {n.progPL} ({n.ก้าวหน้า_pct}%)</div>
                </td>
                <td>
                  <span className="badge populist">{n.ประชานิยม_seats} ที่นั่ง</span>
                  <div style={{fontSize:11,color:'var(--text-muted)',marginTop:3}}>เขต {n.popC} + บัญชี {n.popPL} ({n.ประชานิยม_pct}%)</div>
                </td>
                <td>
                  <span className="badge conservative">{n.อนุรักษ์_seats} ที่นั่ง</span>
                  <div style={{fontSize:11,color:'var(--text-muted)',marginTop:3}}>เขต {n.consC} + บัญชี {n.consPL} ({n.อนุรักษ์_pct}%)</div>
                </td>
                <td>{(n.totalVotes/1e6).toFixed(1)}M</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
