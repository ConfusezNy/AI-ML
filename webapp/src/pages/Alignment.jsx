import { useState, useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Legend, PieChart, Pie, Cell
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

const PARTIES = {
  progressive: {
    '2562': ['อนาคตใหม่'],
    '2566': ['ก้าวไกล'],
    '2569': ['ประชาชน'],
  },
  populist: {
    '2562': ['เพื่อไทย', 'ไทยรักษาชาติ'],
    '2566': ['เพื่อไทย'],
    '2569': ['เพื่อไทย'],
  },
  conservative: {
    '2562': ['พลังประชารัฐ', 'ประชาธิปัตย์', 'ภูมิใจไทย'],
    '2566': ['ภูมิใจไทย', 'พลังประชารัฐ', 'รวมไทยสร้างชาติ', 'ประชาธิปัตย์'],
    '2569': ['ภูมิใจไทย', 'กล้าธรรม', 'ประชาธิปัตย์', 'พลังประชารัฐ'],
  }
}

export default function Alignment({ features, totalSeats }) {
  const [selectedYear, setSelectedYear] = useState('2569')
  
  const alignData = useMemo(() => {
    if (!features?.province_features?.[selectedYear]) return null
    const pf = features.province_features[selectedYear]
    const provs = Object.values(pf)
    
    // Per-province alignment data
    const provinces = provs.map(p => ({
      ...p,
      name: p.province_name.replace('กรุงเทพมหานคร', 'กทม.'),
      dominant: p.dominant_alignment,
    })).sort((a, b) => b.conservative_share - a.conservative_share)

    // Regional summary
    const regions = {}
    provs.forEach(p => {
      const r = p.region || 'ไม่ระบุ'
      if (!regions[r]) regions[r] = { name: r, prog: 0, pop: 0, cons: 0, total: 0, provCount: 0 }
      regions[r].prog += p.progressive_share * p.total_votes / 100
      regions[r].pop += p.populist_share * p.total_votes / 100
      regions[r].cons += p.conservative_share * p.total_votes / 100
      regions[r].total += p.total_votes
      regions[r].provCount++
    })
    
    const regionArr = Object.values(regions).map(r => ({
      name: r.name.replace('ภาค',''),
      ก้าวหน้า: r.total > 0 ? +(r.prog / r.total * 100).toFixed(1) : 0,
      ประชานิยม: r.total > 0 ? +(r.pop / r.total * 100).toFixed(1) : 0,
      อนุรักษ์: r.total > 0 ? +(r.cons / r.total * 100).toFixed(1) : 0,
      provinces: r.provCount,
    }))

    // National totals (constituency seats from province_features)
    const totals = { prog: 0, pop: 0, cons: 0, other: 0 }
    provs.forEach(p => {
      totals.prog += p.progressive_seats || 0
      totals.pop += p.populist_seats || 0
      totals.cons += p.conservative_seats || 0
    })

    // Party list seats from totalSeats.json
    const ts = totalSeats?.[selectedYear]?.alignment_totals
    const partylist = {
      progressive: ts?.progressive?.partylist || 0,
      populist:    ts?.populist?.partylist    || 0,
      conservative: ts?.conservative?.partylist || 0,
      others:      ts?.others?.partylist      || 0,
    }
    const constSeats = {
      progressive: ts?.progressive?.constituency || totals.prog,
      populist:    ts?.populist?.constituency    || totals.pop,
      conservative: ts?.conservative?.constituency || totals.cons,
      others:      ts?.others?.constituency      || 0,
    }

    return { provinces, regionArr, totals, partylist, constSeats }
  }, [features, selectedYear])

  if (!alignData) return <div className="loading"><div className="loading-spinner"></div>Loading...</div>

  const pieData = [
    { name: 'ก้าวหน้า', value: alignData.totals.prog, fill: COLORS.progressive },
    { name: 'ประชานิยม', value: alignData.totals.pop, fill: COLORS.populist },
    { name: 'อนุรักษ์', value: alignData.totals.cons, fill: COLORS.conservative },
  ]

  return (
    <div>
      <div className="page-header">
        <h2>🏛️ แนวทางการเมือง</h2>
        <p>จำแนกพรรคการเมืองเป็น 3 ฝ่าย — วิเคราะห์เชิงอุดมการณ์</p>
      </div>

      <div className="year-tabs">
        {['2562', '2566', '2569'].map(y => (
          <button key={y} className={`year-tab ${selectedYear === y ? 'active' : ''}`}
                  onClick={() => setSelectedYear(y)}>ปี {y}</button>
        ))}
      </div>

      {/* Party Groups */}
      <div className="stats-grid" style={{marginBottom: 24}}>
        {['progressive', 'populist', 'conservative'].map(align => (
          <div className={`stat-card ${align}`} key={align}>
            <div className="stat-label" style={{marginBottom: 12}}>
              {LABELS[align]}
            </div>
            <div style={{display:'flex', flexWrap:'wrap', gap: 6}}>
              {(PARTIES[align]?.[selectedYear] || []).map(party => (
                <span className={`badge ${align}`} key={party}>{party}</span>
              ))}
            </div>
            <div style={{marginTop: 12}}>
              <div style={{fontSize: 32, fontWeight: 800, color: COLORS[align]}}>
                {alignData.constSeats[align] + alignData.partylist[align]}
              </div>
              <div style={{fontSize: 13, color: 'var(--text-muted)', marginTop: 4}}>
                ที่นั่ง (เขต {alignData.constSeats[align]} + บัญชี {alignData.partylist[align]})
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="charts-grid">
        <div className="card">
          <div className="card-title">สัดส่วนที่นั่งตามแนวทาง</div>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                   outerRadius={110} innerRadius={55} paddingAngle={3}
                   label={({name, value}) => `${name} ${value}`} stroke="none">
                {pieData.map((e, i) => <Cell key={i} fill={e.fill} />)}
              </Pie>
              <Tooltip contentStyle={{background:'#1a2035',border:'1px solid #2d3652',borderRadius:'8px',color:'#f0f4ff'}} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="card-title">% คะแนนตามภาค</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={alignData.regionArr}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2d3652" />
              <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} />
              <YAxis stroke="#64748b" fontSize={11} unit="%" />
              <Tooltip contentStyle={{background:'#1a2035',border:'1px solid #2d3652',borderRadius:'8px',color:'#f0f4ff'}}
                       formatter={(v) => `${v}%`} />
              <Legend />
              <Bar dataKey="ก้าวหน้า" fill={COLORS.progressive} stackId="a" />
              <Bar dataKey="ประชานิยม" fill={COLORS.populist} stackId="a" />
              <Bar dataKey="อนุรักษ์" fill={COLORS.conservative} stackId="a" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Province Table */}
      <div className="card">
        <div className="card-title">77 จังหวัด — สัดส่วนคะแนนตามแนวทาง</div>
        <table className="data-table">
          <thead>
            <tr>
              <th>จังหวัด</th>
              <th>ภาค</th>
              <th>ฝ่ายครอง</th>
              <th>ก้าวหน้า</th>
              <th>ประชานิยม</th>
              <th>อนุรักษ์</th>
              <th>สัดส่วน</th>
            </tr>
          </thead>
          <tbody>
            {alignData.provinces.map(p => (
              <tr key={p.province_id}>
                <td style={{fontWeight:600, color:'var(--text-primary)'}}>{p.name}</td>
                <td>{(p.region || '').replace('ภาค','')}</td>
                <td><span className={`badge ${p.dominant}`}>{LABELS[p.dominant]}</span></td>
                <td style={{color: COLORS.progressive}}>{p.progressive_share.toFixed(1)}%</td>
                <td style={{color: COLORS.populist}}>{p.populist_share.toFixed(1)}%</td>
                <td style={{color: COLORS.conservative}}>{p.conservative_share.toFixed(1)}%</td>
                <td style={{width: 200}}>
                  <div className="alignment-bar">
                    <div style={{width:`${p.progressive_share}%`, background:COLORS.progressive}}></div>
                    <div style={{width:`${p.populist_share}%`, background:COLORS.populist}}></div>
                    <div style={{width:`${p.conservative_share}%`, background:COLORS.conservative}}></div>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
