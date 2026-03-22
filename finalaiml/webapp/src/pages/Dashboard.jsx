import { useState, useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  /* eslint-disable no-unused-vars */
  PieChart, Pie, Cell, Legend, CartesianGrid
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

export default function Dashboard({ data, features, totalSeats }) {
  const [selectedYear, setSelectedYear] = useState('2569')
  
  const yearData = useMemo(() => {
    if (!features?.province_features?.[selectedYear]) return null
    const pf = features.province_features[selectedYear]
    
    let totalProg = 0, totalPop = 0, totalCons = 0
    let totalVotesSum = 0, totalEligible = 0
    
    const provinces = Object.values(pf).map(p => {
      totalProg += p.progressive_share * p.total_votes / 100
      totalPop += p.populist_share * p.total_votes / 100
      totalCons += p.conservative_share * p.total_votes / 100
      totalVotesSum += p.total_votes
      totalEligible += p.total_eligible
      return p
    })

    // Use totalSeats for official seat counts (constituency + partylist)
    const ts = totalSeats?.[selectedYear]?.alignment_totals
    const progTotal = ts?.progressive?.total || 0
    const popTotal = ts?.populist?.total || 0
    const consTotal = ts?.conservative?.total || 0
    const othTotal = ts?.others?.total || 0
    const progConst = ts?.progressive?.constituency || 0
    const popConst = ts?.populist?.constituency || 0
    const consConst = ts?.conservative?.constituency || 0
    const progPL = ts?.progressive?.partylist || 0
    const popPL = ts?.populist?.partylist || 0
    const consPL = ts?.conservative?.partylist || 0
    
    return {
      provinces,
      totalVotes: totalVotesSum,
      totalEligible,
      totalAllSeats: totalSeats?.[selectedYear]?.total_seats || 500,
      turnout: totalEligible > 0 ? (totalVotesSum / totalEligible * 100).toFixed(1) : 0,
      alignment: {
        progressive: { total: progTotal, constituency: progConst, partylist: progPL, pct: (totalProg/totalVotesSum*100).toFixed(1) },
        populist: { total: popTotal, constituency: popConst, partylist: popPL, pct: (totalPop/totalVotesSum*100).toFixed(1) },
        conservative: { total: consTotal, constituency: consConst, partylist: consPL, pct: (totalCons/totalVotesSum*100).toFixed(1) },
        others: { total: othTotal }
      }
    }
  }, [features, selectedYear, totalSeats])
  
  if (!yearData) return <div className="loading"><div className="loading-spinner"></div>Loading...</div>

  const pieData = [
    { name: 'ก้าวหน้า', value: yearData.alignment.progressive.total, fill: COLORS.progressive },
    { name: 'ประชานิยม', value: yearData.alignment.populist.total, fill: COLORS.populist },
    { name: 'อนุรักษ์นิยม', value: yearData.alignment.conservative.total, fill: COLORS.conservative },
    { name: 'อื่นๆ', value: yearData.alignment.others.total, fill: COLORS.others },
  ]

  const regionData = []
  const regionMap = {}
  yearData.provinces.forEach(p => {
    if (!regionMap[p.region]) {
      regionMap[p.region] = { name: p.region, progressive: 0, populist: 0, conservative: 0 }
    }
    regionMap[p.region].progressive += p.progressive_seats || 0
    regionMap[p.region].populist += p.populist_seats || 0
    regionMap[p.region].conservative += p.conservative_seats || 0
  })
  Object.values(regionMap).forEach(r => regionData.push(r))

  const topProvinces = [...yearData.provinces]
    .sort((a, b) => b.total_votes - a.total_votes)
    .slice(0, 15)
    .map(p => ({
      name: p.province_name.replace('กรุงเทพมหานคร', 'กทม.'),
      ก้าวหน้า: p.progressive_share,
      ประชานิยม: p.populist_share,
      อนุรักษ์: p.conservative_share,
    }))

  return (
    <div>
      <div className="page-header">
        <h2>Dashboard ผลเลือกตั้งไทย</h2>
        <p>ภาพรวมผลเลือกตั้ง {selectedYear === '2569' ? 'ล่าสุด' : `ปี ${selectedYear}`} — 77 จังหวัด</p>
      </div>

      <div className="year-tabs">
        {['2562', '2566', '2569'].map(y => (
          <button key={y} className={`year-tab ${selectedYear === y ? 'active' : ''}`}
                  onClick={() => setSelectedYear(y)}>
            ปี {y}
          </button>
        ))}
      </div>

      <div className="stats-grid">
        <div className="stat-card accent">
          <div className="stat-label">ผู้มาใช้สิทธิ์</div>
          <div className="stat-value">{yearData.turnout}%</div>
          <div className="stat-change">{(yearData.totalVotes/1e6).toFixed(1)}M / {(yearData.totalEligible/1e6).toFixed(1)}M คน</div>
        </div>
        <div className="stat-card progressive">
          <div className="stat-label">🟠 ก้าวหน้า</div>
          <div className="stat-value" style={{color: COLORS.progressive}}>{yearData.alignment.progressive.total}</div>
          <div className="stat-change">ที่นั่ง (เขต {yearData.alignment.progressive.constituency} + บัญชี {yearData.alignment.progressive.partylist})</div>
        </div>
        <div className="stat-card populist">
          <div className="stat-label">🔴 ประชานิยม</div>
          <div className="stat-value" style={{color: COLORS.populist}}>{yearData.alignment.populist.total}</div>
          <div className="stat-change">ที่นั่ง (เขต {yearData.alignment.populist.constituency} + บัญชี {yearData.alignment.populist.partylist})</div>
        </div>
        <div className="stat-card conservative">
          <div className="stat-label">🔵 อนุรักษ์นิยม</div>
          <div className="stat-value" style={{color: COLORS.conservative}}>{yearData.alignment.conservative.total}</div>
          <div className="stat-change">ที่นั่ง (เขต {yearData.alignment.conservative.constituency} + บัญชี {yearData.alignment.conservative.partylist})</div>
        </div>
      </div>

      <div className="charts-grid">
        <div className="card">
          <div className="card-title">สัดส่วนที่นั่ง {yearData.totalAllSeats} ที่นั่ง — ปี {selectedYear}</div>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                   outerRadius={110} innerRadius={60} paddingAngle={3}
                   label={({name, value}) => `${name} ${value}`}
                   stroke="none">
                {pieData.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
              </Pie>
              <Tooltip contentStyle={{background:'#1a2035',border:'1px solid #2d3652',borderRadius:'8px',color:'#f0f4ff'}} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="card-title">ที่นั่งรายภาค ปี {selectedYear}</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={regionData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#2d3652" />
              <XAxis type="number" stroke="#64748b" fontSize={11} />
              <YAxis dataKey="name" type="category" width={120} stroke="#64748b" fontSize={11} />
              <Tooltip contentStyle={{background:'#1a2035',border:'1px solid #2d3652',borderRadius:'8px',color:'#f0f4ff'}} />
              <Bar dataKey="progressive" name="ก้าวหน้า" fill={COLORS.progressive} stackId="a" radius={[0,0,0,0]} />
              <Bar dataKey="populist" name="ประชานิยม" fill={COLORS.populist} stackId="a" />
              <Bar dataKey="conservative" name="อนุรักษ์" fill={COLORS.conservative} stackId="a" radius={[0,4,4,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card" style={{marginBottom: 24}}>
        <div className="card-title">สัดส่วนคะแนน Top 15 จังหวัด</div>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={topProvinces} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#2d3652" />
            <XAxis type="number" domain={[0, 100]} stroke="#64748b" fontSize={11} unit="%" />
            <YAxis dataKey="name" type="category" width={100} stroke="#94a3b8" fontSize={12} />
            <Tooltip contentStyle={{background:'#1a2035',border:'1px solid #2d3652',borderRadius:'8px',color:'#f0f4ff'}}
                     formatter={(v) => `${v.toFixed(1)}%`} />
            <Bar dataKey="ก้าวหน้า" fill={COLORS.progressive} stackId="a" />
            <Bar dataKey="ประชานิยม" fill={COLORS.populist} stackId="a" />
            <Bar dataKey="อนุรักษ์" fill={COLORS.conservative} stackId="a" />
            <Legend />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
