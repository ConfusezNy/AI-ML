# Election Prediction Web App — Implementation Plan

## สถานะปัจจุบัน

| Phase | สถานะ | รายละเอียด |
|---|---|---|
| Phase 1: Data Collection | ✅ เสร็จ | ดึงข้อมูล 2566+2569 จาก The Standard (Protobuf decode) |
| Phase 2: AI Models | 🔜 ถัดไป | LSTM / BiLSTM / Transformer |
| Phase 3: Web App | ⏳ รอ | Dashboard + Compare + AI Prediction |
| Phase 4: Deploy | ⏳ รอ | Vercel/Netlify |

---

## ข้อมูลที่มี vs ข้อมูลที่ต้องการเพิ่ม

### ✅ ข้อมูลที่มีแล้ว (จาก Phase 1)
- คะแนนรายเขต 400 เขต × 2 ปี (2566, 2569)
- คะแนนบัญชีรายชื่อ 400 เขต × 2 ปี
- ชื่อพรรค, จังหวัด, ภาค, ผู้มีสิทธิ์, บัตรเสีย

### 📊 ข้อมูลที่ควรเพิ่ม

#### 1. ข้อมูลประชากร & ผู้มีสิทธิ์เลือกตั้ง

| ปี (พ.ศ.) | ผู้มีสิทธิ์เลือกตั้ง | แหล่งข้อมูล |
|---|---|---|
| 2566 | 52,322,824 คน | กกต. (ข้อมูลจริง) |
| 2569 | 52,922,923 คน | กกต. (ข้อมูลจริง) |
| 2573 (คาดการณ์) | ~53.5 ล้านคน | สภาพัฒน์ฯ → ประมาณจาก trend |

> **หมายเหตุ**: ประชากรไทยจะถึงจุดสูงสุด 67.19 ล้านคนในปี 2571
> หลังจากนั้นจะลดลง เนื่องจากเข้าสู่สังคมสูงวัย

#### 2. จำแนกพรรคตามแนวคิดทางการเมือง (Political Alignment)

| กลุ่ม | พรรค | แนวคิด |
|---|---|---|
| 🟠 **ก้าวหน้า (Progressive)** | ประชาชน (ก้าวไกลเดิม) | ปฏิรูปสถาบัน, ลดอำนาจทหาร, สิทธิเท่าเทียม |
| 🔴 **ประชานิยม (Populist)** | เพื่อไทย | นโยบายประชานิยม, digital wallet, ฐานเสียง อีสาน-เหนือ |
| 🔵 **อนุรักษ์นิยม (Conservative)** | ภูมิใจไทย | ชาตินิยมเชิงปฏิบัติ, ถิ่นกัญชา, kingmaker |
| | ประชาธิปัตย์ | อนุรักษ์นิยมเสรี, ฐานเสียงภาคใต้ |
| | พลังประชารัฐ | ชาตินิยม, สายทหาร |
| | รวมไทยสร้างชาติ | สนับสนุน พล.อ.ประยุทธ์ |
| 🟡 **อื่นๆ (Others)** | กล้าธรรม, ไทยสร้างไทย, เศรษฐกิจ | พรรคขนาดเล็ก-กลาง |

#### 3. ข้อมูลเพิ่มเติมที่ช่วย AI ทำนายดีขึ้น (จะ hardcode ลงไป)

| ข้อมูล | วิธีได้มา | ใช้ทำอะไร |
|---|---|---|
| ฐานเสียงประจำภาค | จากผลเลือกตั้ง 2 ปี | Feature สำหรับ AI |
| อัตราผู้มาใช้สิทธิ์ (Turnout) | คำนวณจาก CSV ที่มี | ดูแนวโน้ม |
| สัดส่วนบัตรเสีย | คำนวณจาก CSV ที่มี | Feature สำหรับ AI |
| Swing % (การเปลี่ยนแปลงคะแนน) | เทียบ 2566 vs 2569 | Key feature |

---

## Phase 2: AI Model Design

### Input Features (ต่อเขต)

```
features = [
    region_encoded,           # ภาค (one-hot: 7 ภาค)
    province_encoded,         # จังหวัด (label encoded)
    eligible_voters,          # จำนวนผู้มีสิทธิ์
    turnout_rate_2566,        # อัตราผู้มาใช้สิทธิ์ ปี 2566
    invalid_rate_2566,        # อัตราบัตรเสีย ปี 2566
    vote_share_party_1_2566,  # % คะแนนพรรค 1 ปี 2566
    vote_share_party_2_2566,  # % คะแนนพรรค 2 ปี 2566
    ...                       # % คะแนนพรรค top-10
    alignment_progressive,    # สัดส่วน ฝ่ายก้าวหน้า 2566
    alignment_populist,       # สัดส่วน ฝ่ายประชานิยม 2566
    alignment_conservative,   # สัดส่วน ฝ่ายอนุรักษ์นิยม 2566
]
```

### Target (สิ่งที่ทำนาย)
```
target = [
    vote_share_party_1_2569,  # % คะแนนแต่ละพรรค ปี 2569
    vote_share_party_2_2569,
    ...
    winner_party_2569,        # พรรคที่ชนะ
]
```

### Sequence Design
```
จังหวัด เชียงใหม่ (10 เขต):
  [เขต1_features] → [เขต2_features] → ... → [เขต10_features]
  
LSTM/BiLSTM: อ่าน sequence นี้ เรียนรู้ pattern ภายในจังหวัด
Transformer: ดูความสัมพันธ์ระหว่างเขตทั้งหมด (self-attention)
```

### Evaluation Metrics
- **MAE** = Mean Absolute Error (ค่าความผิดพลาดเฉลี่ย %)
- **RMSE** = Root Mean Squared Error
- **Accuracy** = ทำนายพรรคผู้ชนะถูกกี่ %
- **F1 Score** = per-party prediction quality

---

## Phase 3: Web App Design

### Tech Stack
- **Framework**: Vite + React (เร็ว, lightweight)
- **Charts**: Chart.js หรือ Recharts
- **Map**: SVG Thailand map (interactive)
- **Styling**: CSS Dark Theme + Glassmorphism
- **Data**: Static JSON (ไม่ต้อง backend)

### หน้าเว็บ (6 หน้า)

#### 1. 🏠 Dashboard (หน้าแรก)
- **Hero**: แผนที่ประเทศไทย แยกสีตามพรรคที่ชนะแต่ละเขต
- **Summary Cards**: จำนวนที่นั่ง แต่ละพรรค (ปี 2569)
- **Pie Chart**: สัดส่วนที่นั่ง
- **Bar Chart**: Top 10 พรรค
- **Toggle**: สลับดู 2566 / 2569

#### 2. 📊 Compare (เปรียบเทียบ 2566 vs 2569)
- **Side-by-side**: ที่นั่ง + คะแนนรวม
- **Swing Chart**: พรรคไหนได้/เสียที่นั่ง
- **Filter**: เลือกดูตามภาค/จังหวัด
- **Table**: ข้อมูลรายเขต sortable

#### 3. 🤖 AI Prediction (หน้า AI — หัวใจของงาน)
- **Model Selector**: เลือกดูผล LSTM / BiLSTM / Transformer
- **Prediction Map**: แผนที่แสดงผลทำนาย ปี 2573
- **Accuracy Dashboard**: เปรียบเทียบ 3 models (MAE, RMSE, Accuracy)
- **Confusion Matrix**: แสดง per-party prediction quality
- **Insights Panel**: สรุปว่า model ไหนดีที่สุด และทำไม

#### 4. 🗳️ Political Alignment (แนวทางการเมือง)
- **Alignment Filter**: ก้าวหน้า / ประชานิยม / อนุรักษ์นิยม
- **Stacked Bar**: สัดส่วนแนวทาง แยกตามภาค
- **Treemap**: พรรคกลุ่มไหนมีฐานเสียงที่ไหน
- **Trend**: แนวโน้ม 2566→2569 ฝ่ายไหนขึ้น/ลง

#### 5. 🗺️ Province Detail (รายจังหวัด)
- กดจังหวัดจาก map → ดูข้อมูลละเอียด
- รายเขต: ผู้ชนะ, คะแนน, % มาใช้สิทธิ์
- กราฟเปรียบเทียบ 2566 vs 2569

#### 6. 📈 Demographics & Insights
- **ประชากร**: แนวโน้ม 2566→2573 (สังคมสูงวัย)
- **Turnout**: อัตราผู้มาใช้สิทธิ์ แยกตามภาค
- **Invalid Votes**: บัตรเสีย trend
- **Gen Analysis**: New voters vs aging voters impact

---

## Phase 4: Deploy
- Deploy บน **Vercel** (ฟรี, เร็ว)
- ข้อมูลเป็น static JSON (ไม่ต้อง server)
- เตรียม presentation slides

---

## Timeline (ประมาณการ)

| Phase | งาน | เวลา |
|---|---|---|
| Phase 2 | Feature Engineering + 3 AI Models | 2-3 ชั่วโมง |
| Phase 3 | Web App (6 หน้า) | 3-4 ชั่วโมง |
| Phase 4 | Deploy + Polish | 1 ชั่วโมง |
