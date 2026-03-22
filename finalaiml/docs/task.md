# Election Prediction Web App - Task Checklist

## Phase 1: Data Collection ✅ เสร็จแล้ว
- [x] Research data source (election2569.thestandard.co)
- [x] Download Protobuf .bin files (master, score-66, score-69)
- [x] Decode Protobuf → CSV/JSON
- [x] Fix Thai encoding (party names, provinces, regions)
- [x] Output files:
  - `constituency_2566_fixed.csv` (4,671 rows)
  - `constituency_2569_fixed.csv` (3,486 rows)
  - `partylist_2566_fixed.csv` (24,578 rows)
  - `partylist_2569_fixed.csv` (22,715 rows)

## Phase 2: AI Model Development 🔜 ถัดไป
- [ ] Feature Engineering (สร้าง features จากข้อมูลรายเขต)
- [ ] Build LSTM model
- [ ] Build BiLSTM model
- [ ] Build Transformer model
- [ ] Train & Evaluate (compare MAE, RMSE, Accuracy)
- [ ] Export ผลทำนาย → predictions.json

## Phase 3: Web Application
- [ ] Set up project (Next.js / Vite)
- [ ] Dashboard page (กราฟรวม, จำนวนที่นั่ง)
- [ ] Compare page (2566 vs 2569)
- [ ] AI Prediction page (ผลทำนาย + เปรียบเทียบ 3 models)
- [ ] Region/Province drill-down
- [ ] Responsive + dark theme design

## Phase 4: Deploy & Present
- [ ] Deploy web app (Vercel / Netlify)
- [ ] Prepare presentation slides
- [ ] Document AI methodology
