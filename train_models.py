"""
เทรนโมเดล Transformer สำหรับทำนายผลเลือกตั้ง
แล้วนำโมเดลที่เทรนแล้วมาทำนายการเลือกตั้งปี 2573
"""
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# กำหนด seed เพื่อให้ผลลัพธ์เหมือนกันทุกครั้งที่รัน
torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Dataset — ห่อข้อมูลให้ PyTorch ใช้งานได้
# ============================================================
class ElectionDataset(Dataset):
    def __init__(self, X, y_reg, y_cls):
        # แปลงข้อมูล numpy ให้เป็น Tensor ที่ PyTorch อ่านได้
        self.X = torch.FloatTensor(X)           # features ของแต่ละจังหวัด
        self.y_reg = torch.FloatTensor(y_reg)   # เป้าหมาย: % คะแนน (Regression)
        self.y_cls = torch.LongTensor(y_cls)    # เป้าหมาย: แนวทางที่ชนะ (Classification)
    
    def __len__(self):
        return len(self.X)  # จำนวน sample ทั้งหมด
    
    def __getitem__(self, idx):
        # คืนค่า 1 sample เมื่อ DataLoader เรียก
        return self.X[idx], self.y_reg[idx], self.y_cls[idx]


# ============================================================
# โมเดล Transformer
# ============================================================
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, num_classes=4, dropout=0.3):
        super().__init__()

        # [1] แปลง 29 features → vector ขนาด 64
        #     เหมือนแปลภาษาข้อมูลให้โมเดลเข้าใจ
        self.input_proj = nn.Linear(input_size, d_model)

        # [2] Positional Encoding
        #     ปกติ Transformer ใช้กับ sequence (หลาย token)
        #     แต่เราใส่แค่ 1 token (1 จังหวัด) จึงใช้ learnable parameter แทน
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, d_model))

        # [3] Transformer Encoder — หัวใจหลักของโมเดล
        #     Self-Attention วิเคราะห์ว่า feature ไหนสำคัญ
        #     nhead=4 → ดูข้อมูลแบบขนาน 4 มุมมองพร้อมกัน
        #     dim_feedforward=128 → เครือข่าย FFN ภายใน
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # [4] Regression Head — ทำนาย % คะแนน 3 แนวทาง
        #     64 → 32 → 3 (ก้าวหน้า%, ประชานิยม%, อนุรักษ์%)
        self.fc_reg = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 3)
        )

        # [5] Classification Head — ทำนายแนวทางที่ชนะ (4 class)
        #     64 → 32 → 4 (progressive / populist / conservative / others)
        self.fc_cls = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        # x มีขนาด (batch_size, 29)
        x = x.unsqueeze(1)        # เพิ่ม dimension → (batch, 1, 29) [seq_len=1]
        x = self.input_proj(x)    # แปลง → (batch, 1, 64)
        x = x + self.pos_encoding # รวม positional encoding
        out = self.transformer(x) # ผ่าน Self-Attention → (batch, 1, 64)
        out = out.mean(dim=1)     # รวมผลด้วย Global Average Pooling → (batch, 64)
        reg = self.fc_reg(out)    # ทำนาย % คะแนน → (batch, 3)
        cls = self.fc_cls(out)    # ทำนายแนวทางชนะ → (batch, 4)
        return reg, cls


# ============================================================
# ฟังก์ชันเทรนโมเดล
# ============================================================
def train_model(model, train_loader, val_loader, model_name, epochs=200, lr=0.001):
    """เทรนโมเดลพร้อม Early Stopping (หยุดเมื่อไม่ดีขึ้นแล้ว)"""
    # ใช้ GPU ถ้ามี, ไม่งั้นใช้ CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Adam Optimizer — ปรับ weight อัตโนมัติ, weight_decay ป้องกัน overfit
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # ลด learning rate อัตโนมัติเมื่อ val_loss หยุดลด
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    
    # Loss functions สำหรับแต่ละ task
    criterion_reg = nn.MSELoss()         # Regression: ผิดเยอะ → โดนลงโทษมาก
    criterion_cls = nn.CrossEntropyLoss()# Classification: ทำนายผิด class → โดนลงโทษ
    
    best_val_loss = float("inf")
    patience = 30         # หยุดถ้า val_loss ไม่ดีขึ้นภายใน 30 epoch
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(epochs):
        # ===== ช่วง Train =====
        model.train()  # เปิด mode train (dropout ทำงาน)
        train_loss = 0
        for X, y_reg, y_cls in train_loader:
            X, y_reg, y_cls = X.to(device), y_reg.to(device), y_cls.to(device)
            
            # Forward pass: ส่งข้อมูลผ่านโมเดล
            pred_reg, pred_cls = model(X)

            # คำนวณ Loss รวมทั้ง 2 task
            loss = criterion_reg(pred_reg, y_reg) + criterion_cls(pred_cls, y_cls)
            
            # Backward pass: คำนวณ gradient แล้วปรับ weight
            optimizer.zero_grad()  # ล้าง gradient เก่า
            loss.backward()        # คำนวณ gradient (เรียนรู้จากความผิดพลาด)
            optimizer.step()       # อัพเดท weight
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # ===== ช่วง Validate =====
        model.eval()  # ปิด mode train (dropout ไม่ทำงาน)
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():  # ไม่ต้องคำนวณ gradient ช่วง validate
            for X, y_reg, y_cls in val_loader:
                X, y_reg, y_cls = X.to(device), y_reg.to(device), y_cls.to(device)
                pred_reg, pred_cls = model(X)
                loss = criterion_reg(pred_reg, y_reg) + criterion_cls(pred_cls, y_cls)
                val_loss += loss.item()
                
                # นับว่าทำนาย class ถูกกี่ตัว
                _, predicted = pred_cls.max(1)
                total += y_cls.size(0)
                correct += predicted.eq(y_cls).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = correct / total * 100  # % ความแม่นยำ
        
        # ปรับ learning rate ตาม val_loss
        scheduler.step(val_loss)
        
        # บันทึกประวัติ (สำหรับ plot กราฟ loss ใน dashboard)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        # Early Stopping — บันทึก model ที่ดีที่สุด
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # save weight ของ model ที่ดีที่สุด
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{model_name}.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break  # หยุดก่อนถึง 200 epoch
        
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1:3d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.1f}%")
    
    # โหลด weight ที่ดีที่สุดกลับมา
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{model_name}.pth"), weights_only=True))
    
    return model, history


# ============================================================
# ฟังก์ชันประเมินผล
# ============================================================
def evaluate_model(model, X_test, y_reg_test, y_cls_test, device="cpu"):
    """วัดประสิทธิภาพโมเดลบน Test Set"""
    model.eval()
    model = model.to(device)
    
    X_t = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        pred_reg, pred_cls = model(X_t)
    
    pred_reg = pred_reg.cpu().numpy()
    pred_cls_labels = pred_cls.cpu().argmax(dim=1).numpy()
    
    # Regression metrics
    mae = np.mean(np.abs(pred_reg - y_reg_test))                         # ค่าเฉลี่ยความผิดพลาด
    rmse = np.sqrt(np.mean((pred_reg - y_reg_test) ** 2))               # ลงโทษ error ใหญ่มากกว่า
    
    # R² — โมเดลอธิบายได้กี่ % (1.0 = perfect)
    ss_res = np.sum((y_reg_test - pred_reg) ** 2)
    ss_tot = np.sum((y_reg_test - np.mean(y_reg_test, axis=0)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # Classification accuracy
    accuracy = np.mean(pred_cls_labels == y_cls_test) * 100
    
    # แยก accuracy รายแนวทาง
    cls_names = ["progressive", "populist", "conservative", "others"]
    per_class = {}
    for i, name in enumerate(cls_names):
        mask = y_cls_test == i
        if mask.sum() > 0:
            per_class[name] = float(np.mean(pred_cls_labels[mask] == i) * 100)
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "accuracy": float(accuracy),
        "per_class_accuracy": per_class,
        "predictions_reg": pred_reg.tolist(),
        "predictions_cls": pred_cls_labels.tolist(),
    }


# ============================================================
# Main — รันเมื่อเรียกไฟล์นี้โดยตรง
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Training Transformer Model for Election Prediction")
    print("=" * 60)
    
    # ----- [1] โหลดข้อมูล -----
    print("\n[1] Loading features...")
    with open(os.path.join(DATA_DIR, "ml_features.json"), encoding="utf-8") as f:
        data = json.load(f)
    
    pairs = data["training_pairs"]          # ชุดข้อมูล: (ข้อมูลปีก่อน → ผลปีถัดไป)
    pred_input = data["prediction_input_2573"]  # ข้อมูลปี 2569 สำหรับทำนาย 2573
    
    # ----- [2] เตรียม Array -----
    X = np.array([p["input"] for p in pairs])          # features: shape (154, 29)
    y_reg = np.array([p["target_regression"] for p in pairs])  # เป้า regression: (154, 3)
    
    # แปลงชื่อแนวทางเป็นตัวเลข
    align_map = {"progressive": 0, "populist": 1, "conservative": 2, "others": 3}
    y_cls = np.array([align_map.get(p["target_classification"], 3) for p in pairs])
    
    print(f"  X shape: {X.shape}")
    print(f"  y_reg shape: {y_reg.shape}")
    print(f"  y_cls distribution: {Counter(y_cls.tolist())}")
    
    # ----- [3] Normalize ข้อมูล -----
    # StandardScaler ทำให้แต่ละ feature มี mean=0, std=1
    # สำคัญ! Transformer ทำงานได้ดีกว่าเมื่อ input scale เท่ากัน
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ----- [4] แบ่ง Train/Test -----
    # test_size=0.2 → 80% train, 20% test
    # stratify=y_cls → แต่ละ class มีสัดส่วนเท่ากันใน train และ test
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test, idx_train, idx_test = \
        train_test_split(X_scaled, y_reg, y_cls, np.arange(len(X)), test_size=0.2, random_state=42, stratify=y_cls)
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # ----- [5] Data Augmentation -----
    # ข้อมูลน้อยเกินไป (123 train samples) → เพิ่ม noise เล็กน้อย 3 รอบ
    # เหมือนทำให้โมเดลเห็นข้อมูล "เกือบเหมือนกัน" หลายแบบ → ป้องกัน overfit
    aug_X, aug_y_reg, aug_y_cls = [], [], []
    for i in range(3):  # ทำ 3 รอบ → ได้ 123 × 4 = 492 samples
        noise = np.random.normal(0, 0.05, X_train.shape)  # noise เล็กน้อย (σ=0.05)
        aug_X.append(X_train + noise)
        aug_y_reg.append(y_reg_train)
        aug_y_cls.append(y_cls_train)
    
    X_train_aug = np.vstack([X_train] + aug_X)
    y_reg_train_aug = np.vstack([y_reg_train] + aug_y_reg)
    y_cls_train_aug = np.concatenate([y_cls_train] + aug_y_cls)
    
    print(f"  After augmentation: {len(X_train_aug)} samples")
    
    # ----- [6] สร้าง DataLoader -----
    # DataLoader จัดการแบ่ง batch ให้อัตโนมัติ
    # batch_size=16 → เทรนทีละ 16 จังหวัด
    train_ds = ElectionDataset(X_train_aug, y_reg_train_aug, y_cls_train_aug)
    test_ds = ElectionDataset(X_test, y_reg_test, y_cls_test)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)  # สุ่มลำดับทุก epoch
    test_loader = DataLoader(test_ds, batch_size=32)
    
    input_size = X.shape[1]  # = 29
    
    # ----- [7] เทรน Transformer -----
    print("\n[2] Training Transformer...")
    transformer = TransformerModel(input_size)
    transformer, trans_hist = train_model(transformer, train_loader, test_loader, "transformer")
    result = evaluate_model(transformer, X_test, y_reg_test, y_cls_test)
    print(f"  ✅ Accuracy: {result['accuracy']:.1f}% | MAE: {result['mae']:.2f} | RMSE: {result['rmse']:.2f} | R²: {result['r2']:.3f}")
    print(f"  Per-class accuracy: {result['per_class_accuracy']}")
    
    # ============================================================
    # ทำนายผลเลือกตั้งปี 2573
    # ============================================================
    print("\n[3] Predicting 2573 election results...")

    # ใช้ข้อมูลปี 2569 เป็น input → โมเดลทำนายปี 2573
    X_pred = np.array([p["input"] for p in pred_input])
    # ใช้ scaler เดิม (ต้องไม่ fit ใหม่ เพราะต้องการ scale เดียวกับตอนเทรน)
    X_pred_scaled = scaler.transform(X_pred)
    
    align_names = ["progressive", "populist", "conservative", "others"]
    
    transformer.eval()
    with torch.no_grad():
        pred_reg, pred_cls = transformer(torch.FloatTensor(X_pred_scaled))
    
    pred_reg = pred_reg.numpy()
    pred_cls_labels = pred_cls.argmax(dim=1).numpy()  # เลือก class ที่มีโอกาสสูงสุด
    
    # รวมผลทำนายทุกจังหวัด
    predictions_2573 = []
    for i, p in enumerate(pred_input):
        predictions_2573.append({
            "province_id": p["province_id"],
            "province_name": p["province_name"],
            "region": p["region"],
            "predicted_progressive": round(float(pred_reg[i][0]), 2),     # % คะแนนก้าวหน้า
            "predicted_populist": round(float(pred_reg[i][1]), 2),         # % คะแนนประชานิยม
            "predicted_conservative": round(float(pred_reg[i][2]), 2),     # % คะแนนอนุรักษ์
            "predicted_winner": align_names[pred_cls_labels[i]],           # แนวทางที่ชนะ
        })
    
    # ----- บันทึกผลลัพธ์ -----
    output = {
        "model": "Transformer",
        "metrics": result,                   # ความแม่นยำของโมเดล
        "predictions_2573": predictions_2573, # ผลทำนาย 77 จังหวัด
        "training_history": trans_hist,       # ประวัติ loss (สำหรับ plot กราฟ)
        "test_indices": idx_test.tolist(),
        "test_provinces": [pairs[i]["province_name"] for i in idx_test],
    }
    
    output_path = os.path.join(DATA_DIR, "model_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")
    
    # สรุปผลการทำนาย
    print("\n" + "=" * 60)
    print("2573 Predictions (Transformer)")
    print("=" * 60)
    
    winner_dist = Counter(p["predicted_winner"] for p in predictions_2573)
    
    print("\nPredicted alignment distribution (77 provinces):")
    for align, count in winner_dist.most_common():
        print(f"  {align:20s}: {count:2d} จังหวัด ({count/77*100:.1f}%)")
    
    print("\nSample predictions:")
    for p in predictions_2573[:10]:
        print(f"  {p['province_name']:20s} | {p['predicted_winner']:15s} | "
              f"ก้าว {p['predicted_progressive']:5.1f}% "
              f"ปชน {p['predicted_populist']:5.1f}% "
              f"อนร {p['predicted_conservative']:5.1f}%")
