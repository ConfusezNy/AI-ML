# 🧠 Machine Learning Labs Collection

> โปรเจกต์รวมการทดลอง Machine Learning ครอบคลุม Neural Networks, Support Vector Machines และ Convolutional Neural Networks พร้อม dataset จริง

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 สารบัญ

- [ภาพรวมโปรเจกต์](#-ภาพรวมโปรเจกต์)
- [โครงสร้างโปรเจกต์](#-โครงสร้างโปรเจกต์)
- [เทคโนโลยีที่ใช้](#-เทคโนโลยีที่ใช้)
- [การติดตั้ง](#-การติดตั้ง)
- [Datasets](#-datasets)
- [ภาพรวม Labs](#-ภาพรวม-labs)
- [วิธีใช้งาน](#-วิธีใช้งาน)
- [ผลลัพธ์](#-ผลลัพธ์)
- [การมีส่วนร่วม](#-การมีส่วนร่วม)

---

## 🎯 ภาพรวมโปรเจกต์

โปรเจกต์นี้ประกอบด้วย **18 Labs** ครอบคลุม 3 แนวทาง Machine Learning หลัก:

- **Neural Networks (NN)** - 6 labs เรียนรู้ feedforward networks พื้นฐาน
- **Support Vector Machines (SVM)** - 6 labs ศึกษา kernel methods
- **Convolutional Neural Networks (CNN)** - 6 labs สำหรับ deep learning กับรูปภาพและ sequences

แต่ละ lab แสดงการประยุกต์ใช้จริงกับ dataset ในโลกจริง รวมถึง medical imaging, time-series forecasting และ multi-class classification

---

## 📁 โครงสร้างโปรเจกต์

```
MachineLearning/
│
├── shared_data/              # ที่เก็บ dataset กลาง (534 MB)
│   ├── iris/                 # Iris flower dataset (CSV + รูปภาพ)
│   ├── bloodcells/           # รูปภาพเซลล์เม็ดเลือด (8 classes)
│   ├── covid19/              # ข้อมูล COVID-19 time-series (ไทย)
│   ├── fungi/                # การจำแนกชนิดเห็ดรา
│   ├── ppid/                 # การระบุปรสิตโปรโตซัว
│   ├── digits/               # ตัวเลขเขียนด้วยมือ (MNIST)
│   └── faces/                # Face recognition dataset
│
├── NN_Lab/                   # Neural Network Labs
│   ├── LAB1_Digits/          # จำแนกตัวเลข MNIST
│   ├── LAB2_FaceRecognition/ # จดจำใบหน้าด้วย NN
│   ├── LAB3_Iris/            # จำแนกชนิดดอกไอริส
│   ├── LAB4_Fungi/           # จำแนกรูปภาพเห็ดรา
│   ├── LAB5_BloodCells/      # จำแนกชนิดเซลล์เม็ดเลือด
│   └── LAB6_COVID19/         # พยากรณ์ COVID-19 time-series
│
├── SVM_Lab/                  # Support Vector Machine Labs
│   ├── LAB1_Iris_sklearn/    # Iris กับ sklearn (baseline)
│   ├── LAB2_Iris_csv/        # Iris จาก CSV เปรียบเทียบ kernel
│   ├── LAB3_Iris_Image/      # จำแนกรูปภาพ Iris
│   ├── LAB4_PPID/            # ตรวจจับปรสิตโปรโตซัว
│   ├── LAB5_BloodCells/      # จำแนกเซลล์เม็ดเลือดด้วย SVM
│   └── LAB6_COVID19/         # พยากรณ์ COVID-19 ด้วย SVR
│
├── CNN_Lab/                  # Convolutional Neural Network Labs
│   ├── LAB1_Digits/          # CNN สำหรับจดจำตัวเลข
│   ├── LAB2_FaceRecognition/ # จดจำใบหน้าด้วย CNN
│   ├── LAB3_Iris/            # Iris ด้วย CNN
│   ├── LAB4_Fungi/           # จำแนกเห็ดราด้วย CNN
│   ├── LAB5_BloodCells/      # จำแนกเซลล์เม็ดเลือดด้วย CNN
│   └── LAB6_COVID19/         # พยากรณ์ COVID-19 ด้วย 1D-CNN
│
├── .gitignore                # การตั้งค่า Git ignore
└── README.md                 # ไฟล์นี้
```

### 🎨 หลักการออกแบบ

- **Centralized Data Management**: เก็บ dataset ทั้งหมดไว้ที่ `shared_data/` เพื่อไม่ให้ซ้ำซ้อน
- **Modular Architecture**: แต่ละ lab สามารถรันได้อิสระ
- **Consistent Naming**: ตั้งชื่อไฟล์แบบสม่ำเสมอทุก lab
- **Clean Separation**: แยกโค้ด, ข้อมูล และ documentation ชัดเจน

---

## 🛠 เทคโนโลยีที่ใช้

### Core Libraries

| Technology | Version | วัตถุประสงค์ |
|------------|---------|--------------|
| **Python** | 3.12+ | ภาษาโปรแกรม |
| **TensorFlow** | 2.15+ | Deep learning framework (NN & CNN) |
| **Keras** | 3.0+ | High-level neural networks API |
| **scikit-learn** | 1.4+ | Machine learning algorithms (SVM, preprocessing) |
| **NumPy** | 1.26+ | การคำนวณเชิงตัวเลข |
| **pandas** | 2.1+ | จัดการและวิเคราะห์ข้อมูล |
| **Matplotlib** | 3.8+ | Data visualization |
| **Pillow** | 10.0+ | ประมวลผลรูปภาพ |

### เทคนิค Machine Learning

- **Feedforward Neural Networks** (Dense layers, backpropagation)
- **Support Vector Machines** (Linear, Polynomial, RBF kernels)
- **Convolutional Neural Networks** (Conv2D, MaxPooling, 1D-CNN สำหรับ sequences)
- **Data Preprocessing** (Normalization, standardization, train-test split)
- **Hyperparameter Tuning** (Grid search, learning rate optimization)

---

## 📦 การติดตั้ง

### ความต้องการเบื้องต้น

- **Python 3.12+** ติดตั้งแล้ว ([ดาวน์โหลด](https://www.python.org/downloads/))
- **Git** ติดตั้งแล้ว ([ดาวน์โหลด](https://git-scm.com/downloads))
- พื้นที่ว่าง 2-4 GB (สำหรับ datasets)

### Quick Start

1. **Clone repository**
   ```bash
   git clone https://github.com/ConfusezNy/AI-ML.git
   cd MachineLearning
   ```

2. **สร้าง virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **เปิดใช้งาน environment**
   - **Windows (PowerShell)**
     ```powershell
     .\.venv\Scripts\Activate
     ```
   - **macOS/Linux**
     ```bash
     source .venv/bin/activate
     ```

4. **ติดตั้ง dependencies**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   pip install tensorflow numpy pandas matplotlib scikit-learn pillow
   ```

5. **ตรวจสอบการติดตั้ง**
   ```bash
   python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
   python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
   ```

### 🐳 Docker (ตัวเลือกเพิ่มเติม)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python"]
```

---

## 📊 Datasets

เก็บ dataset ทั้งหมดไว้ที่ `shared_data/` และใช้ร่วมกันระหว่าง labs เพื่อประหยัดพื้นที่

| Dataset | ประเภท | ขนาด | Classes | การใช้งาน |
|---------|--------|------|---------|-----------|
| **Iris** | CSV + รูปภาพ | ~5 MB | 3 ชนิด | Classification |
| **Blood Cells** | รูปภาพ | ~200 MB | 8 ชนิด | Medical imaging |
| **COVID-19** | Time-series | ~87 MB | N/A | Forecasting |
| **Fungi** | รูปภาพ | ~150 MB | 5 ชนิด | Multi-class |
| **PPID** | กล้องจุลทรรศน์ | ~80 MB | 3 ปรสิต | Medical detection |
| **Digits** | Built-in | sklearn | 10 ตัวเลข | Baseline testing |
| **Faces** | External | TBD | หลายคน | Recognition |

### 📥 การเตรียม Dataset

หลังจาก clone แล้ว ใส่ dataset ของคุณลงใน `shared_data/`:

```bash
# ตัวอย่าง: ดาวน์โหลดและแตก datasets
cd shared_data/iris
# เพิ่ม Iris.csv และรูปภาพของคุณที่นี่

cd ../bloodcells
# เพิ่มโฟลเดอร์ bloodcells_dataset ที่นี่
```

> **หมายเหตุ**: Dataset **ไม่ถูก track** โดย Git เนื่องจากขนาดใหญ่ ดาวน์โหลดแยกหรือติดต่อเจ้าของ repository

---

## 🔬 ภาพรวม Labs

### Neural Networks (NN_Lab)

| Lab | หัวข้อ | แนวคิดสำคัญ |
|-----|--------|-------------|
| **LAB1** | จำแนกตัวเลข | Feedforward networks, softmax activation |
| **LAB2** | จดจำใบหน้า | Feature extraction, identity verification |
| **LAB3** | ชนิดดอก Iris | Multi-class classification, hyperparameter tuning |
| **LAB4** | รูปภาพเห็ดรา | Image preprocessing, flatten layers |
| **LAB5** | เซลล์เม็ดเลือด | Medical imaging, class imbalance |
| **LAB6** | พยากรณ์ COVID-19 | Time-series, sliding window, regression |

### Support Vector Machines (SVM_Lab)

| Lab | หัวข้อ | แนวคิดสำคัญ |
|-----|--------|-------------|
| **LAB1** | Iris (sklearn) | Baseline comparison, kernel methods |
| **LAB2** | Iris (CSV) | Data loading, kernel comparison |
| **LAB3** | รูปภาพ Iris | Image vectorization, linear kernel |
| **LAB4** | PPID | Medical detection, confusion matrix |
| **LAB5** | เซลล์เม็ดเลือด | Multi-class SVM, LinearSVC |
| **LAB6** | COVID-19 | SVR regression, time-series prediction |

### Convolutional Neural Networks (CNN_Lab)

| Lab | หัวข้อ | แนวคิดสำคัญ |
|-----|--------|-------------|
| **LAB1** | ตัวเลข | Conv2D, MaxPooling, filters |
| **LAB2** | ใบหน้า | Deep CNN, feature maps |
| **LAB3** | Iris | CNN บนรูปภาพขนาดเล็ก |
| **LAB4** | เห็ดรา | Transfer learning concepts |
| **LAB5** | เซลล์เม็ดเลือด | Medical CNN, data augmentation |
| **LAB6** | COVID-19 | 1D-CNN สำหรับ sequences |

---

## 🚀 วิธีใช้งาน

### การรัน Lab

แต่ละ lab สามารถรันได้อิสระ ไปที่โฟลเดอร์ lab แล้วรัน:

```bash
# ตัวอย่าง: รัน CNN Digits lab
cd CNN_Lab/LAB1_Digits
python lab1_cnn_digits.py
```

### ผลลัพธ์ที่คาดหวัง

```
Training CNN: 1 Conv layers, 32 filters
Accuracy = 95.00%

Training CNN: 2 Conv layers, 64 filters
Accuracy = 97.50%
...
```

### รันทีละหลาย Labs

รัน labs ทั้งหมดในหมวดหมู่:

```bash
# รัน NN labs ทั้งหมด
for lab in NN_Lab/*/lab*.py; do python "$lab"; done
```

### Jupyter Notebooks (ตัวเลือก)

แปลง script เป็น notebook สำหรับการทดลองแบบ interactive:

```bash
pip install jupytext
jupytext --to notebook CNN_Lab/LAB1_Digits/lab1_cnn_digits.py
jupyter notebook
```

---

## 📈 ผลลัพธ์

### ตัวอย่าง Accuracies

| Lab | NN | SVM | CNN |
|-----|-----|-----|-----|
| **Iris** | 96.7% | 100% | 98.3% |
| **Blood Cells** | 92.5% | 89.2% | 95.8% |
| **Fungi** | 88.3% | N/A | 91.7% |
| **PPID** | N/A | 94.5% | N/A |

### Visualizations

ทุก lab มี:
- **กราฟ Accuracy** ตาม epochs/iterations
- **Confusion matrices** สำหรับ classification
- **การพยากรณ์ vs. ค่าจริง** สำหรับ regression
- **ตัวอย่างการทำนาย** พร้อมป้ายกำกับถูก/ผิด

---

## 🤝 การมีส่วนร่วม

ยินดีต้อนรับการมีส่วนร่วม! กรุณาทำตามขั้นตอนนี้:

1. Fork repository
2. สร้าง feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit การเปลี่ยนแปลง (`git commit -m 'Add AmazingFeature'`)
4. Push ไปยัง branch (`git push origin feature/AmazingFeature`)
5. เปิด Pull Request

### มาตรฐานการเขียนโค้ด

- ทำตาม PEP 8 style guide
- เพิ่ม docstrings ให้กับ functions
- ใส่ comments สำหรับ logic ที่ซับซ้อน
- ทดสอบก่อน commit

---

## 📄 License

โปรเจกต์นี้อยู่ภายใต้ MIT License - ดูไฟล์ [LICENSE](LICENSE) สำหรับรายละเอียด

---

## 👥 ผู้พัฒนา

<<<<<<< HEAD
- **ชื่อของคุณ** - *Initial work* - [GitHub](https://github.com/yourusername)
=======
- **ชื่อของคุณ** - *Initial work* - [GitHub](https://github.com/ConfusezNy)
>>>>>>> 87d8bcc44469d1203e0803f57bd06687fdd9151b

---

## 🙏 กิตติกรรมประกาศ

- ผู้ให้ dataset (UCI ML Repository, Kaggle, Our World in Data)
- ชุมชน TensorFlow และ scikit-learn
- อาจารย์ผู้สอนและผู้ช่วยสอน

---

## 📞 ติดต่อ

สำหรับคำถามหรือความร่วมมือ:

<<<<<<< HEAD
- **Email**: your.email@example.com
- **GitHub Issues**: [สร้าง issue](https://github.com/yourusername/MachineLearning/issues)
- **LinkedIn**: [โปรไฟล์ของคุณ](https://linkedin.com/in/yourprofile)
=======
- **Email**: natthachai2000.dev@gmail.com
- **LinkedIn**: [โปรไฟล์ของคุณ](https://www.linkedin.com/in/natthachai-yimchai-333642399/)
>>>>>>> 87d8bcc44469d1203e0803f57bd06687fdd9151b

---

<div align="center">

**⭐ กด Star repository นี้ถ้าคุณคิดว่ามีประโยชน์!**

สร้างด้วย ❤️ เพื่อการศึกษา Machine Learning

</div>
