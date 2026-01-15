# Machine Learning Labs

โปรเจกต์รวม Lab สำหรับ Neural Networks, SVM และ CNN

## โครงสร้างโปรเจกต์

```
MachineLearning/
├── shared_data/              # Dataset ทั้งหมดรวมไว้ที่นี่
│   ├── iris/
│   │   ├── Iris.csv         # Iris dataset (CSV)
│   │   └── images/          # Iris images
│   │       ├── iris-setosa/
│   │       ├── iris-versicolour/
│   │       └── iris-virginica/
│   ├── bloodcells/
│   │   └── bloodcells_dataset/
│   │       ├── basophil/
│   │       ├── eosinophil/
│   │       ├── erythroblast/
│   │       ├── ig/
│   │       ├── lymphocyte/
│   │       ├── monocyte/
│   │       ├── neutrophil/
│   │       └── platelet/
│   ├── covid19/
│   │   └── owid-covid-data.csv
│   ├── fungi/
│   │   ├── train/           # H1, H2, H3, H5, H6
│   │   ├── test/
│   │   └── valid/
│   ├── ppid/
│   │   └── Dataset/
│   │       ├── Cryptosporidium cyst/
│   │       ├── Entamoeba histolytica/
│   │       └── Giardia cyst/
│   ├── digits/              # MNIST หรือ digits data
│   └── faces/               # Face recognition data
│
├── NN_Lab/                  # Neural Network Labs
│   ├── LAB1_Digits/
│   ├── LAB2_FaceRecognition/
│   ├── LAB3_Iris/
│   ├── LAB4_Fungi/
│   ├── LAB5_BloodCells/
│   └── LAB6_COVID19/
│
├── SVM_Lab/                 # Support Vector Machine Labs
│   ├── LAB1_Iris_sklearn/
│   ├── LAB2_Iris_csv/
│   ├── LAB3_Iris_Image/
│   ├── LAB4_PPID/
│   ├── LAB5_BloodCells/
│   └── LAB6_COVID19/
│
└── CNN_Lab/                 # Convolutional Neural Network Labs
    ├── LAB1_Digits/
    ├── LAB2_FaceRecognition/
    ├── LAB3_Iris/
    ├── LAB4_Fungi/
    ├── LAB5_BloodCells/
    └── LAB6_COVID19/
```

## ข้อดีของโครงสร้างแบบ shared_data

✅ **ไม่มีข้อมูลซ้ำซ้อน** - ประหยัดพื้นที่ disk  
✅ **แก้ไข/อัพเดท dataset ที่เดียว** - ทุก Lab ใช้ข้อมูลเดียวกัน  
✅ **เห็นภาพรวม dataset ชัดเจน** - รู้ว่ามี dataset อะไรบ้าง  
✅ **ง่ายต่อการ backup** - backup โฟลเดอร์เดียว  

## Dataset ที่ใช้ร่วมกัน

| Dataset | ใช้ใน Labs |
|---------|------------|
| **Iris** | NN_Lab/LAB3, SVM_Lab/LAB2, SVM_Lab/LAB3, CNN_Lab/LAB3 |
| **BloodCells** | NN_Lab/LAB5, SVM_Lab/LAB5, CNN_Lab/LAB5 |
| **COVID-19** | NN_Lab/LAB6, SVM_Lab/LAB6, CNN_Lab/LAB6 |
| **Fungi** | NN_Lab/LAB4, CNN_Lab/LAB4 |
| **PPID** | SVM_Lab/LAB4 |

## การใช้งาน

แต่ละ Lab จะอ้างอิง dataset จาก `shared_data/` เช่น:

```python
# Iris CSV
df = pd.read_csv("shared_data/iris/Iris.csv")

# Iris Images
dataset_path = "shared_data/iris/images"

# Blood Cells
base_path = "shared_data/bloodcells/bloodcells_dataset"

# COVID-19
csv_path = "shared_data/covid19/owid-covid-data.csv"

# Fungi
train_path = "shared_data/fungi/train"
test_path = "shared_data/fungi/test"
```

## การติดตั้ง Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow pillow
```

## หมายเหตุ

- โฟลเดอร์ `data/` ในแต่ละ Lab จะไม่มีไฟล์ dataset แล้ว
- ย้าย dataset ทั้งหมดไปที่ `shared_data/` แทน
- อัพเดท path ในโค้ดทั้งหมดเรียบร้อยแล้ว
