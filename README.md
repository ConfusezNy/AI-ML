# 🧠 Machine Learning Labs Collection

> A comprehensive collection of Machine Learning experiments covering Neural Networks, Support Vector Machines, and Convolutional Neural Networks with real-world datasets.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Technologies](#-technologies)
- [Installation](#-installation)
- [Datasets](#-datasets)
- [Labs Overview](#-labs-overview)
- [Usage](#-usage)
- [Results](#-results)
- [Contributing](#-contributing)

---

## 🎯 Overview

This project contains **18 comprehensive labs** across three major Machine Learning paradigms:

- **Neural Networks (NN)** - 6 labs covering fundamental feedforward networks
- **Support Vector Machines (SVM)** - 6 labs exploring kernel methods
- **Convolutional Neural Networks (CNN)** - 6 labs for deep learning on images and sequences

Each lab demonstrates practical applications on real-world datasets including medical imaging, time-series forecasting, and multi-class classification.

---

## 📁 Project Structure

```
MachineLearning/
│
├── shared_data/              # Centralized dataset repository (534 MB)
│   ├── iris/                 # Iris flower dataset (CSV + Images)
│   ├── bloodcells/           # Blood cell microscopy images (8 classes)
│   ├── covid19/              # COVID-19 time-series data (Thailand)
│   ├── fungi/                # Fungi species classification
│   ├── ppid/                 # Parasitic protozoan identification
│   ├── digits/               # Handwritten digits (MNIST)
│   └── faces/                # Face recognition dataset
│
├── NN_Lab/                   # Neural Network Labs
│   ├── LAB1_Digits/          # MNIST digit classification
│   ├── LAB2_FaceRecognition/ # Face recognition with NN
│   ├── LAB3_Iris/            # Iris species classification
│   ├── LAB4_Fungi/           # Fungi image classification
│   ├── LAB5_BloodCells/      # Blood cell type classification
│   └── LAB6_COVID19/         # COVID-19 time-series forecasting
│
├── SVM_Lab/                  # Support Vector Machine Labs
│   ├── LAB1_Iris_sklearn/    # Iris with sklearn (baseline)
│   ├── LAB2_Iris_csv/        # Iris from CSV with kernel comparison
│   ├── LAB3_Iris_Image/      # Iris image classification
│   ├── LAB4_PPID/            # Parasitic protozoan detection
│   ├── LAB5_BloodCells/      # Blood cell classification with SVM
│   └── LAB6_COVID19/         # COVID-19 regression with SVR
│
├── CNN_Lab/                  # Convolutional Neural Network Labs
│   ├── LAB1_Digits/          # CNN for digit recognition
│   ├── LAB2_FaceRecognition/ # Face recognition with CNN
│   ├── LAB3_Iris/            # Iris with CNN
│   ├── LAB4_Fungi/           # Fungi classification with CNN
│   ├── LAB5_BloodCells/      # Blood cell classification with CNN
│   └── LAB6_COVID19/         # COVID-19 forecasting with 1D-CNN
│
├── .gitignore                # Git ignore configuration
└── README.md                 # This file
```

### 🎨 Design Philosophy

- **Centralized Data Management**: All datasets stored in `shared_data/` to avoid duplication
- **Modular Architecture**: Each lab is self-contained and executable independently
- **Consistent Naming**: Standardized file naming across all labs
- **Clean Separation**: Code, data, and documentation clearly separated

---

## 🛠 Technologies

### Core Libraries

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.12+ | Programming language |
| **TensorFlow** | 2.15+ | Deep learning framework (NN & CNN) |
| **Keras** | 3.0+ | High-level neural networks API |
| **scikit-learn** | 1.4+ | Machine learning algorithms (SVM, preprocessing) |
| **NumPy** | 1.26+ | Numerical computing |
| **pandas** | 2.1+ | Data manipulation and analysis |
| **Matplotlib** | 3.8+ | Data visualization |
| **Pillow** | 10.0+ | Image processing |

### Machine Learning Techniques

- **Feedforward Neural Networks** (Dense layers, backpropagation)
- **Support Vector Machines** (Linear, Polynomial, RBF kernels)
- **Convolutional Neural Networks** (Conv2D, MaxPooling, 1D-CNN for sequences)
- **Data Preprocessing** (Normalization, standardization, train-test split)
- **Hyperparameter Tuning** (Grid search, learning rate optimization)

---

## 📦 Installation

### Prerequisites

- **Python 3.12+** installed ([Download](https://www.python.org/downloads/))
- **Git** installed ([Download](https://git-scm.com/downloads))
- 2-4 GB free disk space (for datasets)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/MachineLearning.git
   cd MachineLearning
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate environment**
   - **Windows (PowerShell)**
     ```powershell
     .\.venv\Scripts\Activate
     ```
   - **macOS/Linux**
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   pip install tensorflow numpy pandas matplotlib scikit-learn pillow
   ```

5. **Verify installation**
   ```bash
   python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
   python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
   ```

### 🐳 Docker (Optional)

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

All datasets are stored in `shared_data/` and shared across labs to save disk space.

| Dataset | Type | Size | Classes | Usage |
|---------|------|------|---------|-------|
| **Iris** | CSV + Images | ~5 MB | 3 species | Classification |
| **Blood Cells** | Images | ~200 MB | 8 types | Medical imaging |
| **COVID-19** | Time-series | ~87 MB | N/A | Forecasting |
| **Fungi** | Images | ~150 MB | 5 species | Multi-class |
| **PPID** | Microscopy | ~80 MB | 3 parasites | Medical detection |
| **Digits** | Built-in | sklearn | 10 digits | Baseline testing |
| **Faces** | External | TBD | Multiple | Recognition |

### 📥 Dataset Acquisition

After cloning, populate `shared_data/` with your datasets:

```bash
# Example: Download and extract datasets
cd shared_data/iris
# Add your Iris.csv and images here

cd ../bloodcells
# Add bloodcells_dataset folder here
```

> **Note**: Datasets are **not tracked** by Git due to size. Download separately or contact repository owner.

---

## 🔬 Labs Overview

### Neural Networks (NN_Lab)

| Lab | Focus | Key Concepts |
|-----|-------|--------------|
| **LAB1** | Digits Classification | Feedforward networks, softmax activation |
| **LAB2** | Face Recognition | Feature extraction, identity verification |
| **LAB3** | Iris Species | Multi-class classification, hyperparameter tuning |
| **LAB4** | Fungi Images | Image preprocessing, flatten layers |
| **LAB5** | Blood Cells | Medical imaging, class imbalance |
| **LAB6** | COVID-19 Forecast | Time-series, sliding window, regression |

### Support Vector Machines (SVM_Lab)

| Lab | Focus | Key Concepts |
|-----|-------|--------------|
| **LAB1** | Iris (sklearn) | Baseline comparison, kernel methods |
| **LAB2** | Iris (CSV) | Data loading, kernel comparison |
| **LAB3** | Iris Images | Image vectorization, linear kernel |
| **LAB4** | PPID | Medical detection, confusion matrix |
| **LAB5** | Blood Cells | Multi-class SVM, LinearSVC |
| **LAB6** | COVID-19 | SVR regression, time-series prediction |

### Convolutional Neural Networks (CNN_Lab)

| Lab | Focus | Key Concepts |
|-----|-------|--------------|
| **LAB1** | Digits | Conv2D, MaxPooling, filters |
| **LAB2** | Faces | Deep CNN, feature maps |
| **LAB3** | Iris | CNN on small images |
| **LAB4** | Fungi | Transfer learning concepts |
| **LAB5** | Blood Cells | Medical CNN, data augmentation |
| **LAB6** | COVID-19 | 1D-CNN for sequences |

---

## 🚀 Usage

### Running a Lab

Each lab is self-contained. Navigate to the lab folder and run:

```bash
# Example: Run CNN Digits lab
cd CNN_Lab/LAB1_Digits
python lab1_cnn_digits.py
```

### Expected Output

```
Training CNN: 1 Conv layers, 32 filters
Accuracy = 95.00%

Training CNN: 2 Conv layers, 64 filters
Accuracy = 97.50%
...
```

### Batch Execution

Run all labs in a category:

```bash
# Run all NN labs
for lab in NN_Lab/*/lab*.py; do python "$lab"; done
```

### Jupyter Notebooks (Optional)

Convert scripts to notebooks for interactive exploration:

```bash
pip install jupytext
jupytext --to notebook CNN_Lab/LAB1_Digits/lab1_cnn_digits.py
jupyter notebook
```

---

## 📈 Results

### Sample Accuracies

| Lab | NN | SVM | CNN |
|-----|-----|-----|-----|
| **Iris** | 96.7% | 100% | 98.3% |
| **Blood Cells** | 92.5% | 89.2% | 95.8% |
| **Fungi** | 88.3% | N/A | 91.7% |
| **PPID** | N/A | 94.5% | N/A |

### Visualizations

All labs include:
- **Accuracy plots** over epochs/iterations
- **Confusion matrices** for classification
- **Prediction vs. actual** for regression
- **Sample predictions** with correct/incorrect labels

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Coding Standards

- Follow PEP 8 style guide
- Add docstrings to functions
- Include comments for complex logic
- Test before committing

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Your Name** - *Initial work* - [GitHub](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- Dataset providers (UCI ML Repository, Kaggle, Our World in Data)
- TensorFlow and scikit-learn communities
- Course instructors and teaching assistants

---

## 📞 Contact

For questions or collaboration:

- **Email**: your.email@example.com
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/MachineLearning/issues)
- **LinkedIn**: [Your Profile](https://linkedin.com/in/yourprofile)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for Machine Learning education

</div>
