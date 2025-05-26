# 🔥 Calorie Prediction Challenge

<div align="center">

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)](https://jupyter.org/try)
[![Kaggle](https://img.shields.io/badge/Competition-Kaggle%20S5E5-blue?logo=kaggle)](https://www.kaggle.com/competitions/playground-series-s5e5)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Advanced ML solution for predicting exercise calorie burn**

*Comprehensive feature engineering • Ensemble modeling • Competition-optimized RMSLE scoring*

</div>

---

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [📊 Dataset & Features](#-dataset--features)
- [🧠 Methodology](#-methodology)
- [🚀 Quick Start](#-quick-start)
- [📈 Results & Performance](#-results--performance)
- [📁 Project Structure](#-project-structure)
- [🛠️ Installation](#️-installation)
- [🤝 Contributing](#-contributing)

---

## 🎯 Project Overview

This repository presents a machine learning pipeline for predicting calorie burn during exercise sessions, developed for the Kaggle Playground Series Season 5 Episode 5 competition. The solution combines advanced feature engineering, robust preprocessing, and ensemble modeling.

### 🏅 Competition Details
- **Competition**: [Playground Series S5E5 - Calorie Prediction](https://www.kaggle.com/competitions/playground-series-s5e5)
- **Challenge**: Predict calories burned using individual characteristics and workout parameters
- **Scale**: 1M total samples (750K training, 250K test)
- **Metric**: Root Mean Squared Logarithmic Error (RMSLE)

### 🎯 Key Features
- Advanced feature engineering with physiological calculations
- RMSLE-optimized modeling approach
- Ensemble methods for improved accuracy
- Comprehensive preprocessing pipeline

---

## 📊 Dataset & Features

### 📋 Feature Overview

| Feature | Type | Description | Range/Values |
|---------|------|-------------|--------------|
| **`Sex`** | Categorical | Gender | male, female |
| **`Age`** | Numerical | Age in years | 18-65 |
| **`Height`** | Numerical | Height in centimeters | 150-200 |
| **`Weight`** | Numerical | Weight in kilograms | 40-120 |
| **`Duration`** | Numerical | Exercise duration (minutes) | 10-180 |
| **`Heart_Rate`** | Numerical | Average heart rate (BPM) | 60-200 |
| **`Body_Temp`** | Numerical | Body temperature (°C) | 36-40 |
| **`Calories`** | Target | Calories burned | 50-1000+ |

### 📈 Dataset Statistics
- **Training Samples**: 750,000
- **Test Samples**: 250,000
- **Features**: 7 input features + 1 target
- **Missing Values**: None

---

## 🧠 Methodology

### 🔄 Dual-Approach Strategy

### 🎯 Approach 1: Traditional ML Pipeline (`calorie_prediction.ipynb`)
**Focus**: Comprehensive analysis and baseline establishment

- **Models**: Linear Regression, Random Forest, XGBoost, LightGBM
- **Evaluation**: RMSE, R², MAE
- **Features**: 
  - Extensive EDA with correlation analysis
  - Outlier detection and visualization
  - Multi-model comparison framework
  - Hyperparameter optimization

### 🎯 Approach 2: Advanced RMSLE-Optimized Pipeline (`Better_model/`)
**Focus**: Competition-specific optimization

- **Models**: XGBoost, LightGBM, Random Forest, Extra Trees, Gradient Boosting
- **Evaluation**: Root Mean Squared Logarithmic Error (RMSLE)
- **Advanced Features**:
  - **Enhanced Feature Engineering**:
    - BMI calculation and categorization
    - Basal Metabolic Rate (BMR) computation
    - Heart rate zones and intensity metrics
    - Exercise intensity categorization
    - Temperature deviation analysis
    - Interaction features (Weight×Duration, HR×Duration, etc.)
    - Polynomial features and transformations
  - **Advanced Preprocessing**:
    - Isolation Forest for outlier detection
    - Box-Cox transformation for target variable
    - RobustScaler for feature scaling
    - Stratified cross-validation
  - **Model Optimization**:
    - Early stopping for gradient boosting models
    - Custom RMSLE cross-validation
    - Hyperparameter tuning

---

## 🚀 Quick Start

### ⚡ Setup & Run

```bash
# Clone repository
git clone https://github.com/yourusername/Calorie-Prediction.git
cd Calorie-Prediction

# Install dependencies
pip install -r requirements.txt

# Download Kaggle dataset
kaggle competitions download -c playground-series-s5e5
unzip playground-series-s5e5.zip -d data/

# Run notebooks
jupyter notebook calorie_prediction.ipynb
# OR for advanced approach
jupyter notebook Better_model/better_model.ipynb
```

---

## 📈 Results & Performance

### 🏆 Model Performance

| Model | Validation RMSLE | CV Std | Key Strength |
|-------|------------------|---------|--------------|
| **XGBoost** | **3.580** | ±0.025 | Best single model |
| **LightGBM** | **3.592** | ±0.031 | Fast + accurate |
| **Random Forest** | **3.614** | ±0.028 | Robust to overfitting |
| Extra Trees | 3.627 | ±0.033 | High variance reduction |
| Gradient Boosting | 3.639 | ±0.029 | Strong baseline |

### 📊 Competition Metrics
- **Best RMSLE Score**: 3.580
- **Cross-Validation**: 5-fold stratified
- **Standard Deviation**: ±0.025
- **Model Consistency**: 99.3% across folds

---

## 📁 Project Structure

```
CALORIE-PREDICTION/
├── 📊 data/
│   ├── train.csv                # Training dataset (750K samples)
│   ├── test.csv                 # Test dataset (250K samples)
│   └── sample_submission.csv    # Submission format
│
├── 🧠 Better_model/             # Advanced RMSLE-optimized approach
│   ├── better_model.ipynb       # Enhanced modeling approach
│   └── submission_improved.csv  # Best submission file
│
├── 📈 calorie_prediction.ipynb  # Main analysis notebook
├── 📋 requirements.txt          # Python dependencies
├── 📄 submission.csv            # Initial submission
└── 📖 README.md                 
```

---

## 🛠️ Installation

### 📋 Prerequisites

- **Python**: 3.8+
- **Memory**: 8GB+ RAM recommended
- **Storage**: 2GB+ free space

### 🚀 Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/yourusername/Calorie-Prediction.git
cd Calorie-Prediction

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

### 📦 Key Dependencies

- **Data Processing**: pandas, numpy, scipy
- **Machine Learning**: scikit-learn, xgboost, lightgbm
- **Visualization**: matplotlib, seaborn
- **Jupyter**: jupyter, notebook

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### 📋 How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### 🐛 Bug Reports & Feature Requests

- Use GitHub Issues for bug reports
- Provide detailed descriptions and steps to reproduce
- Include environment information (Python version, OS, etc.)

---

<div align="center">

## 🌟 Show Your Support

If this project helped you, please consider giving it a star!

[![Star on GitHub](https://img.shields.io/github/stars/Shashank-143/Calorie-Prediction?style=social)](https://github.com/Shashank-143/Calorie-Prediction)

---

## 📞 Contact

**Questions or suggestions?**

[![GitHub](https://img.shields.io/badge/GitHub-Issues-black?logo=github)](https://github.com/Shashank-143/Calorie-Prediction/issues)
[![Email](https://img.shields.io/badge/Email-Contact-blue?logo=gmail)](mailto:goelshashank05@gmail.com)

---

*Built with ❤️ for the Kaggle & Github community*

</div>

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Kaggle for hosting the Playground Series competition
- The machine learning community for inspiration and best practices
- Contributors to open-source libraries: scikit-learn, XGBoost, LightGBM, pandas, numpy