# Task 1: ML Classification Project — Credit Card Fraud Detection

**InternSpark ML Internship | Task 1**

## Project Overview
A supervised classification project to detect fraudulent credit card transactions using Logistic Regression and Random Forest, with SMOTE for handling class imbalance.

## Dataset
- **Name:** Credit Card Fraud Detection
- **Source:** [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **File:** `creditcard.csv` (download from Kaggle, ~150MB — stored on Google Drive)
- **Samples:** 284,807 transactions | **Fraud rate:** 0.17%

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
```

## Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/task1-ml-classification.git
cd task1-ml-classification
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

### 4. Download Dataset
- Download `creditcard.csv` from the [Google Drive link](YOUR_DRIVE_LINK) or from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Place it in the project root directory (same folder as the notebook)

## Running the Notebook

### Option A: Jupyter Notebook
```bash
jupyter notebook task1_ml_classification.ipynb
```
Then run all cells: **Kernel → Restart & Run All**

### Option B: JupyterLab
```bash
jupyter lab task1_ml_classification.ipynb
```

## Project Structure
```
task1-ml-classification/
│
├── task1_ml_classification.ipynb   # Main notebook
├── creditcard.csv                  # Dataset (download separately)
├── README.md                       # This file
│
└── outputs/                        # Auto-generated plots
    ├── eda_class_distribution.png
    ├── eda_correlation_heatmap.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── model_comparison.png
    └── feature_importance.png
```

## Approach Summary

| Step | Details |
|---|---|
| Preprocessing | StandardScaler on Time & Amount features |
| Imbalance Handling | SMOTE oversampling on training set only |
| Train/Test Split | 80/20 stratified split |
| Cross-Validation | 5-Fold Stratified K-Fold |
| Algorithms | Logistic Regression vs Random Forest |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC |

## Results Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.97 | ~0.06 | ~0.92 | ~0.11 | ~0.97 |
| Random Forest | ~0.99 | ~0.85 | ~0.82 | ~0.84 | ~0.97 |

**Winner: Random Forest** — Best F1 Score, handles non-linear patterns well.
