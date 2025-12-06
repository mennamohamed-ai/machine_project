# Machine Learning Projects

A collection of advanced machine learning projects including data processing, classification, and prediction tasks.

---

## 📋 Project Contents

The project contains two main projects:

1. **[Bank Loan Approval](#-bank-loan-approval)** - Predicting personal loan acceptance
2. **[Tomato Classification](#-tomato-classification)** - Classifying tomatoes (Fresh/Rotten)

---

## 🏦 Bank Loan Approval

A project aimed at predicting whether a customer will accept a personal loan offer from the bank.

### 📊 Dataset
- **Number of Rows:** 5,000
- **Number of Columns:** 13
- **Final Features After Processing:** 12
- **Split:** 70% training, 15% validation, 15% test
  - Training: 3,500 samples
  - Validation: 750 samples
  - Test: 750 samples

### 🔧 Data Preprocessing
- Removal of ID columns
- **Numeric Data Scaling:** StandardScaler
- **Categorical Data Encoding:** OneHotEncoder
- **Missing Values Handling:**
  - For numeric data: Median imputation
  - For categorical data: Most frequent value imputation

### 🤖 Models Used

#### 1. Linear Regression
- **MSE:** 0.0550
- **RMSE:** 0.2345
- **Accuracy:** 92.13%
- **AUC:** 0.9671
- **Generated Visualizations (All on Test Data):**
  - `cm_lr.png` - Confusion matrix
  - `roc_lr.png` - ROC curve
  - `loss_curve_lr.png` - Loss curve

#### 2. KNN Regression (GridSearch)
- **Best k:** 5
- **MSE:** 0.0386
- **RMSE:** 0.1965
- **Accuracy:** 94.80%
- **AUC:** 0.9496
- **Generated Visualizations (All on Test Data):**
  - `cm_knn.png` - Confusion matrix
  - `roc_knn.png` - ROC curve
  - `loss_curve_knn.png` - Loss curve

### 📂 File Structure
```
bank/
├── bank_loan_project.py      # Main project code
├── bankloan.csv             # Input data
└── bank_loan_outputs/
    ├── report.md            # Results report
    ├── cm_lr.png            # Confusion matrix for linear regression (test data)
    ├── roc_lr.png           # ROC curve for linear regression (test data)
    ├── loss_curve_lr.png    # Loss curve for linear regression (test data)
    ├── cm_knn.png           # Confusion matrix for KNN (test data)
    ├── roc_knn.png          # ROC curve for KNN (test data)
    └── loss_curve_knn.png   # Loss curve for KNN (test data)
```

### ▶️ How to Run
```bash
python bank_loan_project.py
```

---

## 🍅 Tomato Classification

A project for classifying tomatoes into two categories: **Fresh (0)** or **Rotten (1)** using image processing and machine learning techniques.

### 📊 Dataset
- **Total Images:** 14,518
- **Classes:** 2 (Fresh = 0, Rotten = 1)
- **Image Size:** 64×64 pixels
- **Split:**
  - Training: 12,214 samples (Fresh: 10,751, Rotten: 1,463)
  - Validation: 1,530 samples (Fresh: 1,238, Rotten: 292)
  - Test: 774 samples (Fresh: 617, Rotten: 157)

### 🖼️ Feature Extraction (HOG - Histogram of Oriented Gradients)
- **Number of Orientations:** 9
- **Pixels per Cell:** (8, 8)
- **Cells per Block:** (2, 2)
- **Feature Vector Length:** 1,764
- **Normalization:** L2-Hys with clipping value 0.2
- **Scaling:** StandardScaler

### 🤖 Models Used

#### 1. Logistic Regression
- **Hyperparameters:**
  - max_iter: 1000
  - solver: 'liblinear'
  - threshold: 0.5
- **Accuracy:** 85.66%
- **ROC AUC:** 0.8309
- **Generated Visualizations (All on Test Data):**
  - `confusion_logistic.png` - Confusion matrix
  - `roc_logistic.png` - ROC curve
  - `loss_curve_logistic.png` - Loss curve

#### 2. KMeans Clustering
- **Number of Clusters:** 2
- **Label Assignment:** Majority voting on validation set
- **Accuracy:** 79.72%
- **Generated Visualizations (All on Test Data):**
  - `confusion_kmeans.png` - Confusion matrix
  - `roc_kmeans.png` - ROC curve
  - `loss_curve_kmeans.png` - Loss curve

### 📂 File Structure
```
tometo/
├── tomato_project.py        # Main project code
├── tometo/                  # Raw data folder
│   ├── data.yaml           # Data configuration file
│   ├── train/              # Training data
│   │   ├── images/
│   │   └── labels/         # YOLO labels
│   ├── val/                # Validation data
│   │   ├── images/
│   │   └── labels/
│   └── test/               # Test data
│       ├── images/
│       └── labels/
└── tometo_outputs/
    ├── project_report.md   # Results report
    ├── logistic.joblib     # Saved logistic regression model
    ├── kmeans.joblib       # Saved KMeans model
    ├── scaler.joblib       # Saved data scaler
    ├── confusion_logistic.png  # Confusion matrix (test data)
    ├── roc_logistic.png        # ROC curve (test data)
    ├── loss_curve_logistic.png # Loss curve (test data)
    ├── confusion_kmeans.png    # Confusion matrix (test data)
    ├── roc_kmeans.png          # ROC curve (test data)
    └── loss_curve_kmeans.png   # Loss curve (test data)
```

### ▶️ How to Run
```bash
python tomato_project.py
```

---

## 📋 Requirements

### Required Libraries:
```
numpy
pandas
matplotlib
scikit-learn
joblib
Pillow
```

### Installation:
```bash
pip install numpy pandas matplotlib scikit-learn joblib pillow
```

---

## 📈 Comparative Results

### Bank Loan Project
| Model | Accuracy | AUC | RMSE |
|-------|----------|-----|------|
| Linear Regression | 92.13% | 0.9671 | 0.2345 |
| KNN (k=5) | 94.80% | 0.9496 | 0.1965 |

**Best Model:** KNN with 94.80% accuracy

### Tomato Classification
| Model | Accuracy | ROC AUC |
|-------|----------|---------|
| Logistic Regression | 85.66% | 0.8309 |
| KMeans | 79.72% | - |

**Best Model:** Logistic Regression with 85.66% accuracy

---

## 🔍 Technical Notes

### Bank Loan Project
- Used 70/15/15 split to obtain separate training, validation, and test data
- All model evaluations (loss curve, accuracy, confusion matrix, ROC curve) are performed on test data only
- KNN model achieved better performance with higher accuracy and lower squared error

### Tomato Classification
- HOG was used as a powerful feature extractor for images
- Advanced L2-Hys normalization was applied to improve feature quality
- All model evaluations (loss curve, accuracy, confusion matrix, ROC curve) are performed on test data only
- Logistic Regression showed better performance than KMeans in this scenario
- Data imbalance (more fresh tomatoes) may affect performance

---
