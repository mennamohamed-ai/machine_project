# Bank Loan Approval Project - Detailed Guide

A comprehensive explanation of the bank loan prediction project, step by step.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Libraries Used](#libraries-used)
3. [Configuration](#configuration)
4. [Helper Functions](#helper-functions)
5. [Data Loading & Exploration](#data-loading--exploration)
6. [Data Preprocessing](#data-preprocessing)
7. [Data Splitting](#data-splitting)
8. [Model 1: Linear Regression](#model-1-linear-regression)
9. [Learning Curve](#learning-curve)
10. [Model 2: KNN Regression](#model-2-knn-regression)
11. [Results & Report](#results--report)

---

## 🎯 Project Overview

**Goal:** Predict whether a customer will accept a personal loan offer from the bank.

**Type:** Binary Classification/Regression Problem
- **Target Variable:** `Personal.Loan` (0 = No, 1 = Yes)
- **Dataset Size:** 5,000 samples
- **Features:** 12 (after preprocessing)

**Approach:**
- Compare two machine learning models: Linear Regression and KNN
- Evaluate performance using multiple metrics
- Generate visualizations and a comprehensive report

---

## 📦 Libraries Used

### Data Processing
```python
import numpy as np           # Numerical computations
import pandas as pd          # Data manipulation and analysis
```

### Machine Learning
```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
```

### Evaluation Metrics
```python
from sklearn.metrics import (
    mean_squared_error,      # MSE
    roc_curve, auc,          # ROC curve and AUC
    confusion_matrix,        # Confusion matrix
    accuracy_score           # Accuracy
)
```

### Visualization
```python
import matplotlib.pyplot as plt
```

---

## ⚙️ Configuration

```python
# File paths
DATA_PATH = r"D:\machin_project\bank\bankloan.csv"
OUTPUT_DIR = r"D:\machin_project\bank\bank_loan_outputs"

# Column configuration
TARGET_COL = "Personal.Loan"      # What we're predicting
ID_COLS = ["ID"]                  # Columns to drop

# Random seed for reproducibility
RANDOM_STATE = 42

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

**Why RANDOM_STATE?**
- Ensures the same random splits every time you run the code
- Makes results reproducible and comparable

---

## 🛠️ Helper Functions

### 1. `save_fig(fig, name)`

**Purpose:** Save matplotlib figures with consistent settings

```python
def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.tight_layout()           # Remove extra whitespace
    fig.savefig(path, dpi=150)   # Save at 150 DPI
    plt.close(fig)               # Free memory
```

**Usage:** `save_fig(fig, "my_plot.png")`

---

### 2. `plot_confusion(cm, labels, title, fname)`

**Purpose:** Create and save a confusion matrix heatmap

```python
def plot_confusion(cm, labels, title, fname):
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, cmap="Blues")      # Blue colormap
    
    # Add labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Add values in cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    
    save_fig(fig, fname)
```

**Confusion Matrix Interpretation:**
```
         Predicted: 0   Predicted: 1
Actual: 0    TN             FP        (True Negatives, False Positives)
Actual: 1    FN             TP        (False Negatives, True Positives)
```

---

### 3. `plot_roc_curve(y_true, y_score, title, fname)`

**Purpose:** Plot ROC curve and calculate AUC

```python
def plot_roc_curve(y_true, y_score, title, fname):
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)
    
    # Plot
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}", linewidth=2)
    ax.plot([0,1], [0,1], linestyle="--", color="gray")  # Diagonal line
    
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(title)
    ax.legend()
    
    save_fig(fig, fname)
    return auc_score
```

**ROC Curve Interpretation:**
- **AUC = 1.0:** Perfect model
- **AUC = 0.5:** Random guessing
- **AUC > 0.8:** Good model
- Curve closer to top-left = better performance

---

## 📊 Data Loading & Exploration

### Step 1: Load CSV File

```python
print("Loading:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
```

**Output:** Loads 5,000 rows × 13 columns

### Step 2: Remove ID Columns

```python
for c in ID_COLS:
    if c in df.columns:
        df.drop(columns=[c], inplace=True)
```

**Why?** ID columns don't help predict loan acceptance

### Step 3: Check Dataset Size

```python
n_rows, n_cols = df.shape
print("Rows:", n_rows, "Cols:", n_cols)  # Output: Rows: 5000 Cols: 12
```

### Step 4: Validate Target Column

```python
if TARGET_COL not in df.columns:
    raise ValueError(f"Target {TARGET_COL} not found!")

df[TARGET_COL] = df[TARGET_COL].astype(int)
```

### Step 5: Separate Features and Target

```python
X = df.drop(columns=[TARGET_COL])  # All features except target
y = df[TARGET_COL].astype(float)    # Target variable
```

### Step 6: Identify Column Types

```python
numeric_cols = X.select_dtypes(include=['int64','float64','int32','float32']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()

print("Numeric columns:", numeric_cols)        # Age, Income, etc.
print("Categorical columns:", categorical_cols) # Gender, Education, etc.
```

---

## 🔄 Data Preprocessing

**Why Preprocessing?**
- Machine learning models work better with clean, normalized data
- Different data types need different handling

### Pipeline Approach

We create separate processing pipelines for numeric and categorical data:

#### For Numeric Columns:

```python
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
```

**What it does:**
1. **Imputation:** Fill missing values with median
   - Median is robust to outliers
   - Better than mean for skewed distributions
   
2. **Scaling:** Normalize to mean=0, std=1
   - Formula: `X_scaled = (X - mean) / std`
   - Makes all features have similar ranges
   - Important for distance-based models (like KNN)

#### For Categorical Columns:

```python
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
```

**What it does:**
1. **Imputation:** Fill missing values with most frequent category
   - Uses the mode (most common value)
   
2. **One-Hot Encoding:** Convert categories to binary columns
   - Example: Gender [M, F, M] → Gender_M [1, 0, 1], Gender_F [0, 1, 0]
   - Creates n binary columns for n categories

#### Combine Both Pipelines:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

X_pre = preprocessor.fit_transform(X)
feature_count = X_pre.shape[1]  # 12 final features
```

---

## 📈 Data Splitting (70/15/15)

**Why Split Data?**
- **Training Set:** Used to train the model
- **Validation Set:** Used to tune hyperparameters
- **Test Set:** Used to evaluate final performance (never seen during training)

### Step 1: First Split (85% train+val, 15% test)

```python
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_pre, y, test_size=0.15, random_state=RANDOM_STATE
)
```

**Result:**
- Training + Validation: 4,250 samples (85%)
- Test: 750 samples (15%)

### Step 2: Second Split (70% train from remaining 85%)

```python
val_relative = 0.15 / 0.85  # ≈ 0.176

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=val_relative,
    random_state=RANDOM_STATE
)
```

**Final Result:**
```
Training:   3,500 samples (70%)
Validation:   750 samples (15%)
Test:         750 samples (15%)
Total:      5,000 samples
```

**Verification:**
```python
print("Train:", X_train.shape[0])  # 3500
print("Val:", X_val.shape[0])      # 750
print("Test:", X_test.shape[0])    # 750
```

---

## 🤖 Model 1: Linear Regression

### What is Linear Regression?

A model that finds the best-fit line through data:
```
y = w1*x1 + w2*x2 + ... + wn*xn + b
```

Where:
- `w` = weights (learned from training data)
- `b` = bias (intercept)
- `x` = features

### Training the Model

```python
print("\nTraining Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
```

**What happens:**
- Finds optimal weights that minimize prediction error
- Uses least squares method

### Making Predictions

```python
y_pred_lr = lr.predict(X_test)
```

**Output:** Continuous values between 0 and 1

### Converting to Classification

Since we need binary predictions (0 or 1):

```python
y_lr_class = (y_pred_lr >= 0.5).astype(int)
```

**Rule:** If prediction ≥ 0.5, classify as 1; otherwise, classify as 0

### Calculating Metrics

#### MSE (Mean Squared Error)

```python
mse_lr = mean_squared_error(y_test, y_pred_lr)
```

**Formula:**
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- Measures average squared difference
- Smaller is better
- Result: **0.0550**

#### RMSE (Root Mean Squared Error)

```python
rmse_lr = np.sqrt(mse_lr)
```

**Formula:**
$$RMSE = \sqrt{MSE}$$

- Same scale as original target
- Result: **0.2345**

#### Accuracy

```python
acc_lr = accuracy_score(y_test, y_lr_class)
```

**Formula:**
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

- Percentage of correct predictions
- Result: **92.13%**

#### Confusion Matrix

```python
cm_lr = confusion_matrix(y_test, y_lr_class)
```

Shows breakdown of correct and incorrect predictions:
```
[[TN  FP]
 [FN  TP]]
```

#### ROC AUC

```python
auc_lr = plot_roc_curve(y_test, y_pred_lr, "Linear Regression ROC", "roc_lr.png")
```

**Result:** **0.9671** (Excellent - close to 1.0)

### Generated Visualizations

1. **cm_lr.png** - Confusion matrix heatmap
2. **roc_lr.png** - ROC curve

---

## 📉 Learning Curve

**Purpose:** Detect overfitting/underfitting and show model improvement

### How It Works

```python
fractions = np.linspace(0.1, 1.0, 10)  # [0.1, 0.2, ..., 1.0]
train_mse = []
val_mse = []

for frac in fractions:
    # Use only 'frac' portion of training data
    n_sub = max(5, int(frac * len(X_train)))
    idx = np.random.choice(len(X_train), n_sub, replace=False)
    
    # Train model on subset
    model = LinearRegression()
    model.fit(X_train[idx], y_train.iloc[idx])
    
    # Calculate errors
    train_error = mean_squared_error(y_train.iloc[idx], model.predict(X_train[idx]))
    val_error = mean_squared_error(y_val, model.predict(X_val))
    
    train_mse.append(train_error)
    val_mse.append(val_error)
```

### Plot the Curve

```python
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(fractions, train_mse, label="Train MSE")
ax.plot(fractions, val_mse, label="Val MSE")
ax.legend()
ax.set_title("Learning Curve (LR)")
save_fig(fig, "learning_lr.png")
```

### Interpretation

- **Good Model:** Both curves converge and stay low
- **Underfitting:** Both curves high (model too simple)
- **Overfitting:** Train low, validation high (model memorizes)
- **In This Case:** Both curves decrease → model learning well

---

## 🤖 Model 2: KNN Regression with GridSearchCV

### What is KNN?

**K-Nearest Neighbors:**
- Predicts based on k nearest training samples
- For regression: average of neighbors' values
- For classification: majority voting

**Example (k=5):**
```
New point → Find 5 closest training points → Average their values
```

### Hyperparameter Tuning with GridSearchCV

**Problem:** How to choose the best k value?

**Solution:** Try all values and pick the best

```python
print("\nTraining KNN (GridSearchCV)...")
param_grid = {"n_neighbors": list(range(1,21))}  # Try k = 1 to 20

knn = GridSearchCV(
    KNeighborsRegressor(),
    param_grid,
    cv=5,                                  # 5-fold cross-validation
    scoring="neg_mean_squared_error"
)

knn.fit(X_train, y_train)
```

**What GridSearchCV does:**
1. Tests k = 1, 2, 3, ..., 20
2. For each k, performs 5-fold cross-validation
3. Calculates MSE for each configuration
4. Selects k with best (lowest) MSE

```python
best_k = knn.best_params_["n_neighbors"]
best_knn = knn.best_estimator_

print("Best k:", best_k)  # Output: 5
```

### Validation Curve

```python
k_vals = list(range(1,21))
mse_vals = -knn.cv_results_["mean_test_score"]  # Extract cross-val scores

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(k_vals, mse_vals, marker="o")
ax.set_title("Validation Curve (KNN)")
save_fig(fig, "validation_knn.png")
```

**Shows:** How MSE changes as k increases
- Low k: Overfitting (too sensitive to noise)
- High k: Underfitting (too smooth)
- Optimal k: Around 5

### Making Predictions and Evaluating

```python
y_pred_knn = best_knn.predict(X_test)

# Calculate metrics (same as Linear Regression)
mse_knn = mean_squared_error(y_test, y_pred_knn)     # 0.0386
rmse_knn = np.sqrt(mse_knn)                          # 0.1965

y_knn_class = (y_pred_knn >= 0.5).astype(int)
acc_knn = accuracy_score(y_test, y_knn_class)        # 94.80%
cm_knn = confusion_matrix(y_test, y_knn_class)

auc_knn = plot_roc_curve(y_test, y_pred_knn, "KNN ROC", "roc_knn.png")  # 0.9496
```

### Results Comparison

| Metric | Linear Regression | KNN (k=5) | Winner |
|--------|-------------------|-----------|--------|
| MSE | 0.0550 | 0.0386 | KNN ✅ |
| RMSE | 0.2345 | 0.1965 | KNN ✅ |
| Accuracy | 92.13% | 94.80% | KNN ✅ |
| AUC | 0.9671 | 0.9496 | Linear ✅ |

**Conclusion:** KNN performs better overall

---

## 📝 Results & Report

### Generating the Report

```python
report = f"""
# Bank Loan Approval – Results

### Dataset
- Rows: {n_rows}
- Columns: {n_cols}
- Final features: {feature_count}
- Train/Val/Test = {X_train.shape[0]} / {X_val.shape[0]} / {X_test.shape[0]}

---

## Linear Regression
- MSE: {mse_lr:.4f}
- RMSE: {rmse_lr:.4f}
- Accuracy: {acc_lr:.4f}
- AUC: {auc_lr:.4f}

## KNN Regression (k = {best_k})
- MSE: {mse_knn:.4f}
- RMSE: {rmse_knn:.4f}
- Accuracy: {acc_knn:.4f}
- AUC: {auc_knn:.4f}
"""

with open(os.path.join(OUTPUT_DIR, "report.md"), "w", encoding="utf-8") as f:
    f.write(report)
```

### Output Files

The script generates:

```
bank_loan_outputs/
├── report.md                  # Summary report
├── cm_lr.png                  # Confusion matrix (Linear Regression)
├── roc_lr.png                 # ROC curve (Linear Regression)
├── learning_lr.png            # Learning curve (Linear Regression)
├── cm_knn.png                 # Confusion matrix (KNN)
├── roc_knn.png                # ROC curve (KNN)
└── validation_knn.png         # Validation curve (KNN)
```

---

## 🎯 Key Takeaways

1. **Data Preprocessing is Crucial:**
   - Handling missing values
   - Scaling numeric features
   - Encoding categorical features

2. **Train/Val/Test Split:**
   - Prevents overfitting
   - Ensures fair evaluation
   - 70/15/15 is a common split

3. **Multiple Metrics Matter:**
   - Accuracy alone can be misleading
   - Use MSE, RMSE, Confusion Matrix, ROC AUC together

4. **Model Comparison:**
   - Train multiple models
   - Compare performance
   - Choose best for your use case

5. **Visualization is Important:**
   - Confusion matrices show error types
   - ROC curves show trade-offs
   - Learning curves show model behavior

---

## 🚀 How to Run

```bash
cd D:\machin_project\bank
python bank_loan_project.py
```

**Expected Output:**
```
Loading: D:\machin_project\bank\bankloan.csv
Rows: 5000 Cols: 12
Numeric columns: [...]
Categorical columns: [...]
Final feature count: 12
Train: 3500
Val: 750
Test: 750

Training Linear Regression...

Training KNN (GridSearchCV)...
Best k: 5

Done! All outputs saved in: D:\machin_project\bank\bank_loan_outputs
```

---

## 📚 Further Learning

- Explore feature importance
- Try other models (Random Forest, SVM, Neural Networks)
- Implement cross-validation manually
- Try different preprocessing techniques
- Handle class imbalance if present

---

**Last Updated:** December 2025
