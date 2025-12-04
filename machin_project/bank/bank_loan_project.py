"""

- Preprocessing
- 70/15/15 split
- Linear Regression + KNN Regression
- Learning Curve
- Validation Curve
- Confusion Matrix
- ROC Curve
- AUC
- Accuracy
- Markdown Report

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    mean_squared_error, roc_curve, auc,
    confusion_matrix, accuracy_score
)

# ---------------- CONFIG ----------------# 
# Change file path to match your file location
DATA_PATH = r"D:\machin_project\bank\bankloan.csv"
TARGET_COL = "Personal.Loan"
ID_COLS = ["ID"]
OUTPUT_DIR = r"D:\machin_project\bank\bank_loan_outputs"
RANDOM_STATE = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ----------------------------------------


# ------------ Helper Functions ----------
def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, labels, title, fname):
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    save_fig(fig, fname)


def plot_roc_curve(y_true, y_score, title, fname):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}", linewidth=2)
    ax.plot([0,1],[0,1], linestyle="--", color="gray")

    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend()

    save_fig(fig, fname)
    return auc_score


# -------------- Load Data --------------
print("Loading:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Drop ID columns
for c in ID_COLS:
    if c in df.columns:
        df.drop(columns=[c], inplace=True)

n_rows, n_cols = df.shape
print("Rows:", n_rows, "Cols:", n_cols)

# Ensure target exists
if TARGET_COL not in df.columns:
    raise ValueError(f"Target {TARGET_COL} not found!")

# Target must be numeric 0/1
df[TARGET_COL] = df[TARGET_COL].astype(int)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(float)

numeric_cols = X.select_dtypes(include=['int64','float64','int32','float32']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# ----------- Preprocessing -------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Fit transform
X_pre = preprocessor.fit_transform(X)
feature_count = X_pre.shape[1]

print("Final feature count:", feature_count)

# ------------ Split 70/15/15 -----------
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_pre, y, test_size=0.15, random_state=RANDOM_STATE
)

val_relative = 0.15 / 0.85

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=val_relative,
    random_state=RANDOM_STATE
)

print("Train:", X_train.shape[0])
print("Val:", X_val.shape[0])
print("Test:", X_test.shape[0])


# -------- Linear Regression ------------
print("\nTraining Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)

y_lr_class = (y_pred_lr >= 0.5).astype(int)
acc_lr = accuracy_score(y_test, y_lr_class)
cm_lr = confusion_matrix(y_test, y_lr_class)

plot_confusion(cm_lr, ["0","1"], "Linear Regression CM", "cm_lr.png")
auc_lr = plot_roc_curve(y_test, y_pred_lr, "Linear Regression ROC", "roc_lr.png")


# ---------- Learning Curve -------------
fractions = np.linspace(0.1, 1.0, 10)
train_mse = []
val_mse = []

for frac in fractions:
    n_sub = max(5, int(frac * len(X_train)))
    idx = np.random.choice(len(X_train), n_sub, replace=False)
    model = LinearRegression()
    model.fit(X_train[idx], y_train.iloc[idx])
    train_mse.append(mean_squared_error(y_train.iloc[idx], model.predict(X_train[idx])))
    val_mse.append(mean_squared_error(y_val, model.predict(X_val)))

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(fractions, train_mse, label="Train MSE")
ax.plot(fractions, val_mse, label="Val MSE")
ax.legend()
ax.set_title("Learning Curve (LR)")
save_fig(fig, "learning_lr.png")


# ------------- KNN Model ---------------
print("\nTraining KNN (GridSearch)...")
param_grid = {"n_neighbors": list(range(1,21))}
knn = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring="neg_mean_squared_error")
knn.fit(X_train, y_train)

best_k = knn.best_params_["n_neighbors"]
best_knn = knn.best_estimator_

print("Best k:", best_k)

# Validation curve
k_vals = list(range(1,21))
mse_vals = -knn.cv_results_["mean_test_score"]

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(k_vals, mse_vals, marker="o")
ax.set_title("Validation Curve (KNN)")
save_fig(fig, "validation_knn.png")

y_pred_knn = best_knn.predict(X_test)
mse_knn = mean_squared_error(y_test, y_pred_knn)
rmse_knn = np.sqrt(mse_knn)

y_knn_class = (y_pred_knn >= 0.5).astype(int)
acc_knn = accuracy_score(y_test, y_knn_class)
cm_knn = confusion_matrix(y_test, y_knn_class)

plot_confusion(cm_knn, ["0","1"], "KNN Confusion Matrix", "cm_knn.png")
auc_knn = plot_roc_curve(y_test, y_pred_knn, "KNN ROC", "roc_knn.png")


# ---------- Save Report ---------------
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

print("\nDone! All outputs saved in:", OUTPUT_DIR)
