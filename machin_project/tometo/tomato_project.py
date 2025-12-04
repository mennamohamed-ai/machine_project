"""
HOG + Logistic Regression + KMeans
Binary Classification: Fresh (class=2) vs Rotten (class=3)
Outputs:
- confusion_logistic.png
- roc_logistic.png
- learning_curve_logistic.png
- confusion_kmeans.png
- roc_kmeans.png
- inertia_vs_k.png
- logistic.joblib
- kmeans.joblib
- scaler.joblib
- project_report.md
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import glob
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, log_loss
from sklearn.model_selection import train_test_split
import joblib

# ---------------- CONFIG ----------------

DATA_DIR = r"D:\machin_project\tometo\tometo"
OUTPUT_DIR = r"D:\machin_project\tometo\tometo_outputs"
IMAGE_SIZE = (64, 64)
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)
ORIENTATIONS = 9
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- HOG CONV ----------
def _conv2d(img, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh//2, kw//2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    return out

# ---------- HOG ----------
def compute_hog(img, ppc=(8,8), cpb=(2,2), orientations=9):
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

    gx = _conv2d(img, kx)
    gy = _conv2d(img, ky)
    mag = np.hypot(gx, gy)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180

    ch, cw = ppc
    ncy, ncx = img.shape[0]//ch, img.shape[1]//cw
    bins = orientations
    bin_w = 180/bins

    cell_hist = np.zeros((ncy, ncx, bins))
    for cy in range(ncy):
        for cx in range(ncx):
            m = mag[cy*ch:(cy+1)*ch, cx*cw:(cx+1)*cw].ravel()
            a = ang[cy*ch:(cy+1)*ch, cx*cw:(cx+1)*cw].ravel()
            inds = np.floor(a/bin_w).astype(int)
            inds[inds==bins] = bins-1
            for b, val in zip(inds, m):
                cell_hist[cy, cx, b] += val

    by, bx = cpb
    blocks = []
    eps = 1e-6

    for cy in range(ncy-by+1):
        for cx in range(ncx-bx+1):
            block = cell_hist[cy:cy+by, cx:cx+bx].ravel()
            block = block / (np.linalg.norm(block) + eps)
            block = np.minimum(block, 0.2)
            block = block / (np.linalg.norm(block) + eps)
            blocks.append(block)

    return np.hstack(blocks)

# ---------- LOAD YOLO DATA (Binary Fresh/Rotten) ----------
def load_yolo_dataset(split_dir):
    imgs = sorted(glob.glob(os.path.join(split_dir, "images", "*.*")))
    entries = []

    for img_path in imgs:
        stem = Path(img_path).stem
        label_path = os.path.join(split_dir, "labels", stem + ".txt")
        if not os.path.exists(label_path):
            continue

        ids = []
        with open(label_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                cls = int(float(line.split()[0]))
                ids.append(cls)

        if len(ids)==0:
            continue

        # Fresh = 2 → 0, Rotten = 3 → 1
        if 2 in ids:
            label = 0
        elif 3 in ids:
            label = 1
        else:
            continue

        entries.append((img_path, label))

    return entries

# ---------- EXTRACT HOG ----------
def load_and_extract(entries):
    X, y = [], []
    for img_path, cls in entries:
        try:
            img = Image.open(img_path).convert("L").resize(IMAGE_SIZE)
            arr = np.array(img)/255.0
            hog = compute_hog(arr, PIXELS_PER_CELL, CELLS_PER_BLOCK, ORIENTATIONS)
            X.append(hog)
            y.append(cls)
        except:
            continue
    return np.array(X), np.array(y)

# ---------- PLOT HELPERS ----------
def save_plot(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, name), dpi=150)
    plt.close(fig)

# ---------- REPORT ----------
def generate_report(X_train, y_train, X_val, y_val, X_test, y_test,
                    acc_lr_percent, roc_auc_lr, acc_km_percent,
                    feature_vector_length, scaler_params):

    def split_counts(y):
        from collections import Counter
        c = Counter(y)
        return f"Fresh={c.get(0,0)}, Rotten={c.get(1,0)}"

    report_lines = [
        "# Tomato Classification Report\n",
        "## Dataset",
        f"- Total samples: {len(y_train)+len(y_val)+len(y_test)}",
        "- Classes: 2 (Fresh=0, Rotten=1)",
        f"- Image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
        f"- Split (Train/Val/Test): {len(y_train)} / {len(y_val)} / {len(y_test)}",
        f"  - Train: {split_counts(y_train)}",
        f"  - Val: {split_counts(y_val)}",
        f"  - Test: {split_counts(y_test)}\n",
        "## Feature Extraction",
        "- Method: HOG",
        f"- Pixels per cell: {PIXELS_PER_CELL}",
        f"- Cells per block: {CELLS_PER_BLOCK}",
        f"- Orientations: {ORIENTATIONS}",
        f"- Final feature vector length: {feature_vector_length}",
        "- Normalization: L2-Hys (clipping 0.2)",
        f"\n",
        "## Logistic Regression",
        "- Hyperparameters: max_iter=1000, solver='liblinear', threshold=0.5",
        f"- Accuracy: {acc_lr_percent:.2f}%",
        f"- ROC AUC: {roc_auc_lr:.4f}",
        "- Figures: confusion_logistic.png, roc_logistic.png, learning_curve_logistic.png\n",
        "## KMeans",
        "- Number of clusters: 2",
        "- Cluster label assignment: majority vote on validation set",
        f"- Accuracy: {acc_km_percent:.2f}%",
        "- Figures: confusion_kmeans.png, roc_kmeans.png, inertia_vs_k.png\n",
        "All models and figures saved in output directory."
    ]

    report_text = "\n".join(report_lines)

    with open(os.path.join(OUTPUT_DIR, "project_report.md"), "w") as f:
        f.write(report_text)

    print("Report generated successfully!")

# ---------- MAIN ----------
def main():
    print("Loading dataset...")

    train = load_yolo_dataset(os.path.join(DATA_DIR, "train"))
    val   = load_yolo_dataset(os.path.join(DATA_DIR, "val"))
    test  = load_yolo_dataset(os.path.join(DATA_DIR, "test"))

    print(f"train={len(train)}, val={len(val)}, test={len(test)}")

    X_train, y_train = load_and_extract(train)
    X_val,   y_val   = load_and_extract(val)
    X_test,  y_test  = load_and_extract(test)

    if len(X_val)==0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )

    if len(X_test)==0:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )

    print("Final shapes:", X_train.shape, X_val.shape, X_test.shape)

    # ---------- SCALER ----------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    scaler_params = {"mean": scaler.mean_.tolist(), "var": scaler.var_.tolist()}

    # -------- LOGISTIC REGRESSION --------
    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    clf.fit(X_train_s, y_train)

    y_proba = clf.predict_proba(X_test_s)[:,1]
    y_pred  = (y_proba>=0.5).astype(int)

    acc_lr = accuracy_score(y_test, y_pred)
    acc_lr_percent = acc_lr * 100
    roc_auc_lr = auc(*roc_curve(y_test, y_proba)[:2])
    print(f"Logistic Regression Accuracy: {acc_lr_percent:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Logistic Confusion Matrix")
    save_plot(fig, "confusion_logistic.png")

    # ROC Curve
    fig, ax = plt.subplots()
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax.plot(fpr, tpr)
    ax.plot([0,1],[0,1],"--")
    ax.set_title("ROC - Logistic Regression")
    save_plot(fig, "roc_logistic.png")

    # Learning Curve
    fracs = np.linspace(0.1, 1.0, 8)
    train_loss, val_loss = [], []
    for frac in fracs:
        n = max(20, int(frac * len(X_train_s)))
        idx = np.random.choice(len(X_train_s), n, replace=False)
        model = LogisticRegression(max_iter=1000, solver="liblinear")
        model.fit(X_train_s[idx], y_train[idx])
        p_train = model.predict_proba(X_train_s[idx])[:,1]
        p_val   = model.predict_proba(X_val_s)[:,1]
        train_loss.append(log_loss(y_train[idx], p_train))
        val_loss.append(log_loss(y_val, p_val))

    fig, ax = plt.subplots()
    ax.plot(fracs, train_loss, label="Train")
    ax.plot(fracs, val_loss, label="Val")
    ax.set_title("Learning Curve - Logistic")
    ax.legend()
    save_plot(fig, "learning_curve_logistic.png")

    # -------- KMEANS --------
    km = KMeans(n_clusters=2, random_state=42)
    km.fit(X_train_s)

    val_clusters = km.predict(X_val_s)
    mapping = {}
    for c in [0,1]:
        idx = np.where(val_clusters==c)[0]
        if len(idx)==0:
            mapping[c] = 0
        else:
            mapping[c] = Counter(y_val[idx]).most_common(1)[0][0]

    test_clusters = km.predict(X_test_s)
    km_pred = np.array([mapping[c] for c in test_clusters])

    acc_km = accuracy_score(y_test, km_pred)
    acc_km_percent = acc_km * 100
    print(f"KMeans Accuracy: {acc_km_percent:.2f}%")

    # KMeans Confusion
    cm_k = confusion_matrix(y_test, km_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm_k, cmap="Purples")
    ax.set_title("KMeans Confusion Matrix")
    save_plot(fig, "confusion_kmeans.png")

    # KMeans ROC
    centers = km.cluster_centers_
    d0 = np.linalg.norm(X_test_s - centers[0], axis=1)
    d1 = np.linalg.norm(X_test_s - centers[1], axis=1)
    km_scores = -(d1)

    fpr, tpr, _ = roc_curve(y_test, km_scores)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0,1],[0,1],"--")
    ax.set_title("ROC - KMeans")
    save_plot(fig, "roc_kmeans.png")

    # inertia vs k
    ks = range(1,7)
    inertias = []
    for k in ks:
        km2 = KMeans(n_clusters=k, random_state=42)
        km2.fit(X_train_s)
        inertias.append(km2.inertia_)

    fig, ax = plt.subplots()
    ax.plot(ks, inertias, marker="o")
    ax.set_title("Inertia vs K")
    save_plot(fig, "inertia_vs_k.png")

    # -------- SAVE MODELS --------
    joblib.dump(clf, os.path.join(OUTPUT_DIR, "logistic.joblib"))
    joblib.dump(km, os.path.join(OUTPUT_DIR, "kmeans.joblib"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))

    # -------- GENERATE REPORT --------
    generate_report(X_train, y_train, X_val, y_val, X_test, y_test,
                    acc_lr_percent, roc_auc_lr, acc_km_percent,
                    feature_vector_length=X_train_s.shape[1],
                    scaler_params=scaler_params)

    print("All outputs saved successfully!")

if __name__ == "__main__":
    main()
