import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import glob
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from itertools import product

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc, log_loss,
    classification_report, f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import joblib

# ---------------- CONFIG ----------------
DATA_DIR = r"D:\machin_project\tometo\tometo"
OUTPUT_DIR = r"D:\machin_project\tometo\tometo_output"
IMAGE_SIZE = (64, 64)
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)
ORIENTATIONS = 9
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- HOG CONV ----------
def _conv2d(img, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.sum(padded[i:i + kh, j:j + kw] * kernel)
    return out

# ---------- HOG ----------
def compute_hog(img, ppc=(8, 8), cpb=(2, 2), orientations=9):
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    ky = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])

    gx = _conv2d(img, kx)
    gy = _conv2d(img, ky)
    mag = np.hypot(gx, gy)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180

    ch, cw = ppc
    ncy, ncx = img.shape[0] // ch, img.shape[1] // cw
    bins = orientations
    bin_w = 180 / bins

    cell_hist = np.zeros((ncy, ncx, bins))
    for cy in range(ncy):
        for cx in range(ncx):
            m = mag[cy * ch:(cy + 1) * ch, cx * cw:(cx + 1) * cw].ravel()
            a = ang[cy * ch:(cy + 1) * ch, cx * cw:(cx + 1) * cw].ravel()
            inds = np.floor(a / bin_w).astype(int)
            inds[inds == bins] = bins - 1
            for b, val in zip(inds, m):
                cell_hist[cy, cx, b] += val

    by, bx = cpb
    blocks = []
    eps = 1e-6

    for cy in range(ncy - by + 1):
        for cx in range(ncx - bx + 1):
            block = cell_hist[cy:cy + by, cx:cx + bx].ravel()
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

        if len(ids) == 0:
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

# ---------- HOG FEATURES (NO AUGMENTATION) ----------
def load_and_extract(entries):
    X, y = [], []
    for img_path, cls in entries:
        try:
            img = Image.open(img_path).convert("L").resize(IMAGE_SIZE)
            arr = np.array(img) / 255.0
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

def plot_loss_curve(test_losses, title, fname, x_values=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    if x_values is None:
        x_values = range(1, len(test_losses) + 1)
    ax.plot(x_values, test_losses, label="Test Loss",
            linewidth=2, marker='s', markersize=4, color='red')
    if isinstance(x_values, range) or (
        isinstance(x_values, list)
        and len(x_values) == len(test_losses) and x_values[0] == 1
    ):
        ax.set_xlabel("Epoch/Iteration")
    else:
        ax.set_xlabel("K (Number of Clusters)")
    ax.set_ylabel("Loss (Log Loss)" if "Logistic" in title else "Loss (Inertia)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, fname)

def plot_confusion(cm, title, fname,
                   class_names=("Fresh (0)", "Rotten (1)"), cmap="Blues"):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap=cmap)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                ha="center", va="center",
                color="black", fontsize=12, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_plot(fig, fname)

# ---------- REPORT ----------
def generate_report(
    X_train, y_train, X_val, y_val, X_test, y_test,
    acc_lr_percent, roc_auc_lr, acc_km_percent,
    feature_vector_length, scaler_params,
    class_weights, precision_rotten, recall_rotten, f1_rotten
):
    def split_counts(y):
        c = Counter(y)
        return f"Fresh={c.get(0, 0)}, Rotten={c.get(1, 0)}"

    report_lines = [
        "# Tomato Classification Report",
        "",
        "## Dataset",
        f"- Total samples: {len(y_train) + len(y_val) + len(y_test)}",
        "- Classes: 2 (Fresh=0, Rotten=1)",
        f"- Image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
        f"- Split (Train/Val/Test): {len(y_train)} / {len(y_val)} / {len(y_test)}",
        f"  - Train: {split_counts(y_train)}",
        f"  - Val  : {split_counts(y_val)}",
        f"  - Test : {split_counts(y_test)}",
        "",
        "## Feature Extraction",
        "- Method: HOG (Histogram of Oriented Gradients).",
        f"- Pixels per cell: {PIXELS_PER_CELL}",
        f"- Cells per block: {CELLS_PER_BLOCK}",
        f"- Orientations: {ORIENTATIONS}",
        f"- Final feature vector length: {feature_vector_length}",
        "- Normalization: L2-Hys with clipping at 0.2",
        "",
        "## Class Weights (to handle class imbalance)",
        f"- Fresh class weight: {class_weights[0]:.4f}",
        f"- Rotten class weight: {class_weights[1]:.4f}",
        "",
        "## Logistic Regression (supervised, with class weights)",
        "- Model: LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced').",
        "- Input: standardized HOG features (StandardScaler).",
        f"- Test Accuracy: {acc_lr_percent:.2f}%",
        f"- Test ROC AUC: {roc_auc_lr:.4f}",
        "### Rotten class metrics:",
        f"- Precision: {precision_rotten:.4f}",
        f"- Recall   : {recall_rotten:.4f}",
        f"- F1-Score : {f1_rotten:.4f}",
        "- Figures: confusion_logistic.png, roc_logistic.png, loss_curve_logistic.png",
        "",
        "## KMeans (unsupervised clustering)",
        "- Model: KMeans(n_clusters=2, random_state=42).",
        "- Clusters are compared to labels only for evaluation; no label mapping is applied.",
        f"- Test Accuracy : {acc_km_percent:.2f}%",
        "- Figures: confusion_kmeans.png, roc_kmeans.png, loss_curve_kmeans.png",
        "",
        "All models and figures are saved in the output directory."
    ]

    with open(os.path.join(OUTPUT_DIR, "project_report.md"), "w", encoding="utf-8") as f:
     f.write("\n".join(report_lines))


    print("Report generated successfully!")

# ---------- MAIN ----------
def main():
    print("Loading dataset...")

    train = load_yolo_dataset(os.path.join(DATA_DIR, "train"))
    val = load_yolo_dataset(os.path.join(DATA_DIR, "val"))
    test = load_yolo_dataset(os.path.join(DATA_DIR, "test"))

    print(f"train={len(train)}, val={len(val)}, test={len(test)}")

    X_train, y_train = load_and_extract(train)
    X_val, y_val = load_and_extract(val)
    X_test, y_test = load_and_extract(test)

    print("\nClass counts:")
    print("train:", Counter(y_train))
    print("val  :", Counter(y_val))
    print("test :", Counter(y_test))

    if len(X_val) == 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )

    if len(X_test) == 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )

    print("\nFinal shapes:", X_train.shape, X_val.shape, X_test.shape)

    # ---------- SCALER ----------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "var": scaler.var_.tolist()
    }

    # ---------- CLASS WEIGHTS ----------
    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    # ---------- LOGISTIC REGRESSION (WITH CLASS WEIGHTS) ----------
    clf_sgd = SGDClassifier(
    loss="log_loss",
    learning_rate="constant",
    eta0=0.01,
    max_iter=1,
    random_state=RANDOM_STATE,
    warm_start=True,
    alpha=0.0001
    
)


    test_losses_lr = []
    n_epochs = 50

    for epoch in range(n_epochs):
        clf_sgd.partial_fit(X_train_s, y_train, classes=np.unique(y_train))
        try:
            test_proba = clf_sgd.predict_proba(X_test_s)[:, 1]
            test_losses_lr.append(log_loss(y_test, test_proba))
        except:
            test_score = clf_sgd.decision_function(X_test_s)
            test_proba = 1 / (1 + np.exp(-test_score))
            test_losses_lr.append(log_loss(y_test, test_proba))

    clf = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        class_weight="balanced"
    )
    clf.fit(X_train_s, y_train)

    y_proba = clf.predict_proba(X_test_s)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc_lr = accuracy_score(y_test, y_pred)
    acc_lr_percent = acc_lr * 100
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba)
    roc_auc_lr = auc(fpr_lr, tpr_lr)

    precision_rotten = precision_score(y_test, y_pred, pos_label=1)
    recall_rotten = recall_score(y_test, y_pred, pos_label=1)
    f1_rotten = f1_score(y_test, y_pred, pos_label=1)

    print(f"\nLogistic Regression Accuracy: {acc_lr_percent:.2f}%")
    print(f"ROC AUC: {roc_auc_lr:.4f}")
    print("\nRotten class performance:")
    print(f"Precision: {precision_rotten:.4f}")
    print(f"Recall   : {recall_rotten:.4f}")
    print(f"F1-Score : {f1_rotten:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Fresh", "Rotten"]))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion(cm,
                   "Logistic Regression - Confusion Matrix",
                   "confusion_logistic.png")

    fig, ax = plt.subplots()
    ax.plot(fpr_lr, tpr_lr, label=f"ROC (AUC = {roc_auc_lr:.4f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", label="Random")
    ax.set_title("ROC Curve - Logistic Regression")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "roc_logistic.png")

    plot_loss_curve(test_losses_lr,
                    "Loss Curve - Logistic Regression",
                    "loss_curve_logistic.png")

    # ---------- KMEANS  ----------
    km = KMeans(n_clusters=2, random_state=42)
    km.fit(X_train_s)

    # Use raw cluster IDs as predictions (0/1) and compare directly
    km_pred = km.predict(X_test_s)

    acc_km = accuracy_score(y_test, km_pred)
    acc_km_percent = acc_km * 100
    print(f"\nKMeans Accuracy : {acc_km_percent:.2f}%")

    cm_k = confusion_matrix(y_test, km_pred)
    plot_confusion(cm_k,
                   "KMeans - Confusion Matrix (raw clusters)",
                   "confusion_kmeans.png",
                   cmap="Purples")

    centers = km.cluster_centers_
    d0 = np.linalg.norm(X_test_s - centers[0], axis=1)
    d1 = np.linalg.norm(X_test_s - centers[1], axis=1)
    km_scores = -(d1)

    fpr_k, tpr_k, _ = roc_curve(y_test, km_scores)
    fig, ax = plt.subplots()
    ax.plot(fpr_k, tpr_k)
    ax.plot([0, 1], [0, 1], "--")
    ax.set_title("ROC - KMeans")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    save_plot(fig, "roc_kmeans.png")

    ks_loss = list(range(1, 7))
    test_losses_km = []
    for k in ks_loss:
        km_temp = KMeans(n_clusters=k, random_state=42)
        km_temp.fit(X_train_s)
        test_inertia = sum(
            min(np.linalg.norm(x - c) ** 2 for c in km_temp.cluster_centers_)
            for x in X_test_s
        )
        test_losses_km.append(test_inertia)

    plot_loss_curve(test_losses_km,
                    "Loss Curve - KMeans",
                    "loss_curve_kmeans.png",
                    x_values=ks_loss)

    joblib.dump(clf, os.path.join(OUTPUT_DIR, "logistic.joblib"))
    joblib.dump(km, os.path.join(OUTPUT_DIR, "kmeans.joblib"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))

    generate_report(
        X_train, y_train, X_val, y_val, X_test, y_test,
        acc_lr_percent, roc_auc_lr, acc_km_percent,
        feature_vector_length=X_train_s.shape[1],
        scaler_params=scaler_params,
        class_weights=class_weights,
        precision_rotten=precision_rotten,
        recall_rotten=recall_rotten,
        f1_rotten=f1_rotten
    )

    print("\nAll outputs saved successfully!")

if __name__ == "__main__":
    main()
