# Tomato Classification Project - Detailed Guide

A comprehensive explanation of the tomato image classification project, step by step.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Libraries Used](#libraries-used)
3. [Configuration](#configuration)
4. [Custom HOG Implementation](#custom-hog-implementation)
5. [HOG Feature Extraction](#hog-feature-extraction)
6. [Loading YOLO Dataset](#loading-yolo-dataset)
7. [Data Loading & Preprocessing](#data-loading--preprocessing)
8. [Data Scaling](#data-scaling)
9. [Model 1: Logistic Regression](#model-1-logistic-regression)
10. [Model 2: KMeans Clustering](#model-2-kmeans-clustering)
11. [Results & Report](#results--report)

---

## 🎯 Project Overview

**Goal:** Classify tomato images into two categories: Fresh or Rotten

**Type:** Binary Image Classification Problem
- **Target Classes:** Fresh (0), Rotten (1)
- **Total Images:** 14,518 samples
- **Image Size:** 64×64 pixels
- **Features:** 1,764 (HOG descriptors)

**Data Split:**
- Training: 12,214 images (84.1%)
- Validation: 1,530 images (10.5%)
- Test: 774 images (5.3%)

**Approach:**
1. Extract HOG (Histogram of Oriented Gradients) features from images
2. Train two models: Logistic Regression and KMeans
3. Compare performance and generate visualizations

---

## 📦 Libraries Used

### Environment & File Management
```python
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Limit CPU cores for parallelization
import glob                               # File pattern matching
from pathlib import Path                  # Path operations
from collections import Counter           # Count occurrences
```

### Image Processing
```python
from PIL import Image                     # Load and manipulate images
import numpy as np                        # Numerical operations
```

### Machine Learning
```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
```

### Model Evaluation
```python
from sklearn.metrics import (
    accuracy_score,      # Correctness
    confusion_matrix,    # TP/TN/FP/FN breakdown
    roc_curve, auc,      # ROC and AUC
    log_loss             # Logistic loss
)
```

### Model Persistence
```python
import joblib                             # Save/load models
```

### Visualization
```python
import matplotlib.pyplot as plt
```

---

## ⚙️ Configuration

```python
# Paths
DATA_DIR = r"D:\machin_project\tometo\tometo"
OUTPUT_DIR = r"D:\machin_project\tometo\tometo_outputs"

# Image settings
IMAGE_SIZE = (64, 64)          # Resize all images to 64x64

# HOG parameters
PIXELS_PER_CELL = (8, 8)       # Cell size
CELLS_PER_BLOCK = (2, 2)       # Block size
ORIENTATIONS = 9               # Number of direction bins

# Reproducibility
RANDOM_STATE = 42

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

**Why These HOG Settings?**
- **PIXELS_PER_CELL (8, 8):** Standard size, balances detail and speed
- **CELLS_PER_BLOCK (2, 2):** Captures local contrast normalization
- **ORIENTATIONS 9:** Sufficient to capture edge directions

---

## 🖼️ Custom HOG Implementation

### Why Custom HOG?

Instead of using scikit-learn's built-in HOG, we implement it from scratch to:
- Learn how feature extraction works
- Have more control over the process
- Understand image processing fundamentals

### Part 1: Convolution Helper (`_conv2d`)

```python
def _conv2d(img, kernel):
    """Apply 2D convolution to an image"""
    kh, kw = kernel.shape
    pad_h, pad_w = kh//2, kw//2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    out = np.zeros_like(img)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    
    return out
```

**What it does:**
1. Pads image edges to maintain size
2. Slides kernel over image
3. Computes element-wise multiplication and sum at each position

**Example:** Sobel filter
```
Kernel for horizontal edges:
[-1  0  1]
[-2  0  2]
[-1  0  1]
```

---

## 📐 HOG Feature Extraction

### Complete HOG Function

```python
def compute_hog(img, ppc=(8,8), cpb=(2,2), orientations=9):
```

#### Step 1: Compute Gradients

```python
kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])      # Sobel X
ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])      # Sobel Y

gx = _conv2d(img, kx)  # Horizontal gradients
gy = _conv2d(img, ky)  # Vertical gradients
```

**Purpose:** Find edges and their directions

#### Step 2: Calculate Magnitude and Angle

```python
mag = np.hypot(gx, gy)              # Magnitude: √(gx² + gy²)
ang = np.rad2deg(np.arctan2(gy, gx)) % 180  # Angle: atan2(gy, gx)
```

**Result:**
- `mag`: Edge strength (0-high)
- `ang`: Edge direction (0-180 degrees)

#### Step 3: Create Histograms

```python
ch, cw = ppc  # pixels per cell
ncy, ncx = img.shape[0]//ch, img.shape[1]//cw  # number of cells
bins = orientations  # 9 bins
bin_w = 180/bins  # degrees per bin = 20

cell_hist = np.zeros((ncy, ncx, bins))

for cy in range(ncy):
    for cx in range(ncx):
        # Get gradients in this cell
        m = mag[cy*ch:(cy+1)*ch, cx*cw:(cx+1)*cw].ravel()
        a = ang[cy*ch:(cy+1)*ch, cx*cw:(cx+1)*cw].ravel()
        
        # Assign to bins
        inds = np.floor(a/bin_w).astype(int)
        inds[inds==bins] = bins-1
        
        for b, val in zip(inds, m):
            cell_hist[cy, cx, b] += val
```

**Example for 64×64 image with 8×8 cells:**
- Number of cells: 8×8 = 64 cells
- Each cell has 9 bins (orientations)
- Each bin stores sum of gradient magnitudes

#### Step 4: Block Normalization (L2-Hys)

```python
by, bx = cpb  # cells per block
blocks = []
eps = 1e-6

for cy in range(ncy-by+1):
    for cx in range(ncx-bx+1):
        # Get 2×2 block of cells
        block = cell_hist[cy:cy+by, cx:cx+bx].ravel()
        
        # L2 normalization
        block = block / (np.linalg.norm(block) + eps)
        
        # Clipping (limit max values)
        block = np.minimum(block, 0.2)
        
        # Re-normalize after clipping
        block = block / (np.linalg.norm(block) + eps)
        
        blocks.append(block)

return np.hstack(blocks)
```

**Why Block Normalization?**
- Reduces influence of lighting changes
- Improves robustness to shadows
- L2-Hys = L2 norm + Clipping + Re-norm

### Final Output

For a 64×64 image:
- Cells: 8×8 = 64
- Blocks: 7×7 = 49 (sliding 2×2 blocks)
- Features per block: 2×2×9 = 36
- **Total features: 49×36 = 1,764 ✅**

---

## 📂 Loading YOLO Dataset

### Understanding YOLO Format

**YOLO labels format:** `class_id x_center y_center width height`

Example: `2 0.5 0.5 0.8 0.8` means:
- Class: 2 (Fresh)
- Center: (0.5, 0.5) in normalized coordinates
- Size: 0.8×0.8 of image

### Loading Function

```python
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

        # Map classes: Fresh=2→0, Rotten=3→1
        if 2 in ids:
            label = 0
        elif 3 in ids:
            label = 1
        else:
            continue

        entries.append((img_path, label))

    return entries
```

**Key Points:**
- Reads image and label files with same name
- Extracts class IDs from labels
- Maps YOLO classes to binary labels:
  - Class 2 (Fresh) → Label 0
  - Class 3 (Rotten) → Label 1

### Loading and Extracting Features

```python
def load_and_extract(entries):
    X, y = [], []
    for img_path, cls in entries:
        try:
            img = Image.open(img_path).convert("L")         # Grayscale
            img = img.resize(IMAGE_SIZE)                    # 64×64
            arr = np.array(img)/255.0                       # Normalize 0-1
            hog = compute_hog(arr, PIXELS_PER_CELL, ...)    # Extract HOG
            X.append(hog)
            y.append(cls)
        except:
            continue  # Skip corrupted images
    return np.array(X), np.array(y)
```

**Processing Steps:**
1. Open image
2. Convert to grayscale (single channel)
3. Resize to 64×64
4. Normalize pixel values to [0, 1]
5. Extract HOG features
6. Store feature vector and label

---

## 📊 Data Loading & Preprocessing

### Step 1: Load Dataset Splits

```python
print("Loading dataset...")

train = load_yolo_dataset(os.path.join(DATA_DIR, "train"))
val   = load_yolo_dataset(os.path.join(DATA_DIR, "val"))
test  = load_yolo_dataset(os.path.join(DATA_DIR, "test"))

print(f"train={len(train)}, val={len(val)}, test={len(test)}")
```

**Output:**
```
train=12214, val=1530, test=774
```

### Step 2: Extract Features

```python
X_train, y_train = load_and_extract(train)  # Shape: (12214, 1764)
X_val,   y_val   = load_and_extract(val)    # Shape: (1530, 1764)
X_test,  y_test  = load_and_extract(test)   # Shape: (774, 1764)

print("Final shapes:", X_train.shape, X_val.shape, X_test.shape)
```

### Step 3: Handle Empty Splits

If validation or test sets are empty, split from training:

```python
if len(X_val) == 0:
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
```

**Why stratify?** Maintains class distribution (proportion of Fresh/Rotten)

---

## 🔧 Data Scaling

### StandardScaler

Transforms features to have mean=0 and std=1:

$$X_{scaled} = \frac{X - \mu}{\sigma}$$

```python
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)
```

**Why Scale?**
- Logistic Regression works better with scaled features
- Prevents features with large ranges from dominating
- Improves numerical stability

**Important:** 
- `fit_transform` on training (learns mean/std)
- `transform` on validation/test (uses training stats)

### Save Scaler Parameters

```python
scaler_params = {
    "mean": scaler.mean_.tolist(),
    "var": scaler.var_.tolist()
}
```

For later use or model deployment.

---

## 🤖 Model 1: Logistic Regression

### What is Logistic Regression?

Despite the name, it's a **classification** algorithm, not regression.

**Formula:**
$$P(y=1|x) = \frac{1}{1 + e^{-(w \cdot x + b)}}$$

- `w`: weights (learned)
- `b`: bias
- Output: probability between 0 and 1

### Training

```python
clf = LogisticRegression(
    max_iter=1000,      # Maximum iterations
    solver="liblinear"  # Optimization algorithm
)
clf.fit(X_train_s, y_train)
```

**Hyperparameters:**
- `max_iter=1000`: Enough iterations for convergence
- `solver="liblinear"`: Good for binary classification

### Making Predictions

```python
y_proba = clf.predict_proba(X_test_s)[:,1]  # Probability of class 1
y_pred  = (y_proba >= 0.5).astype(int)      # Convert to binary
```

**Two ways to predict:**
1. **Probabilistic:** `predict_proba()` returns [P(y=0), P(y=1)]
2. **Binary:** `predict()` returns 0 or 1

### Evaluating Performance

#### Accuracy

```python
acc_lr = accuracy_score(y_test, y_pred)
acc_lr_percent = acc_lr * 100  # 85.66%
```

**Formula:**
$$Accuracy = \frac{TP + TN}{Total}$$

#### ROC AUC

```python
roc_auc_lr = auc(*roc_curve(y_test, y_proba)[:2])  # 0.8309
```

**Interpretation:**
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random guessing
- AUC = 0.83: Very good

#### Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)
```

**Visual representation:**
```
         Predicted: Fresh  Predicted: Rotten
Actual: Fresh        TN            FP
Actual: Rotten       FN            TP
```

### Visualization

#### Confusion Matrix Heatmap

```python
fig, ax = plt.subplots()
ax.imshow(cm, cmap="Blues")
ax.set_title("Logistic Confusion Matrix")
save_plot(fig, "confusion_logistic.png")
```

#### ROC Curve

```python
fig, ax = plt.subplots()
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax.plot(fpr, tpr)
ax.plot([0,1],[0,1],"--")  # Random classifier line
ax.set_title("ROC - Logistic Regression")
save_plot(fig, "roc_logistic.png")
```

#### Learning Curve

```python
fracs = np.linspace(0.1, 1.0, 8)  # [0.1, 0.23, ..., 1.0]
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
```

**Interpretation:**
- If curves converge: Good fit
- If gap between them: Possible overfitting
- If both high: Underfitting

---

## 🤖 Model 2: KMeans Clustering

### What is KMeans?

Unsupervised learning algorithm that:
1. Randomly initializes k cluster centers
2. Assigns points to nearest center
3. Updates centers as cluster means
4. Repeats until convergence

### Training KMeans

```python
km = KMeans(n_clusters=2, random_state=42)
km.fit(X_train_s)
```

**For this project:**
- k=2 (Fresh and Rotten)
- Learns to separate the two classes automatically

### Label Mapping

KMeans produces cluster labels (0, 1) but doesn't know which is Fresh vs Rotten.

**Solution: Majority voting on validation set**

```python
val_clusters = km.predict(X_val_s)
mapping = {}

for c in [0, 1]:
    idx = np.where(val_clusters == c)[0]
    if len(idx) == 0:
        mapping[c] = 0
    else:
        # What class is most common in this cluster?
        mapping[c] = Counter(y_val[idx]).most_common(1)[0][0]

print(mapping)  # Example: {0: 1, 1: 0}  # Cluster 0→Rotten, Cluster 1→Fresh
```

### Making Predictions

```python
test_clusters = km.predict(X_test_s)
km_pred = np.array([mapping[c] for c in test_clusters])

acc_km = accuracy_score(y_test, km_pred)
acc_km_percent = acc_km * 100  # 79.72%
```

### Visualizations

#### Confusion Matrix

```python
cm_k = confusion_matrix(y_test, km_pred)
fig, ax = plt.subplots()
ax.imshow(cm_k, cmap="Purples")
ax.set_title("KMeans Confusion Matrix")
save_plot(fig, "confusion_kmeans.png")
```

#### ROC Curve

For clustering, we use distance to cluster centers:

```python
centers = km.cluster_centers_
d0 = np.linalg.norm(X_test_s - centers[0], axis=1)
d1 = np.linalg.norm(X_test_s - centers[1], axis=1)
km_scores = -(d1)  # Score based on distance to cluster 1

fpr, tpr, _ = roc_curve(y_test, km_scores)
fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0,1],[0,1],"--")
ax.set_title("ROC - KMeans")
save_plot(fig, "roc_kmeans.png")
```

#### Inertia vs K

Shows optimal number of clusters:

```python
ks = range(1, 7)
inertias = []

for k in ks:
    km2 = KMeans(n_clusters=k, random_state=42)
    km2.fit(X_train_s)
    inertias.append(km2.inertia_)

fig, ax = plt.subplots()
ax.plot(ks, inertias, marker="o")
ax.set_title("Inertia vs K")
save_plot(fig, "inertia_vs_k.png")
```

**Interpretation:** Look for "elbow" - point where inertia stops decreasing sharply

---

## 💾 Saving Models

```python
joblib.dump(clf, os.path.join(OUTPUT_DIR, "logistic.joblib"))
joblib.dump(km, os.path.join(OUTPUT_DIR, "kmeans.joblib"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
```

**Why save models?**
- Use for future predictions without retraining
- Deployment to production
- Share with team members

**Loading saved models:**
```python
clf = joblib.load("logistic.joblib")
predictions = clf.predict(new_data)
```

---

## 📝 Results & Report

### Report Generation

```python
def generate_report(X_train, y_train, X_val, y_val, X_test, y_test,
                    acc_lr_percent, roc_auc_lr, acc_km_percent,
                    feature_vector_length, scaler_params):
    
    report_lines = [
        "# Tomato Classification Report\n",
        "## Dataset",
        f"- Total samples: {len(y_train)+len(y_val)+len(y_test)}",
        "- Classes: 2 (Fresh=0, Rotten=1)",
        f"- Image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
        f"- Split (Train/Val/Test): {len(y_train)} / {len(y_val)} / {len(y_test)}",
        ...
    ]
    
    report_text = "\n".join(report_lines)
    
    with open(os.path.join(OUTPUT_DIR, "project_report.md"), "w") as f:
        f.write(report_text)
```

### Output Report

```markdown
# Tomato Classification Report

## Dataset
- Total samples: 14,518
- Classes: 2 (Fresh=0, Rotten=1)
- Image size: 64x64
- Split: 12,214 / 1,530 / 774

## Feature Extraction
- Method: HOG
- Orientations: 9
- Final feature vector length: 1,764

## Logistic Regression
- Accuracy: 85.66%
- ROC AUC: 0.8309

## KMeans
- Accuracy: 79.72%
```

---

## 📊 Results Comparison

| Metric | Logistic Regression | KMeans | Winner |
|--------|-------------------|--------|--------|
| Accuracy | 85.66% | 79.72% | LR ✅ |
| ROC AUC | 0.8309 | - | LR ✅ |
| Interpretability | Medium | High | KM ✅ |
| Speed | Fast | Fast | - |

**Conclusion:** Logistic Regression performs better for this task

---

## 🎯 Key Insights

### About HOG Features
- Robust to lighting changes
- Captures edge information effectively
- Works well for object detection
- Computationally efficient

### About Logistic Regression
- Simple and interpretable
- Works well with pre-extracted features
- Fast training and prediction
- Good baseline model

### About KMeans
- Unsupervised (doesn't need labels)
- Useful for exploratory analysis
- Competitive accuracy but lower than supervised
- Good for clustering similar items

### Dataset Characteristics
- **Class Imbalance:** More fresh tomatoes (87.1%) than rotten (12.9%)
- **Impact:** May lead to bias toward Fresh prediction
- **Solution:** Use stratification, adjust class weights, or use different metrics

---

## 🚀 How to Run

```bash
cd D:\machin_project\tometo
python tomato_project.py
```

**Expected Output:**
```
Loading dataset...
train=12214, val=1530, test=774
Final shapes: (12214, 1764) (1530, 1764) (774, 1764)
Logistic Regression Accuracy: 85.66%
KMeans Accuracy: 79.72%
Report generated successfully!
All outputs saved successfully!
```

### Generated Files

```
tometo_outputs/
├── project_report.md
├── logistic.joblib
├── kmeans.joblib
├── scaler.joblib
├── confusion_logistic.png
├── roc_logistic.png
├── learning_curve_logistic.png
├── confusion_kmeans.png
├── roc_kmeans.png
└── inertia_vs_k.png
```

---

## 📚 Further Enhancements

1. **Image Augmentation:**
   - Rotation, flip, brightness changes
   - Increases dataset diversity

2. **Try Different Features:**
   - SIFT, SURF, ORB
   - Deep learning embeddings

3. **Advanced Models:**
   - Random Forest
   - Support Vector Machine (SVM)
   - Deep Neural Networks (CNN)

4. **Handle Class Imbalance:**
   - Oversampling minority class
   - Undersampling majority class
   - SMOTE (Synthetic Minority Over-sampling)

5. **Hyperparameter Tuning:**
   - GridSearchCV for LogisticRegression
   - Different HOG parameters
   - KMeans initialization methods

6. **Real-time Prediction:**
   - Web app for image upload
   - Mobile app integration
   - Deployment to cloud

---

## 🔧 Troubleshooting

### Issue: "Image file not found"
**Solution:** Check YOLO data directory structure and file names

### Issue: Memory error
**Solution:** Reduce batch size or resize images smaller

### Issue: Poor accuracy
**Solution:** 
- Try different HOG parameters
- Check for data quality issues
- Increase training data
- Use class weights for imbalance

---

## 📖 Concepts Reference

| Concept | Meaning |
|---------|---------|
| HOG | Histogram of Oriented Gradients - edge detection feature |
| Gradient | Rate of change in pixel intensity |
| Magnitude | Strength of edge |
| Orientation | Direction of edge |
| Binning | Grouping data into discrete ranges |
| Normalization | Making values comparable across features |
| Cross-validation | Splitting data for robust evaluation |
| Confusion Matrix | Table of TP/TN/FP/FN predictions |
| ROC Curve | Trade-off between TPR and FPR |
| AUC | Area Under ROC Curve - overall performance metric |

---

**Last Updated:** December 2025
