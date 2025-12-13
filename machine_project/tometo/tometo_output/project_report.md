# Tomato Classification Report

## Dataset
- Total samples: 14518
- Classes: 2 (Fresh=0, Rotten=1)
- Image size: 64x64
- Split (Train/Val/Test): 12214 / 1530 / 774
  - Train: Fresh=10751, Rotten=1463
  - Val  : Fresh=1238, Rotten=292
  - Test : Fresh=617, Rotten=157

## Feature Extraction
- Method: HOG (Histogram of Oriented Gradients).
- Pixels per cell: (8, 8)
- Cells per block: (2, 2)
- Orientations: 9
- Final feature vector length: 1764
- Normalization: L2-Hys with clipping at 0.2

## Class Weights (to handle class imbalance)
- Fresh class weight: 0.5680
- Rotten class weight: 4.1743

## Logistic Regression (supervised, with class weights)
- Model: LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced').
- Input: standardized HOG features (StandardScaler).
- Test Accuracy: 74.94%
- Test ROC AUC: 0.8026
### Rotten class metrics:
- Precision: 0.4263
- Recall   : 0.6815
- F1-Score : 0.5245
- Figures: confusion_logistic.png, roc_logistic.png, loss_curve_logistic.png

## KMeans (unsupervised clustering)
- Model: KMeans(n_clusters=2, random_state=42).
- Clusters are compared to labels only for evaluation; no label mapping is applied.
- Test Accuracy: 71.32%
- Figures: confusion_kmeans.png, roc_kmeans.png, loss_curve_kmeans.png

All models and figures are saved in the output directory.