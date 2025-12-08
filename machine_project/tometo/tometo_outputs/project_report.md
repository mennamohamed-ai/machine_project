# Tomato Classification Report

## Dataset
- Total samples: 14518
- Classes: 2 (Fresh=0, Rotten=1)
- Image size: 64x64
- Split (Train/Val/Test): 12214 / 1530 / 774
  - Train: Fresh=10751, Rotten=1463
  - Val: Fresh=1238, Rotten=292
  - Test: Fresh=617, Rotten=157

## Feature Extraction
- Method: HOG
- Pixels per cell: (8, 8)
- Cells per block: (2, 2)
- Orientations: 9
- Final feature vector length: 1764
- Normalization: L2-Hys (clipping 0.2)


## Logistic Regression
- Hyperparameters: max_iter=1000, solver='liblinear', threshold=0.5
- Accuracy: 85.66%
- ROC AUC: 0.8309
- Figures: confusion_logistic.png, roc_logistic.png, loss_curve_logistic.png

## KMeans
- Number of clusters: 2
- Cluster label assignment: majority vote on validation set
- Accuracy: 79.72%
- Figures: confusion_kmeans.png, roc_kmeans.png, inertia_vs_k.png, loss_curve_kmeans.png

All models and figures saved in output directory.