# Beyond Visible Spectrum – Task 2  
## Self-Supervised Satellite Learning (ICPR 2026 – Kaggle)

This repository contains my solution for:

> **Task 2 – Self-Supervised Satellite Learning**  
> Kaggle: Beyond Visible Spectrum – AI for Agriculture (2026P2) 

The objective is to leverage **unlabelled multi-spectral Sentinel imagery** using **Self-Supervised Learning (SimCLR)** to improve crop disease classification under limited labelled data.

---

# Problem Summary

- Multi-class crop disease classification
- Severe class imbalance
- 12-band Sentinel satellite imagery
- Limited labelled samples
- Large unlabelled dataset available for representation learning

---

# Dataset Structure

The dataset used in this project is organized as follows:

```
data/
│
├── ICPR02/
│   └── kaggle/
│       ├── Aphid/
│       ├── Blast/
│       ├── RPH/
│       └── Rust/
│
├── archive/
│   └── share/
│       ├── train/
│       │   ├── rice/
│       │   ├── maize/
│       │   ├── soybean/
│       │   └── wheat/
│       │
│       └── val/
│
└── evaluation/
    └── test/
```

---

## Sentinel Band Files Per Sample

Each sample folder contains 12 `.tif` files:

```
B1.tif   B5.tif   B9.tif
B2.tif   B6.tif   B11.tif
B3.tif   B7.tif   B12.tif
B4.tif   B8.tif
         B8A.tif
```

These are stacked into a **12-channel tensor**.

---

# Pipeline Overview

The training process follows two major stages:

```
Unlabelled Data  →  SimCLR Pretraining  →  Encoder Weights
                            ↓
Labelled Data    →  Supervised Fine-Tuning  →  Final Classifier
```

---

# 1️⃣ Data Processing

### Band Alignment
- B4 used as spatial reference
- Reprojection if CRS available
- Bilinear resize fallback
- Missing bands filled with zeros

### Preprocessing
- Center crop / pad → 64×64
- Per-band normalization (mean/std computed)
- Clip values to [-3, 3]

---

# 2️⃣ Self-Supervised Pretraining (SimCLR)

## Model

- Backbone: **EfficientNet-B0**
- Input channels modified from 3 → 12
- Projection head: 2-layer MLP
- Loss: NT-Xent (temperature = 0.5)

## Training Setup

- Optimizer: AdamW
- Scheduler: CosineAnnealingLR
- Batch size: 32
- Epochs: 25
- Gradient clipping: 1.0

✔ Best encoder saved after SSL stage  
✔ Projection head discarded after pretraining  

---

# 3️⃣ Supervised Fine-Tuning

## Class Imbalance Handling

Due to imbalance:

- Rust ≈ 40
- Blast ≈ 75
- RPH ≈ 495
- Aphid ≈ smaller class

Mitigation strategies:

- WeightedRandomSampler
- Class-weighted CrossEntropy
- Label smoothing (0.1)

---

## Fine-Tuning Configuration

- Encoder initialized from SSL weights
- Encoder LR: 5e-5
- Classifier LR: 3e-4
- Scheduler: OneCycleLR
- Epochs: 60
- Early stopping (patience = 15)
- Dropout: 0.4

Best model saved as:

```
best_model.pth
```

---

# Model Architecture

## Encoder
EfficientNet-B0  
Input channels: 12  
Output embedding: 1280  

## SSL Projection Head
```
Linear → BatchNorm → ReLU → Linear (128 dim)
```

## Classification Head
```
Dropout
Linear(1280 → 256)
BatchNorm
ReLU
Dropout
Linear(256 → 4)
```

---

# Validation Results

After fine-tuning:

- Best validation accuracy recorded during training
- Full classification report generated:
  - Precision
  - Recall
  - F1-score
  - Support

(See notebook outputs for exact metrics.)

---

# Submission Pipeline

Final inference steps:

1. Load `best_model.pth`
2. Run inference on evaluation set
3. Convert predictions to class labels
4. Save submission file

```
submission.csv
```

Format:

```
ID,Category
```

Predicted class distribution printed for sanity check.

---

# Repository Structure

```
.
├── Codebase/
│   └── task2.ipynb
│
├── Results/
│   └── submission(agvis).csv
|   └── README.md
│
└── README.md
```

---

# Requirements

```
torch
timm
rasterio
scikit-image
scikit-learn
numpy
pandas
```

---

# Reproducibility Steps

1. Join Kaggle competition
2. Download dataset
3. Place dataset in correct folder structure
4. Run notebook:
   - Compute band statistics
   - Run SSL pretraining
   - Run supervised fine-tuning
   - Generate submission

---

# Key Contributions of This Solution

✔ Self-Supervised Learning on satellite imagery  
✔ 12-band spectral-aware augmentation  
✔ Strong imbalance mitigation  
✔ Stable training via gradient clipping  
✔ EfficientNet backbone adaptation  
✔ Robust band alignment handling  

---

# Conclusion

This solution demonstrates how **SimCLR-based self-supervised learning** significantly improves performance in:

- Low-labelled
- Multi-spectral
- Highly imbalanced
- Agricultural satellite datasets

By combining representation learning with careful fine-tuning, the model learns strong spectral-spatial features beyond the visible spectrum.

---

## Kaggle Competition Link

https://www.kaggle.com/competitions/beyond-visible-spectrum-ai-for-agriculture-2026p2
