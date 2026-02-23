# Beyond Visible Spectrum – Task 1  
## Multimodal Crop Disease Diagnosis  
ICPR 2026 – Kaggle Competition

This repository contains my solution for:

> **Task 1 – Multimodal Crop Disease Diagnosis**  
> Kaggle: Beyond Visible Spectrum – AI for Agriculture (2026)

---

# Competition Overview

Task 1 focuses on supervised crop disease classification using **multi-spectral Sentinel satellite imagery**.

Participants are required to:

- Process multi-band geospatial TIFF files
- Build a robust deep learning classifier
- Handle severe class imbalance
- Generalize to unseen evaluation samples

---

# Objective

Given a set of multi-band satellite image samples, predict the disease:

- Health  
- Rust
- Other 

Each sample consists of 12 Sentinel spectral bands stored as `.tif` files.

---

# 📂 Dataset Structure

```
data/
│
├── train/
│   ├── Health/
│   │   ├── sample_001/
│   │   ├── sample_002/
│   │   └── ...
│   │
│   ├── Rust/
│   ├── Other/
│
└── test/
    ├── sample_001/
    ├── sample_002/
    └── ...
```

---

## Sentinel Bands per Sample

Each sample folder contains the following 12 spectral bands:

```
B1.tif   B5.tif   B9.tif
B2.tif   B6.tif   B11.tif
B3.tif   B7.tif   B12.tif
B4.tif   B8.tif
         B8A.tif
```

These bands are stacked into a **12-channel tensor** before being passed into the model.

---

# Pipeline Overview

The training pipeline follows a standard supervised learning approach:

```
Raw Multi-Band Data
        ↓
Band Alignment & Normalization
        ↓
12-Channel Tensor Construction
        ↓
EfficientNet-B0 (Modified Input)
        ↓
Custom Classification Head
        ↓
Final Predictions
```

---

# 1️⃣ Data Processing

## Band Alignment

- B4 used as spatial reference
- Reprojection performed when CRS is available
- Bilinear resize fallback when reprojection not possible
- Missing bands filled with zeros

---

## Preprocessing

- Center crop / pad to fixed resolution (e.g., 64×64)
- Per-band mean and standard deviation normalization
- Value clipping for numerical stability

---

## Data Augmentation

Applied during training:

- Random horizontal & vertical flips
- Random rotations
- Gaussian noise injection
- Optional band dropout for robustness

---

# 2️⃣ Model Architecture

## Backbone

**EfficientNet-B0**

Modifications:
- Input channels changed from 3 → 12
- Pretrained weights adapted for multi-spectral input

Output embedding size: 1280

---

## Classification Head

```
Dropout
Linear (1280 → 256)
BatchNorm
ReLU
Dropout
Linear (256 → 4 classes)
```

---

# 3️⃣ Handling Class Imbalance

Observed imbalance among disease classes.

Mitigation strategies:

- WeightedRandomSampler
- Class-weighted CrossEntropy loss
- Label smoothing
- Augmentation strengthening for minority classes

---

# 4️⃣ Training Configuration

- Optimizer: AdamW
- Differential learning rates (encoder vs head)
- Scheduler: OneCycleLR
- Gradient clipping for stability
- Early stopping to prevent overfitting

Best model checkpoint saved as:

```
best_model.pth
```

---

# Kaggle Submission

Steps for generating submission:

1. Load best model checkpoint
2. Perform inference on test dataset
3. Convert class indices to labels
4. Export predictions to CSV

Submission format:

```
ID,Category
```

Generated file:

```
submission.csv
```

---

# Repository Structure

```
.
├── notebooks/
│   └── team-11-submssions.ipynb
│
├── submissions/
│   └── submission.csv
|   └── REAFME.md
|
└── README.md
```

---

# Requirements

```
torch
timm
rasterio
numpy
pandas
scikit-learn
scikit-image
```

---

# Reproducibility

1. Join Kaggle competition
2. Download dataset
3. Place data in correct directory structure
4. Run notebook:
   - Data preprocessing
   - Training
   - Validation
   - Submission generation

---

# Key Strengths of This Approach

✔ Multi-band spectral modeling (12 channels)  
✔ Robust band alignment handling  
✔ Strong class imbalance mitigation  
✔ EfficientNet backbone adaptation  
✔ Stable training via gradient clipping  
✔ Competitive leaderboard performance  

---

# 📌 Conclusion

This Task 1 solution demonstrates effective multi-spectral crop disease classification using:

- Modified EfficientNet architecture
- Careful spectral preprocessing
- Imbalance-aware training strategies

The model achieves competitive performance on the Kaggle leaderboard and generalizes well to unseen satellite imagery.

---

## Competition Link

https://www.kaggle.com/competitions/beyond-visible-spectrum-ai-for-agriculture-2026
