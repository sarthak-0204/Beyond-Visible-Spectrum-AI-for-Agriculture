# Beyond Visible Spectrum – AI for Agriculture (ICPR 2026)

This repository contains our solutions for the Kaggle competition:

> **Beyond Visible Spectrum: AI for Agriculture (2026)**  
> ICPR 2026 Grand Challenge

The competition focuses on leveraging **multi-spectral satellite imagery** for crop disease diagnosis using advanced deep learning techniques.

---

# Competition Overview

Modern agriculture increasingly relies on satellite-based monitoring systems. This competition challenges participants to:

- Process multi-band Sentinel satellite imagery
- Develop robust disease classification models
- Handle strong class imbalance
- Improve generalization to unseen geospatial data
- Explore self-supervised learning under limited labels (Task 2)

The dataset consists of **12-band Sentinel imagery** stored as `.tif` files.

---

# Sentinel Spectral Bands

Each sample contains the following bands:

```
B1   B5   B9
B2   B6   B11
B3   B7   B12
B4   B8
     B8A
```

These are stacked into a **12-channel tensor** before being passed into deep learning models.

---

# Repository Structure

```
.
├── Task1/
│   ├── notebooks/
│   ├── results/
│   └── README.md
│
├── Task2/
│   ├── notebooks/
│   ├── results/
│   └── README.md
│
└── README.md  
```

---

# Tasks Overview

---

## Task 1 – Multimodal Crop Disease Diagnosis

Supervised classification of crop conditions using labelled multi-spectral imagery.

### Classes:
- Rust
- Healthy
- Other

### Approach Summary:

- EfficientNet-B0 backbone (modified for 12-channel input)
- Custom classification head
- Band alignment & normalization pipeline
- Class-weighted loss
- WeightedRandomSampler
- OneCycleLR scheduler
- Early stopping
- Robust augmentation strategy

Detailed implementation and results are available in:

```
Task1/README.md
Task1/results/README.md
```

---

## Task 2 – Self-Supervised Satellite Learning

This task addresses limited labelled data by incorporating **Self-Supervised Learning (SimCLR)**.

### Classes:
- Aphid
- Blast
- RPH
- Rust

### Two-Stage Training Pipeline:

```
Unlabelled Data
        ↓
SimCLR Pretraining
        ↓
Encoder Weights
        ↓
Supervised Fine-Tuning
```

### Approach Summary:

- EfficientNet-B0 (12-band input)
- SimCLR projection head
- NT-Xent loss
- Spectral-aware augmentations
- Class imbalance mitigation
- Fine-tuning with differential learning rates

Detailed implementation and results are available in:

```
Task2/README.md
Task2/results/README.md
```

---

# Key Contributions Across Both Tasks

✔ Multi-spectral 12-band modeling  
✔ Robust geospatial band alignment  
✔ EfficientNet backbone adaptation  
✔ Strong class imbalance handling  
✔ Stable training configuration  
✔ Self-supervised representation learning (Task 2)  
✔ Modular and reproducible pipeline  

---

# Core Technologies

```
PyTorch
timm
rasterio
numpy
pandas
scikit-learn
scikit-image
```

---

# Reproducibility

1. Join the Kaggle competition
2. Download dataset
3. Organize data according to task-specific folder structure
4. Navigate to:
   - `Task1/` for supervised classification
   - `Task2/` for SSL + fine-tuning pipeline
5. Run the respective notebook to train and generate submission

---

# Results

Performance metrics and leaderboard scores are documented separately:

- `Task1/results/README.md`
- `Task2/results/README.md`

---

# Conclusion

This repository demonstrates:

- Effective multi-spectral satellite modeling
- Robust supervised learning under class imbalance
- Successful integration of self-supervised learning
- Competitive performance on the Kaggle leaderboard

Together, Task 1 and Task 2 showcase complementary approaches for crop disease diagnosis beyond the visible spectrum.

---

## 🔗 Competition Link

https://www.kaggle.com/competitions/beyond-visible-spectrum-ai-for-agriculture-2026# Beyond-Visible-Spectrum-AI-for-Agriculture
