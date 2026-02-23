# Wheat Disease Multimodal Classification

## Overview

This repository implements a multimodal deep learning solution for wheat disease classification using UAV-acquired imagery.

The model combines:

- RGB imagery
- Multispectral (MS) data
- Hyperspectral (HS) data

The objective is to classify wheat patches into:

- Health
- Rust
- Other

Evaluation metric: **Macro F1 Score**

---

## Dataset Description

### Data Acquisition

- Platform: DJI M600 Pro UAV  
- Sensor: S185 Snapshot Hyperspectral  
- Spectral Range: 450–950 nm  
- Spectral Resolution: 4 nm  
- Spatial Resolution: ~4 cm/pixel  
- Acquisition Dates:
  - May 3, 2019 (Pre-grouting stage)
  - May 8, 2019 (Middle grouting stage)

---

## Modalities Used

### RGB
True-color images reconstructed from hyperspectral bands.

### Multispectral (5 Bands)
- Blue (~480nm)
- Green (~550nm)
- Red (~650nm)
- Red Edge (740nm)
- NIR (833nm)

Additional vegetation indices:
- NDVI
- NDRE

### Hyperspectral (125 Bands)
- Bands trimmed to remove sensor noise
- Final bands used: 10–110 (101 bands)
- Per-sample spectral normalization applied

---

## Model Architecture

### RGB Encoder
- ConvNeXt-Tiny (ImageNet pretrained)

### MS Encoder
- 2D CNN (7 channels: 5 MS + NDVI + NDRE)

### HS Encoder
- Spectral 1D convolution (per pixel)
- Spatial 2D CNN
- Spectral dropout regularization

### Fusion
- Attention-based multimodal fusion
- Fully connected classification head

---

## Training Strategy

- 5-Fold Stratified Cross Validation
- Optimizer: AdamW
- Learning Rate: 3e-4
- Scheduler: Cosine Annealing
- Label Smoothing: 0.1
- Mixed Precision Training (AMP)
- Gradient Clipping
- Early Stopping

---

## Cross-Validation Performance

| Fold | Macro F1 |
|------|----------|
| 0    | ~0.69 |
| 1    | ~0.58 |
| 2    | ~0.57 |
| 3    | ~0.68 |
| 4    | ~0.63 |

Mean CV: ~0.64

Public Leaderboard Score:
0.6947

---

## Inference

Final submission uses:

- 5-Fold model ensemble
- Logit averaging
- Deterministic inference

Submission format:

---

## Installation

```bash
pip install -r requirements.txt

```txt
torch
torchvision
timm
numpy
pandas
opencv-python
tifffile
scikit-learn
with open("results/cv_scores.txt","w") as f:
    for i,score in enumerate(best_scores):
        f.write(f"Fold {i}: {score:.4f}\n")
    f.write(f"\nMean CV: {np.mean(best_scores):.4f}")
