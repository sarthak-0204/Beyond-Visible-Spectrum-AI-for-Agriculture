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
