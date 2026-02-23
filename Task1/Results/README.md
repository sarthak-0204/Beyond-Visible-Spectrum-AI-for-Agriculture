# Results – Task 1  
## Multimodal Crop Disease Diagnosis  
ICPR 2026 – Kaggle Competition

This directory contains the final results for:

> **Task 1 – Multimodal Crop Disease Classification**  
> EfficientNet-B0 (12-band input) – Fully Supervised Training

---

# Experiment Summary

### Model Configuration

- Backbone: EfficientNet-B0 (modified for 12-channel input)
- Custom classification head (3 output classes)
- Class-weighted CrossEntropy loss
- WeightedRandomSampler for imbalance handling
- OneCycleLR scheduler
- Early stopping
- Gradient clipping

Classes:

- Rust  
- Healthy  
- Other  

Training performed using 12-band Sentinel multi-spectral imagery.

---

# Validation Performance (Notebook Results)

Metrics computed using `sklearn.metrics.classification_report`.

## Overall Metrics

| Metric | Score |
|--------|--------|
| **Validation Accuracy** | 0.69 |
| **Macro F1-Score** | 0.68 |
| **Weighted F1-Score** | 0.69 |

---

## Per-Class Performance

| Class   | Precision | Recall | F1-Score |
|---------|----------|--------|----------|
| Rust    | 0.67 | 0.65 | 0.66 |
| Healthy | 0.72 | 0.74 | 0.73 |
| Other   | 0.64 | 0.61 | 0.62 |


---

# Kaggle Leaderboard Performance

## Public Leaderboard Score

```
0.69473
```

## Public Leaderboard Rank

```
Tied Rank: 100–113
```

The leaderboard score is consistent with the validation metrics, indicating stable generalization to unseen test samples.

---

# Submission Details

Submission file:

```
submission.csv
```

Format:

```
ID,Category
```

Generated using:

1. Best model checkpoint (`best_model.pth`)
2. Inference on evaluation dataset
3. Class index → label mapping
4. CSV export

---

# Directory Contents

```
results/
│
├── best_model.pth
├── submission.csv
└── README.md
```

---

# Observations

- Strongest performance observed on **Healthy** class
- Rust detection reasonably balanced
- “Other” class remains slightly more challenging
- Macro-F1 close to accuracy → indicates relatively balanced predictions

---

# Final Outcome

- **Validation Accuracy:** 0.69  
- **Macro F1:** 0.68  
- **Kaggle Public Score:** 0.69473  
- **Leaderboard Rank:** Tied 100–113  

This establishes a solid supervised multi-spectral baseline for Task 1.

---

Kaggle Competition:  
**Beyond Visible Spectrum – AI for Agriculture (2026)**  
Task 1 – Multimodal Crop Disease Diagnosis
