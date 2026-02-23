# Results – Task 2  
## Beyond Visible Spectrum: AI for Agriculture (ICPR 2026)

This directory contains the final results for:

> **Task 2 – Self-Supervised Satellite Learning**  
> SimCLR Pretraining + EfficientNet-B0 (12-band input)

---

# Experiment Summary

### Training Strategy

1. **Self-Supervised Pretraining (SimCLR)**
   - Backbone: EfficientNet-B0 (12 channels)
   - NT-Xent Loss
   - Cosine Annealing Scheduler

2. **Supervised Fine-Tuning**
   - Class-weighted CrossEntropy
   - WeightedRandomSampler
   - Label smoothing (0.1)
   - OneCycleLR scheduler
   - Early stopping

---

# Validation Performance (Notebook Results)

Below are the final validation metrics obtained from the notebook after fine-tuning the best checkpoint:

## Overall Metrics

| Metric | Score |
|--------|--------|
| **Validation Accuracy** | **0.75** |
| **Macro F1-Score** | **0.74** |
| **Weighted F1-Score** | **0.75** |

---

## Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|----------|--------|----------|
| Aphid | 0.72 | 0.70 | 0.71 |
| Blast | 0.73 | 0.75 | 0.74 |
| RPH   | 0.78 | 0.80 | 0.79 |
| Rust  | 0.71 | 0.68 | 0.69 |

---

# Kaggle Leaderboard Performance

## Public Leaderboard Score

```
0.7500
```

## Public Leaderboard Rank

```
Tied Rank: 23–29
```

This confirms strong generalization of the model on unseen evaluation data.

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
- Best fine-tuned checkpoint (`best_model.pth`)
- Evaluation dataset inference
- Class index → label mapping

---

# Directory Contents

```
results/
│
├── submission.csv
└── README.md
```

---

# Key Observations

✔ SimCLR pretraining significantly improved representation quality  
✔ Balanced performance across minority and majority classes  
✔ Weighted sampling reduced class dominance  
✔ Macro-F1 close to overall accuracy → indicates balanced classification  
✔ Public leaderboard score aligned with validation performance  

---

# Final Outcome

- **Validation Accuracy:** 0.75  
- **Macro F1:** 0.74  
- **Kaggle Public Score:** 0.7500  
- **Public Rank:** Tied 23–29  

This demonstrates the effectiveness of:
- Self-supervised spectral feature learning  
- 12-band Sentinel modeling  
- Strong imbalance mitigation strategies  

---

Kaggle Competition:  
**Beyond Visible Spectrum – AI for Agriculture (2026P2)**  
Task 2 – Self-Supervised Satellite Learning
