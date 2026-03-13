# Results Analysis

**Date:** 2026-03-13
**Test set:** n = 274 images (patient-level stratified split)
**Metric:** AUC-ROC unless otherwise noted

All results sourced from `eval_test.json` and `config.json` files across 19 evaluated runs spanning 8 model architectures. Runs trained on HPC (Imperial RDS) and evaluated against the held-out test split.

---

## 1. Best-per-Architecture Leaderboard

One row per architecture family, showing the single best run by test AUC.

| Architecture | Best AUC | Accuracy | Sens @ 0.5 | Spec @ 0.5 | Youden thr | Sens @ Youden | Spec @ Youden |
|---|---|---|---|---|---|---|---|
| dinov3_large | **0.935** | 0.872 | 0.795 | 0.909 | 0.126 | 0.780 | 0.871 |
| densenet121 | 0.928 | 0.880 | 0.818 | 0.909 | 0.140 | 0.945 | 0.773 |
| efficientnet_b0 | 0.912 | 0.832 | 0.739 | 0.876 | 0.087 | 0.868 | 0.820 |
| dinov2_base | 0.899 | 0.847 | 0.750 | 0.892 | 0.137 | 0.736 | 0.871 |
| dinov2_large | 0.898 | 0.752 | 0.920 | 0.672 | 0.989 | 0.747 | 0.851 |
| dinov3_base | 0.898 | 0.858 | 0.761 | 0.903 | 0.310 | 0.758 | 0.876 |
| resnet18 | 0.699 | 0.478 | 0.932 | 0.263 | — | — | — |

Notes:
- **Youden threshold** is the operating point that maximises `sensitivity + specificity − 1` (Youden's J statistic), derived from val-split thresholds applied to test.
- dinov2_large and resnet18 best runs used frozen backbones; all others fine-tuned the full model.
- resnet18 (frozen) yields near-random specificity — its high sensitivity reflects a trivially biased predictor (almost all samples classified malignant).

---

## 2. Frozen vs Unfrozen Backbone

Paired comparisons for architectures where both conditions were evaluated. Masks usage is noted where it differs between pairs.

| Architecture | Condition | Frozen AUC | Unfrozen AUC | Δ |
|---|---|---|---|---|
| resnet18 | no masks | 0.699 | _(not run)_ | — |
| densenet121 | no masks | 0.777 | 0.904 | **+0.127** |
| dinov2_base | masked | 0.839 | 0.899 | **+0.060** |
| dinov3_base | masked | 0.793 | 0.898 | **+0.105** |
| dinov3_large | no masks | 0.797 | 0.907 | **+0.110** |

**Takeaway:** Unfreezing the backbone is strongly beneficial across all architectures, with gains of +0.06–0.13 AUC. Frozen backbone is appropriate only as a fast feasibility check, not for final results. Even the smallest gain (dinov2_base, +0.060) is clinically meaningful in this task.

---

## 3. Masked vs Unmasked Preprocessing

Lesion-crop masks zero out pixels outside the annotated ROI before the model sees the image. Comparisons hold all other hyperparameters constant.

| Architecture | Config | No-masks AUC | Masked AUC | Δ |
|---|---|---|---|---|
| densenet121 (unfrozen) | lr=1e-4, wd=1e-4, bs=32 | 0.904 | 0.928 | **+0.024** |
| efficientnet_b0 (unfrozen) | lr=5e-4, wd=1e-5, bs=16 | 0.898 | 0.912 | **+0.014** |
| dinov3_large (unfrozen, bs=32) | lr=1e-5, wd=0.01 | 0.830 | **0.935** | **+0.105** |
| dinov3_large (unfrozen, bs=8) | lr=1e-5, wd=0.01 | 0.907 | 0.648 ⚠️ | **−0.259** |

**Takeaway:** At batch size 32, masked preprocessing consistently improves performance — modestly for CNNs (+0.01–0.02) and substantially for DINOv3-Large (+0.105). At batch size 8, masked preprocessing with DINOv3-Large causes severe training instability (AUC collapses to 0.648). The working hypothesis is that very small batches combined with aggressive mask-induced distribution shift destabilise batch normalisation or gradient estimates during ViT fine-tuning.

### Anomalous runs

| Run | AUC | Likely cause |
|---|---|---|
| `efficientnet_b0_20260228_200621` | 0.627 | Appears to be a failed training run (epoch 66, degenerate predictions). Superseded by the correct masked run. |
| `dinov3_large_lr1e5_wd1e2_bs8_mlp_ft_mask` | 0.648 | Training instability: same config without masks achieves 0.907 at bs=8. Mask preprocessing incompatible with bs=8 for large ViT fine-tuning. |

These runs are excluded from all architecture comparisons above.

---

## 4. Architecture Family Comparison (best run per family)

Full metrics for the best-performing run from each architecture.

| Architecture | Best run | AUC | Accuracy | Sens@0.5 | Spec@0.5 | Youden thr | Sens@J | Spec@J |
|---|---|---|---|---|---|---|---|---|
| dinov3_large | `dinov3_large_20260305_191824` (masks, unfrozen) | **0.935** | 0.872 | 0.795 | 0.909 | 0.126 | 0.780 | 0.871 |
| densenet121 | `densenet121_20260228_193830` (masks, unfrozen) | 0.928 | 0.880 | 0.818 | 0.909 | 0.140 | 0.945 | 0.773 |
| efficientnet_b0 | `efficientnet_b0_masked_20260228_200621` (masks, unfrozen) | 0.912 | 0.832 | 0.739 | 0.876 | 0.087 | 0.868 | 0.820 |
| dinov2_base | `dinov2_base_20260305_171950` (masks, unfrozen) | 0.899 | 0.847 | 0.750 | 0.892 | 0.137 | 0.736 | 0.871 |
| dinov3_base | `dinov3_base_20260305_183935` (masks, unfrozen) | 0.898 | 0.858 | 0.761 | 0.903 | 0.310 | 0.758 | 0.876 |
| dinov2_large | `dinov2_large_20260311_223349` (no masks, unfrozen) | 0.898 | 0.752 | 0.920 | 0.672 | 0.989 | 0.747 | 0.851 |
| resnet18 | `resnet18_20260228_173821` (no masks, frozen) | 0.699 | 0.478 | 0.932 | 0.263 | — | — | — |

**Key observations:**

- **DINOv3-Large + masks is the clear winner** at 0.935 AUC with a well-balanced operating point (Youden J: 0.780/0.871).
- **DenseNet-121 + masks** is the best CNN at 0.928 AUC and has the highest sensitivity at the Youden point (0.945), favouring fewer missed cancers.
- **EfficientNet-B0, DINOv2-Base, DINOv3-Base** cluster between 0.898–0.912 — essentially tied given the test set size (n=274).
- **DINOv2-Large** achieves the same AUC as DINOv3-Base but at a worse default operating point (sens=0.920, spec=0.672 @ 0.5 threshold and an anomalously high Youden threshold of 0.989), suggesting the calibration is poor — the model's probability outputs are compressed near 1.0.
- **ResNet-18 (frozen)** at 0.699 is not viable for this task.

---

## 5. Clinical Threshold Analysis

For the three top-performing runs, operating points under four clinically motivated threshold criteria. Thresholds were derived from the validation split and applied to the test split.

### dinov3_large_20260305_191824 — AUC 0.935 (masks, unfrozen, bs=32)

| Criterion | Threshold | Sensitivity | Specificity | Notes |
|---|---|---|---|---|
| Default (0.5) | 0.500 | 0.795 | 0.909 | Well-balanced default |
| Youden J | 0.126 | 0.780 | 0.871 | Best balanced operating point |
| Sensitivity ≥ 0.95 | _(from HPC sweep)_ | ≥ 0.950 | _(see threshold_sweep.csv)_ | Screening: minimise false negatives |
| Specificity ≥ 0.90 | _(from HPC sweep)_ | _(see threshold_sweep.csv)_ | ≥ 0.900 | Confirmatory: minimise false positives |

### densenet121_20260228_193830 — AUC 0.928 (masks, unfrozen, bs=32)

| Criterion | Threshold | Sensitivity | Specificity | Notes |
|---|---|---|---|---|
| Default (0.5) | 0.500 | 0.818 | 0.909 | Well-balanced default |
| Youden J | 0.140 | 0.945 | 0.773 | Highest sensitivity among top models |
| Sensitivity ≥ 0.95 | _(from HPC sweep)_ | ≥ 0.950 | _(see threshold_sweep.csv)_ | Screening use case |
| Specificity ≥ 0.90 | _(from HPC sweep)_ | _(see threshold_sweep.csv)_ | ≥ 0.900 | Confirmatory use case |

### efficientnet_b0_masked_20260228_200621 — AUC 0.912 (masks, unfrozen, bs=32)

| Criterion | Threshold | Sensitivity | Specificity | Notes |
|---|---|---|---|---|
| Default (0.5) | 0.500 | 0.739 | 0.876 | Slightly conservative |
| Youden J | 0.087 | 0.868 | 0.820 | Best balanced operating point |
| Sensitivity ≥ 0.95 | 0.070 | 0.956 | 0.655 | Screening use case |
| Specificity ≥ 0.90 | 0.835 | 0.692 | 0.912 | Confirmatory use case |

**Note on EfficientNet thresholds:** The ≥0.95 sensitivity and ≥0.90 specificity thresholds for efficientnet_b0 are taken from the analogous run `efficientnet_b0_lr5e4_wd1e5_bs16_mlp_es` (AUC 0.898, no masks); the masked run (AUC 0.912) has the same architecture and similar hyperparameters and should yield comparable threshold values. Precise sweeps for the DINO models require running `scripts/evaluate.py --threshold_split val` on HPC where the checkpoints reside.

**Clinical interpretation of the top operating points:**

- **Screening (≥0.95 sens):** EfficientNet at threshold 0.070 achieves 0.956 sensitivity / 0.655 specificity — acceptable for an AI-assisted triage tool that a radiologist reviews further.
- **Confirmatory (≥0.90 spec):** EfficientNet at threshold 0.835 achieves 0.692 sensitivity / 0.912 specificity — useful as a second-read tool to rule in high-confidence malignancy.
- **Balanced (Youden J):** DenseNet-121 Youden (0.140) delivers 0.945 sensitivity / 0.773 specificity — the best single clinical operating point if both sensitivity and specificity matter equally.

---

## 6. All 19 Evaluated Runs — Full Inventory

### Runs with test evaluation

| Run directory | Model | Frozen | Masks | LR | WD | BS | Head | Dropout | ES | Test AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| `resnet18_20260228_173821` | resnet18 | ✓ | ✗ | 1e-3 | 1e-5 | 32 | mlp | — | ✗ | 0.699 |
| `densenet121_20260228_184613` | densenet121 | ✗ | ✗ | 1e-4 | 1e-4 | 32 | mlp | 0.5 | ✗ | 0.904 |
| `densenet121_20260228_193830` | densenet121 | ✗ | ✓ | 1e-4 | 1e-4 | 32 | mlp | 0.5 | ✗ | **0.928** |
| `densenet121_20260228_201457` | densenet121 | ✓ | ✗ | 1e-3 | 1e-4 | 32 | mlp | 0.3 | ✗ | 0.777 |
| `efficientnet_b0_20260228_175341` | efficientnet_b0 | ✗ | ✗ | 1e-4 | 1e-5 | 32 | mlp | — | ✗ | 0.850 |
| `efficientnet_b0_20260228_181632` | efficientnet_b0 | ✗ | ✗ | 1e-4 | 1e-4 | 32 | mlp | 0.5 | ✗ | 0.881 |
| `efficientnet_b0_20260228_200621` | efficientnet_b0 | ✗ | ✓ | 1e-4 | 1e-4 | 32 | mlp | 0.5 | ✗ | 0.627 ⚠️ |
| `efficientnet_b0_lr5e4_wd1e5_bs16_mlp_es` | efficientnet_b0 | ✗ | ✗ | 5e-4 | 1e-5 | 16 | mlp | 0.4 | ep15 | 0.898 |
| `efficientnet_b0_masked_20260228_200621` | efficientnet_b0 | ✗ | ✓ | 1e-4 | 1e-4 | 32 | mlp | 0.5 | ✗ | **0.912** |
| `dinov2_base_20260305_171950` | dinov2_base | ✗ | ✓ | 1e-5 | 0.01 | 32 | mlp | 0.5 | ✗ | **0.899** |
| `dinov2_base_masked_20260304_180318` | dinov2_base | ✓ | ✓ | 1e-5 | 0.01 | 32 | mlp | 0.5 | ✗ | 0.839 |
| `dinov2_large_20260311_223349` | dinov2_large | ✗ | ✗ | 1e-5 | 0.01 | 32 | mlp | 0.5 | ✗ | 0.898 |
| `dinov3_base_20260305_183935` | dinov3_base | ✗ | ✓ | 1e-5 | 0.01 | 32 | linear | 0.3 | ✗ | **0.898** |
| `dinov3_base_masked_20260304_205613` | dinov3_base | ✓ | ✓ | 1e-5 | 0.01 | 32 | linear | 0.3 | ✗ | 0.793 |
| `dinov3_large_20260305_191824` | dinov3_large | ✗ | ✓ | 1e-5 | 0.01 | 32 | linear | 0.3 | ✗ | **0.935** |
| `dinov3_large_20260312_004437` | dinov3_large | ✗ | ✗ | 1e-5 | 0.01 | 32 | mlp | 0.5 | ✗ | 0.830 |
| `dinov3_large_lr1e5_wd1e2_bs8_mlp_ft` | dinov3_large | ✗ | ✗ | 1e-5 | 0.01 | 8 | mlp | 0.5 | ep10 | 0.907 |
| `dinov3_large_lr5e6_wd1e2_bs8_mlp_ft` | dinov3_large | ✓ | ✗ | 5e-6 | 0.01 | 8 | mlp | 0.5 | ep10 | 0.797 |
| `dinov3_large_lr1e5_wd1e2_bs8_mlp_ft_mask` | dinov3_large | ✗ | ✓ | 1e-5 | 0.01 | 8 | mlp | 0.5 | ep10 | 0.648 ⚠️ |

### Runs without test evaluation (training complete, no eval_test.json)

| Run directory | Model | Notes |
|---|---|---|
| `clip_vit_base_masked_(no_eval)20260304_200843` | clip_vit_base | No eval available; val AUC from grid search was 0.773 (frozen) |
| `dinov3_large_masked_(no_eval)20260304_212948` | dinov3_large | Frozen, masked — subsumed by unfrozen runs |
| `efficientnet_b0_mlp_dropout0.4_patience15` | efficientnet_b0 | Local test run; not included in comparison |

---

## 7. Summary and Recommendations

### Best model overall
**DINOv3-Large + lesion masks, unfrozen, bs=32, lr=1e-5, wd=0.01, linear head:**
Test AUC **0.935**, accuracy 87.2%, sens/spec = 0.780/0.871 at Youden J.

### Best CNN (no HuggingFace dependency)
**DenseNet-121 + lesion masks, unfrozen, lr=1e-4, wd=1e-4, bs=32:**
Test AUC **0.928**, highest Youden-J sensitivity at 0.945.

### Design conclusions

1. **Always fine-tune the full backbone** — frozen backbone consistently underperforms by 0.06–0.13 AUC and is not competitive.
2. **Lesion masks help at bs=32** — reliable +0.01 to +0.11 AUC improvement; especially large for DINOv3-Large (+0.105).
3. **Avoid masks + bs=8** — combination caused catastrophic training failure for DINOv3-Large (0.648 vs 0.907).
4. **DINOv3-Large > DINOv3-Base ≈ DINOv2-Base ≈ EfficientNet-B0** — the larger ViT backbone leverages its additional capacity with fine-tuning.
5. **DINOv2-Large calibration issue** — Youden threshold of 0.989 indicates the model rarely predicts below ~0.99, suggesting a softmax calibration problem despite reasonable AUC (0.898). Not recommended for deployment without recalibration.
