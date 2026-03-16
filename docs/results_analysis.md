# Results Analysis

**Date:** 2026-03-13
**Test set:** n = 274 images (patient-level stratified split)
**Metric:** AUC-ROC unless otherwise noted

All AUC values are sourced from `eval_test.json` files produced by `scripts/evaluate.py --split test`. Two sources are used:

- **`runs/`** — local run directories (evaluated locally or synced from HPC)
- **`results_bundle/`** — `~/Downloads/results_bundle/` containing HPC-produced eval files; treated as equally authoritative

> **Masks/evaluation mismatch:** `evaluate.py` does not read `masks_dir` from `config.json` — mask preprocessing is only applied when `--masks_dir` is passed explicitly. All `eval_test.json` files were produced **without** `--masks_dir`, so every run — including those trained with mask-based preprocessing — was evaluated on full uncropped images. The "masks trained" column in tables reflects the training configuration only.

> **Local copy corruption:** The file `runs/efficientnet_b0_masked_20260228_200621/eval_test.json` is corrupt — it is an identical copy of the anomalous `efficientnet_b0_20260228_200621` evaluation (AUC=0.627). The correct evaluation for that run is in `results_bundle/efficientnet_b0_masked_20260228_200621/eval_test.json` (AUC=0.912). All tables use the results_bundle value.

---

## 1. Best-per-Architecture Leaderboard

One row per architecture, best run by test AUC.

| Architecture | Best AUC | Accuracy | Sens @ 0.5 | Spec @ 0.5 | Youden thr | Sens @ J | Spec @ J | Run |
|---|---|---|---|---|---|---|---|---|
| dinov3_large | **0.935** | 0.872 | 0.795 | 0.909 | 0.126 | 0.780 | 0.871 | `dinov3_large_20260305_191824` |
| densenet121 | 0.928 | 0.880 | 0.818 | 0.909 | 0.140 | 0.945 | 0.773 | `densenet121_20260228_193830` |
| efficientnet_b0 | 0.912 | 0.832 | 0.739 | 0.876 | 0.087 | 0.868 | 0.820 | `efficientnet_b0_masked_20260228_200621` |
| dinov2_base | 0.899 | 0.847 | 0.750 | 0.892 | 0.137 | 0.736 | 0.871 | `dinov2_base_20260305_171950` |
| dinov2_large | 0.898 | 0.752 | 0.920 | 0.672 | 0.989 | 0.747 | 0.851 | `dinov2_large_20260311_223349` |
| dinov3_base | 0.898 | 0.858 | 0.761 | 0.903 | 0.310 | 0.758 | 0.876 | `dinov3_base_20260305_183935` |
| resnet18 | 0.699 | 0.478 | 0.932 | 0.263 | — | — | — | `resnet18_20260228_173821` |

Notes:
- Youden threshold and sens/spec @ J are derived from val-split and applied to test. Runs without threshold data (older eval format) show —.
- DINOv2-Large Youden threshold of 0.989 is anomalous — the model's softmax probabilities are heavily compressed near 1.0, indicating calibration problems despite reasonable AUC.
- ResNet-18 (frozen) near-random specificity: classifies almost all samples as malignant.

---

## 2. Frozen vs Unfrozen Backbone

Paired comparisons where both conditions have verified `eval_test.json`. All runs evaluated on full (unmasked) images regardless of training configuration.

| Architecture | Config | Frozen AUC | Unfrozen AUC | Δ |
|---|---|---|---|---|
| densenet121 | no masks, lr=1e-4/1e-3, wd=1e-4 | 0.777 | 0.904 | **+0.127** |
| dinov2_base | masks-trained, lr=1e-5, wd=0.01, bs=32 | 0.839 | 0.899 | **+0.060** |
| dinov3_base | masks-trained, lr=1e-5, wd=0.01, bs=32 | 0.793 | 0.898 | **+0.105** |
| dinov3_large | no masks, lr=1e-5, wd=0.01, bs=8 | 0.797 | 0.907 | **+0.110** |

Note on densenet121: the frozen run used lr=1e-3 while the unfrozen used lr=1e-4 (different hyperparameters). For dinov2/dinov3, both frozen and unfrozen runs were trained with masks but evaluated without.

**Takeaway:** Unfreezing the backbone is consistently beneficial (+0.06–0.13 AUC) across all four architecture families tested. Frozen backbone is not competitive for final results.

---

## 3. Masked vs Unmasked Training

Comparisons holding hyperparameters constant. All evaluations are without masks (see mismatch note above).

| Architecture | Config | Unmasked AUC | Masks-trained AUC | Δ |
|---|---|---|---|---|
| densenet121 (unfrozen) | lr=1e-4, wd=1e-4, bs=32, mlp | 0.904 | **0.928** | **+0.024** |
| efficientnet_b0 (unfrozen) | lr=1e-4, wd=1e-4, bs=32, mlp | 0.881 | **0.912** | **+0.031** |
| dinov3_large (unfrozen, bs=32) | lr=1e-5, wd=0.01 — head differs† | 0.830 | **0.935** | **+0.105** |
| dinov3_large (unfrozen, bs=8) | lr=1e-5, wd=0.01, bs=8, mlp | 0.907 | 0.648 ⚠️ | **−0.259** |

†The dinov3_large bs=32 no-masks run (`dinov3_large_20260312_004437`) used MLP head + 0.5 dropout, while the masked run (`dinov3_large_20260305_191824`) used linear head + 0.3 dropout. Not a fully controlled comparison; the +0.105 delta should be interpreted with that caveat.

The bs=8 pair is fully controlled (same head, dropout, lr, wd) and shows catastrophic failure with masks — training instability likely caused by aggressive mask-induced distribution shift at very small batch sizes during ViT fine-tuning.

**Takeaway:** Mask-based preprocessing helps CNNs modestly (+0.02–0.03) and can substantially boost ViT performance at bs=32 (+0.105 for DINOv3-Large). Avoid masks at bs=8 for large ViT fine-tuning.

---

## 4. Architecture Family Comparison

Full metrics for every verified run, grouped by architecture.

| Run | AUC | Acc | Sens@0.5 | Spec@0.5 | Frozen | Masks | Notes |
|---|---|---|---|---|---|---|---|
| `resnet18_20260228_173821` | 0.699 | 0.478 | 0.932 | 0.263 | ✓ | ✗ | Degenerate: near-all malignant |
| `densenet121_20260228_201457` | 0.777 | 0.573 | 0.898 | 0.419 | ✓ | ✗ | |
| `densenet121_20260228_184613` | 0.904 | 0.825 | 0.727 | 0.871 | ✗ | ✗ | |
| `densenet121_20260228_193830` | **0.928** | 0.880 | 0.818 | 0.909 | ✗ | ✓ | Best CNN |
| `efficientnet_b0_20260228_175341` | 0.850 | 0.770 | 0.670 | 0.817 | ✗ | ✗ | |
| `efficientnet_b0_20260228_181632` | 0.881 | 0.810 | 0.727 | 0.849 | ✗ | ✗ | |
| `efficientnet_b0_lr5e4_wd1e5_bs16_mlp_es` | 0.898 | 0.821 | 0.795 | 0.833 | ✗ | ✗ | Early-stop, ep28 |
| `efficientnet_b0_masked_20260228_200621` | **0.912** | 0.832 | 0.739 | 0.876 | ✗ | ✓ | |
| `efficientnet_b0_20260228_200621` | 0.627 ⚠️ | 0.369 | 0.989 | 0.075 | ✗ | ✓ | Anomalous |
| `dinov2_base_masked_20260304_180318` | 0.839 | 0.642 | 0.932 | 0.505 | ✓ | ✓ | |
| `dinov2_base_20260305_171950` | **0.899** | 0.847 | 0.750 | 0.892 | ✗ | ✓ | |
| `dinov2_large_20260311_223349` | **0.898** | 0.752 | 0.920 | 0.672 | ✗ | ✗ | Calibration issue |
| `dinov3_base_masked_(no_eval)20260304_205613` | 0.793 | 0.489 | 0.909 | 0.290 | ✓ | ✓ | |
| `dinov3_base_20260305_183935` | **0.898** | 0.858 | 0.761 | 0.903 | ✗ | ✓ | |
| `dinov3_large_lr5e6_wd1e2_bs8_mlp_ft` | 0.797 | 0.467 | 0.989 | 0.220 | ✓ | ✗ | bs=8 |
| `dinov3_large_20260312_004437` | 0.830 | 0.573 | 0.920 | 0.409 | ✗ | ✗ | bs=32, mlp |
| `dinov3_large_lr1e5_wd1e2_bs8_mlp_ft` | 0.907 | 0.839 | 0.648 | 0.930 | ✗ | ✗ | bs=8, ep24 |
| `dinov3_large_20260305_191824` | **0.935** | 0.872 | 0.795 | 0.909 | ✗ | ✓ | bs=32, linear |
| `dinov3_large_lr1e5_wd1e2_bs8_mlp_ft_mask` | 0.648 ⚠️ | 0.336 | 1.000 | 0.022 | ✗ | ✓ | Anomalous |

**Key observations:**
- DINOv3-Large + masks (bs=32) is the clear winner at 0.935.
- DenseNet-121 + masks is the best CNN at 0.928, with the highest Youden-J sensitivity (0.945).
- EfficientNet-B0, DINOv2-Base, DINOv2-Large, DINOv3-Base cluster between 0.898–0.912 — essentially tied given n=274.
- DINOv2-Large Youden threshold of 0.989 is a calibration red flag (probabilities compressed near 1.0).
- dinov3_large bs=8 achieves a strong 0.907 without masks but fails catastrophically with masks (0.648 — degenerate: all samples predicted malignant).

---

## 5. Clinical Threshold Analysis

For the four top-performing runs. Thresholds derived from validation split, applied to test split.

### dinov3_large_20260305_191824 — AUC 0.935

| Criterion | Threshold | Sensitivity | Specificity |
|---|---|---|---|
| Default (0.5) | 0.500 | 0.795 | 0.909 |
| Youden J | 0.126 | 0.780 | 0.871 |
| Sensitivity ≥ 0.95 | 0.005 | 0.956 | 0.474 |
| Specificity ≥ 0.90 | 0.485 | 0.659 | 0.912 |

### densenet121_20260228_193830 — AUC 0.928

| Criterion | Threshold | Sensitivity | Specificity |
|---|---|---|---|
| Default (0.5) | 0.500 | 0.818 | 0.909 |
| Youden J | 0.140 | 0.945 | 0.773 |
| Sensitivity ≥ 0.95 | 0.095 | 0.956 | 0.742 |
| Specificity ≥ 0.90 | 0.675 | 0.769 | 0.902 |

### efficientnet_b0_masked_20260228_200621 — AUC 0.912

| Criterion | Threshold | Sensitivity | Specificity |
|---|---|---|---|
| Default (0.5) | 0.500 | 0.739 | 0.876 |
| Youden J | 0.087 | 0.868 | 0.820 |
| Sensitivity ≥ 0.95 | — | — | — |
| Specificity ≥ 0.90 | 0.915 | 0.681 | 0.912 |

Note: this run's threshold sweep did not reach ≥0.95 sensitivity within the grid (threshold=0.0 yields 1.0/0.0, next step jumps past target).

### dinov3_large_lr1e5_wd1e2_bs8_mlp_ft — AUC 0.907

| Criterion | Threshold | Sensitivity | Specificity |
|---|---|---|---|
| Default (0.5) | 0.500 | 0.648 | 0.930 |
| Youden J | 0.002 | 0.879 | 0.845 |
| Sensitivity ≥ 0.95 | — | — | — |
| Specificity ≥ 0.90 | 0.060 | 0.747 | 0.902 |

Note: ≥0.95 sensitivity not achievable within the grid (threshold=0.0 gives 1.0/0.0, no intermediate point ≥0.95).

**Clinical interpretation:**
- **Screening (minimise false negatives):** DenseNet-121 at threshold 0.095 achieves sensitivity 0.956 / specificity 0.742 — strongest screening operating point among verified runs.
- **Confirmatory (minimise false positives):** DINOv3-Large (bs=32) at threshold 0.485 achieves sensitivity 0.659 / specificity 0.912.
- **Balanced:** DenseNet-121 Youden (0.140) delivers 0.945 sensitivity / 0.773 specificity — highest single-point sensitivity across all models.

---

## 6. All 19 Evaluated Runs — Full Inventory

| Run directory | Source | Model | Frozen | Masks | LR | WD | BS | Head | Dropout | ES | Test AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `resnet18_20260228_173821` | local+bundle | resnet18 | ✓ | ✗ | 1e-3 | 1e-5 | 32 | mlp | — | ✗ | 0.699 |
| `densenet121_20260228_184613` | local+bundle | densenet121 | ✗ | ✗ | 1e-4 | 1e-4 | 32 | mlp | 0.5 | ✗ | 0.904 |
| `densenet121_20260228_193830` | local+bundle | densenet121 | ✗ | ✓ | 1e-4 | 1e-4 | 32 | mlp | 0.5 | ✗ | **0.928** |
| `densenet121_20260228_201457` | local+bundle | densenet121 | ✓ | ✗ | 1e-3 | 1e-4 | 32 | mlp | 0.3 | ✗ | 0.777 |
| `efficientnet_b0_20260228_175341` | local+bundle | efficientnet_b0 | ✗ | ✗ | 1e-4 | 1e-5 | 32 | mlp | — | ✗ | 0.850 |
| `efficientnet_b0_20260228_181632` | local+bundle | efficientnet_b0 | ✗ | ✗ | 1e-4 | 1e-4 | 32 | mlp | 0.5 | ✗ | 0.881 |
| `efficientnet_b0_lr5e4_wd1e5_bs16_mlp_es` | bundle only | efficientnet_b0 | ✗ | ✗ | 5e-4 | 1e-5 | 16 | mlp | 0.4 | ep28 | 0.898 |
| `efficientnet_b0_masked_20260228_200621` | bundle only† | efficientnet_b0 | ✗ | ✓ | 1e-4 | 1e-4 | 32 | mlp | 0.5 | ✗ | **0.912** |
| `efficientnet_b0_20260228_200621` | local+bundle | efficientnet_b0 | ✗ | ✓ | 1e-4 | 1e-4 | 32 | mlp | 0.5 | ✗ | 0.627 ⚠️ |
| `dinov2_base_masked_20260304_180318` | local+bundle | dinov2_base | ✓ | ✓ | 1e-5 | 0.01 | 32 | mlp | 0.5 | ✗ | 0.839 |
| `dinov2_base_20260305_171950` | bundle only | dinov2_base | ✗ | ✓ | 1e-5 | 0.01 | 32 | mlp | 0.5 | ✗ | **0.899** |
| `dinov2_large_20260311_223349` | bundle only | dinov2_large | ✗ | ✗ | 1e-5 | 0.01 | 32 | mlp | 0.5 | ✗ | 0.898 |
| `dinov3_base_masked_(no_eval)20260304_205613` | bundle only | dinov3_base | ✓ | ✓ | 1e-5 | 0.01 | 32 | linear | 0.3 | ✗ | 0.793 |
| `dinov3_base_20260305_183935` | bundle only | dinov3_base | ✗ | ✓ | 1e-5 | 0.01 | 32 | linear | 0.3 | ✗ | **0.898** |
| `dinov3_large_lr5e6_wd1e2_bs8_mlp_ft` | bundle only | dinov3_large | ✓ | ✗ | 5e-6 | 0.01 | 8 | mlp | 0.5 | ep87 | 0.797 |
| `dinov3_large_20260312_004437` | bundle only | dinov3_large | ✗ | ✗ | 1e-5 | 0.01 | 32 | mlp | 0.5 | ✗ | 0.830 |
| `dinov3_large_lr1e5_wd1e2_bs8_mlp_ft` | bundle only | dinov3_large | ✗ | ✗ | 1e-5 | 0.01 | 8 | mlp | 0.5 | ep24 | 0.907 |
| `dinov3_large_20260305_191824` | bundle only | dinov3_large | ✗ | ✓ | 1e-5 | 0.01 | 32 | linear | 0.3 | ✗ | **0.935** |
| `dinov3_large_lr1e5_wd1e2_bs8_mlp_ft_mask` | bundle only | dinov3_large | ✗ | ✓ | 1e-5 | 0.01 | 8 | mlp | 0.5 | ep20 | 0.648 ⚠️ |

†The local `runs/efficientnet_b0_masked_20260228_200621/eval_test.json` is a corrupted copy of the anomalous 0.627 run. Use `results_bundle/` as the authoritative source for this run.

---

## 7. Runs Awaiting Evaluation

The following runs have checkpoints locally but no verified `eval_test.json` in either source:

| Run directory | Model | Location | Notes |
|---|---|---|---|
| `efficientnet_b0_20260228_175041` | efficientnet_b0 | local | Early run |
| `efficientnet_b0_20260228_180955` | efficientnet_b0 | local | Intermediate run |
| `efficientnet_b0_mlp_dropout0.4_patience15` | efficientnet_b0 | local | Early-stop, no masks |
| `dinov2_base_20260228_202435` | dinov2_base | local | No masks; needs HF token, slow on MPS |
| `clip_vit_base_masked_(no_eval)20260304_200843` | clip_vit_base | local | Grid search best: val AUC 0.773 (frozen) |
| `dinov3_large_masked_(no_eval)20260304_212948` | dinov3_large | local | Frozen, masks-trained; needs HF token |

```bash
# EfficientNet runs — no HF token needed
uv run python scripts/evaluate.py --run_dir runs/efficientnet_b0_20260228_175041 --split test --threshold_split val
uv run python scripts/evaluate.py --run_dir runs/efficientnet_b0_20260228_180955 --split test --threshold_split val
uv run python scripts/evaluate.py --run_dir runs/efficientnet_b0_mlp_dropout0.4_patience15 --split test --threshold_split val
uv run python scripts/evaluate.py --run_dir "runs/clip_vit_base_masked_(no_eval)20260304_200843" --split test --threshold_split val

# DINOv2/DINOv3 — requires HF token
HF_TOKEN=<token> uv run python scripts/evaluate.py --run_dir runs/dinov2_base_20260228_202435 --split test --threshold_split val
HF_TOKEN=<token> uv run python scripts/evaluate.py --run_dir "runs/dinov3_large_masked_(no_eval)20260304_212948" --split test --threshold_split val
```

---

## 8. Summary and Recommendations

### Best model overall
**DINOv3-Large + lesion masks, unfrozen, bs=32, lr=1e-5, wd=0.01, linear head** (`dinov3_large_20260305_191824`):
Test AUC **0.935**, accuracy 87.2%, default operating point sens/spec = 0.795/0.909.

### Best CNN (no HuggingFace dependency)
**DenseNet-121 + lesion masks, unfrozen, lr=1e-4, wd=1e-4, bs=32** (`densenet121_20260228_193830`):
Test AUC **0.928**, Youden-J sensitivity 0.945 — highest among all models.

### Design conclusions

1. **Always fine-tune the full backbone** — frozen backbone underperforms by 0.06–0.13 AUC consistently.
2. **Lesion masks help at bs=32** — +0.02–0.03 for CNNs, +0.10 for DINOv3-Large.
3. **Avoid masks + bs=8** — combination caused catastrophic failure for DINOv3-Large (0.648 vs 0.907).
4. **DINOv3-Large > DenseNet-121 ≈ EfficientNet-B0 ≈ DINOv2-Base ≈ DINOv3-Base** — the large ViT backbone leads when fine-tuned fully with masks.
5. **DINOv2-Large calibration issue** — Youden threshold 0.989 indicates extreme probability compression; not reliable for deployment without recalibration.
