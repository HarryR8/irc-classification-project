#!/usr/bin/env bash
# run_all_evals.sh — evaluate all good runs in the comparison matrix on the test split.
#
# Conditions evaluated here
#   A  freeze=True  (frozen backbone)
#   B  freeze=False, no masks
#   C  freeze=False, with lesion-crop masks
#
# Usage
#   bash scripts/run_all_evals.sh
#   bash scripts/run_all_evals.sh --images_dir /path/to/data/raw --masks_dir /path/to/data/masks
#
# After each new training batch (Priority 2, 3), fill in the TODO sections below
# and re-run this script to bring the test results up to date.

set -euo pipefail

IMAGES_DIR="data/raw"
MASKS_DIR="data/masks"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --images_dir) IMAGES_DIR="$2"; shift 2 ;;
        --masks_dir)  MASKS_DIR="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

EVAL="uv run python scripts/evaluate.py"

echo "=========================================================="
echo "  run_all_evals.sh — BUS-BRA test-set evaluation"
echo "  images_dir = $IMAGES_DIR"
echo "  masks_dir  = $MASKS_DIR"
echo "=========================================================="

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Runs from Priority 1 (good val AUC, need / have test eval)
#            Condition C runs are evaluated with masks to match training.
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "--- Phase 1: existing runs (Priority 1) ---"

# densenet121  Cond B  val_AUC=0.9166  (no masks during training)
echo ""
echo ">>> densenet121 — Cond B  [densenet121_20260228_184613]"
$EVAL --run_dir runs/densenet121_20260228_184613 \
      --split test --images_dir "$IMAGES_DIR"

# densenet121  Cond C  val_AUC=0.9283  (trained with masks)
echo ""
echo ">>> densenet121 — Cond C  [densenet121_20260228_193830]"
$EVAL --run_dir runs/densenet121_20260228_193830 \
      --split test --images_dir "$IMAGES_DIR" --masks_dir "$MASKS_DIR"

# efficientnet_b0  Cond B  val_AUC=0.8625  (no masks)
echo ""
echo ">>> efficientnet_b0 — Cond B  [efficientnet_b0_20260228_181632]"
$EVAL --run_dir runs/efficientnet_b0_20260228_181632 \
      --split test --images_dir "$IMAGES_DIR"

# efficientnet_b0  Cond C  val_AUC=0.9054  (trained with masks)
echo ""
echo ">>> efficientnet_b0 — Cond C  [efficientnet_b0_20260228_200621]"
$EVAL --run_dir runs/efficientnet_b0_20260228_200621 \
      --split test --images_dir "$IMAGES_DIR" --masks_dir "$MASKS_DIR"

# dinov2_base  Cond A  val_AUC=0.8620  (frozen backbone, trained WITH masks)
echo ""
echo ">>> dinov2_base — Cond A  [dinov2_base_masked_20260304_180318]"
$EVAL --run_dir runs/dinov2_base_masked_20260304_180318 \
      --split test --images_dir "$IMAGES_DIR" --masks_dir "$MASKS_DIR"

# dinov2_base  Cond C  val_AUC=0.8878  (fine-tuned with masks)
# NOTE: config.json incorrectly records head_type=mlp; checkpoint is a linear head.
#       Must override with --head_type linear to load state dict correctly.
echo ""
echo ">>> dinov2_base — Cond C  [dinov2_base_20260305_171950]"
$EVAL --run_dir runs/dinov2_base_20260305_171950 \
      --split test --images_dir "$IMAGES_DIR" --masks_dir "$MASKS_DIR" \
      --head_type linear

# dinov3_base  Cond C  val_AUC=0.8984  (fine-tuned with masks, linear head)
echo ""
echo ">>> dinov3_base — Cond C  [dinov3_base_20260305_183935]"
$EVAL --run_dir runs/dinov3_base_20260305_183935 \
      --split test --images_dir "$IMAGES_DIR" --masks_dir "$MASKS_DIR"

# dinov3_large  Cond C  val_AUC=0.8869  (fine-tuned with masks, linear head)
echo ""
echo ">>> dinov3_large — Cond C  [dinov3_large_20260305_191824]"
$EVAL --run_dir runs/dinov3_large_20260305_191824 \
      --split test --images_dir "$IMAGES_DIR" --masks_dir "$MASKS_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Runs from Priority 2 (retrains with grid-search HPs)
#            Fill in run dir names after training completes.
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "--- Phase 2: Priority 2 retrains (fill in run dirs after training) ---"

# resnet18  Cond A  retrain  [lr=1e-3, wd=1e-5, bs=32, mlp, drop=0.3]
# TODO: replace <FILL> with actual run dir produced by train_all.sh
# echo ""
# echo ">>> resnet18 — Cond A retrain"
# $EVAL --run_dir runs/<FILL_resnet18_A> --split test --images_dir "$IMAGES_DIR"

# densenet121  Cond A  retrain  [lr=5e-4, wd=1e-4, bs=32, mlp, drop=0.3]
# TODO: replace <FILL> with actual run dir produced by train_all.sh
# echo ""
# echo ">>> densenet121 — Cond A retrain"
# $EVAL --run_dir runs/<FILL_densenet121_A> --split test --images_dir "$IMAGES_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Runs from Priority 3 (new cells in the matrix)
#            Fill in run dir names after training completes.
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "--- Phase 3: Priority 3 new runs (fill in run dirs after training) ---"

# resnet18  Cond B  [lr=1e-4, wd=1e-4, mlp, drop=0.5]
# $EVAL --run_dir runs/<FILL_resnet18_B> --split test --images_dir "$IMAGES_DIR"

# resnet18  Cond C  [lr=1e-4, wd=1e-4, mlp, drop=0.5, masks]
# $EVAL --run_dir runs/<FILL_resnet18_C> --split test --images_dir "$IMAGES_DIR" --masks_dir "$MASKS_DIR"

# resnet50  Cond A  [lr=1e-3, wd=1e-4, bs=16, mlp, drop=0.3]
# $EVAL --run_dir runs/<FILL_resnet50_A> --split test --images_dir "$IMAGES_DIR"

# resnet50  Cond B  [lr=1e-4, wd=1e-4, mlp, drop=0.5]
# $EVAL --run_dir runs/<FILL_resnet50_B> --split test --images_dir "$IMAGES_DIR"

# resnet50  Cond C  [lr=1e-4, wd=1e-4, mlp, drop=0.5, masks]
# $EVAL --run_dir runs/<FILL_resnet50_C> --split test --images_dir "$IMAGES_DIR" --masks_dir "$MASKS_DIR"

# clip_vit_base  Cond A  [lr=1e-3, wd=1e-4, bs=16, mlp, drop=0.3]
# $EVAL --run_dir runs/<FILL_clip_A> --split test --images_dir "$IMAGES_DIR"

# clip_vit_base  Cond B  [lr=1e-5, wd=1e-4, mlp, drop=0.5]
# $EVAL --run_dir runs/<FILL_clip_B> --split test --images_dir "$IMAGES_DIR"

# dinov2_base  Cond A (new, mlp head)  [lr=1e-5, wd=0.01, mlp, drop=0.3]
# $EVAL --run_dir runs/<FILL_dinov2_base_A> --split test --images_dir "$IMAGES_DIR"

# dinov2_base  Cond B retrain  [lr=1e-5, wd=0.01, mlp, drop=0.5]
# $EVAL --run_dir runs/<FILL_dinov2_base_B> --split test --images_dir "$IMAGES_DIR"

# dinov2_large  Cond A  [lr=1e-5, wd=0.01, mlp, drop=0.3]
# $EVAL --run_dir runs/<FILL_dinov2_large_A> --split test --images_dir "$IMAGES_DIR"

# dinov2_large  Cond B  [lr=1e-5, wd=0.01, mlp, drop=0.5]
# $EVAL --run_dir runs/<FILL_dinov2_large_B> --split test --images_dir "$IMAGES_DIR"

# dinov2_large  Cond C  [lr=1e-5, wd=0.01, mlp, drop=0.5, masks]
# $EVAL --run_dir runs/<FILL_dinov2_large_C> --split test --images_dir "$IMAGES_DIR" --masks_dir "$MASKS_DIR"

# dinov3_base  Cond A  [lr=1e-5, wd=0.01, mlp, drop=0.3]
# $EVAL --run_dir runs/<FILL_dinov3_base_A> --split test --images_dir "$IMAGES_DIR"

# dinov3_base  Cond B  [lr=1e-5, wd=0.01, mlp, drop=0.5]
# $EVAL --run_dir runs/<FILL_dinov3_base_B> --split test --images_dir "$IMAGES_DIR"

# dinov3_large  Cond A  [lr=1e-5, wd=0.01, mlp, drop=0.3]
# $EVAL --run_dir runs/<FILL_dinov3_large_A> --split test --images_dir "$IMAGES_DIR"

# dinov3_large  Cond B  [lr=1e-5, wd=0.01, mlp, drop=0.5]
# $EVAL --run_dir runs/<FILL_dinov3_large_B> --split test --images_dir "$IMAGES_DIR"

echo ""
echo "=========================================================="
echo "  Phase 1 evaluations complete."
echo "  Fill in Phase 2/3 run dirs and re-run after training."
echo "=========================================================="
