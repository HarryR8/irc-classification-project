#!/usr/bin/env bash
# train_all.sh — run all training jobs needed to fill the 7-model × 3-condition
# comparison matrix with standardised hyperparameters.
#
# Conditions
#   A  freeze=True,  no masks   (frozen backbone, head-only training)
#   B  freeze=False, no masks   (full fine-tuning)
#   C  freeze=False, with masks (full fine-tuning + lesion-crop preprocessing)
#
# Standard hyperparameters by group (updated with grid-search results from 2026-03-10)
#   CNN  (resnet18, resnet50, efficientnet_b0, densenet121)
#     A: per-model grid-search-optimal lr/wd/bs  mlp  dropout=0.3  epochs=50
#     B/C: lr=1e-4  wd=1e-4  mlp  dropout=0.5  epochs=100
#   CLIP (clip_vit_base)
#     A: lr=1e-3  wd=1e-4  bs=16  mlp  dropout=0.3  epochs=50
#     B: lr=1e-5  wd=1e-4  mlp  dropout=0.5  epochs=100  [fine-tuning, lower lr]
#   DINO (dinov2_base, dinov2_large, dinov3_base, dinov3_large)
#     A: lr=1e-5  wd=0.01  mlp  dropout=0.3  epochs=50
#     B/C: lr=1e-5  wd=0.01  mlp  dropout=0.5  epochs=100
#
# Grid search results (condition A, frozen backbone, 2026-03-10):
#   densenet121:   best lr=5e-4, wd=1e-4, bs=32  val_AUC=0.9392
#   resnet50:      best lr=1e-3, wd=1e-4, bs=16  val_AUC=0.9220
#   resnet18:      best lr=1e-3, wd=1e-5, bs=32  val_AUC=0.9144
#   clip_vit_base: best lr=1e-3, wd=1e-4, bs=16  val_AUC=0.7726
#
# Existing runs that are already good — DO NOT retrain:
#   efficientnet_b0_lr5e4_wd1e5_bs16_mlp_es  (A)  val_AUC=0.9094  [from train_efficientnet_hpc.pbs]
#   efficientnet_b0_20260228_181632           (B)  val_AUC=0.8625
#   efficientnet_b0_20260228_200621           (C)  val_AUC=0.9054
#   densenet121_20260228_184613               (B)  val_AUC=0.9166
#   densenet121_20260228_193830               (C)  val_AUC=0.9283
#   dinov2_base_20260305_171950               (C)  val_AUC=0.8878
#   dinov3_base_20260305_183935               (C)  val_AUC=0.8984  [linear head, acceptable]
#   dinov3_large_20260305_191824              (C)  val_AUC=0.8869  [linear head, acceptable]
#
# Usage
#   bash scripts/train_all.sh
#   bash scripts/train_all.sh --masks_dir /path/to/masks  # override mask path (HPC)

set -euo pipefail

MASKS_DIR="data/masks"
# Allow overriding masks path for HPC environments:
#   bash scripts/train_all.sh --masks_dir /rds/general/.../data/masks
if [[ "${1:-}" == "--masks_dir" && -n "${2:-}" ]]; then
    MASKS_DIR="$2"
fi

TRAIN="uv run python scripts/train.py"

echo "=========================================================="
echo "  train_all.sh — BUS-BRA comparison matrix training"
echo "  masks_dir = $MASKS_DIR"
echo "=========================================================="

# ── resnet18 ────────────────────────────────────────────────────────────────
echo ""
echo ">>> resnet18 — A: freeze=True, no masks  [grid-search optimal: lr=1e-3, wd=1e-5, bs=32]"
$TRAIN --model resnet18 --freeze_backbone --head_type mlp --dropout 0.3 \
       --lr 1e-3 --weight_decay 1e-5 --batch_size 32 --epochs 50

echo ""
echo ">>> resnet18 — B: freeze=False, no masks  [NEW]"
$TRAIN --model resnet18 --head_type mlp --dropout 0.5 \
       --lr 1e-4 --weight_decay 1e-4 --epochs 100

echo ""
echo ">>> resnet18 — C: freeze=False, masks  [NEW]"
$TRAIN --model resnet18 --head_type mlp --dropout 0.5 \
       --lr 1e-4 --weight_decay 1e-4 --epochs 100 \
       --masks_dir "$MASKS_DIR"

# ── resnet50 ─────────────────────────────────────────────────────────────────
echo ""
echo ">>> resnet50 — A: freeze=True, no masks  [grid-search optimal: lr=1e-3, wd=1e-4, bs=16]"
$TRAIN --model resnet50 --freeze_backbone --head_type mlp --dropout 0.3 \
       --lr 1e-3 --weight_decay 1e-4 --batch_size 16 --epochs 50

echo ""
echo ">>> resnet50 — B: freeze=False, no masks  [NEW]"
$TRAIN --model resnet50 --head_type mlp --dropout 0.5 \
       --lr 1e-4 --weight_decay 1e-4 --epochs 100

echo ""
echo ">>> resnet50 — C: freeze=False, masks  [NEW]"
$TRAIN --model resnet50 --head_type mlp --dropout 0.5 \
       --lr 1e-4 --weight_decay 1e-4 --epochs 100 \
       --masks_dir "$MASKS_DIR"

# ── densenet121 ──────────────────────────────────────────────────────────────
echo ""
echo ">>> densenet121 — A: freeze=True, no masks  [RETRAIN: grid-search optimal lr=5e-4, wd=1e-4, bs=32, prev AUC=0.7514]"
$TRAIN --model densenet121 --freeze_backbone --head_type mlp --dropout 0.3 \
       --lr 5e-4 --weight_decay 1e-4 --batch_size 32 --epochs 50
# B (densenet121_20260228_184613, val_AUC=0.9166) and
# C (densenet121_20260228_193830, val_AUC=0.9283) are already good — DO NOT retrain.

# ── clip_vit_base ─────────────────────────────────────────────────────────────
echo ""
echo ">>> clip_vit_base — A: freeze=True, no masks  [grid-search optimal: lr=1e-3, wd=1e-4, bs=16]"
$TRAIN --model clip_vit_base --freeze_backbone --head_type mlp --dropout 0.3 \
       --lr 1e-3 --weight_decay 1e-4 --batch_size 16 --epochs 50

echo ""
echo ">>> clip_vit_base — B: freeze=False, no masks  [fine-tuning; CLIP needs much lower lr]"
$TRAIN --model clip_vit_base --head_type mlp --dropout 0.5 \
       --lr 1e-5 --weight_decay 1e-4 --epochs 100

# ── dinov2_base ──────────────────────────────────────────────────────────────
echo ""
echo ">>> dinov2_base — A: freeze=True, no masks  [NEW]"
$TRAIN --model dinov2_base --freeze_backbone --head_type mlp --dropout 0.3 \
       --lr 1e-5 --weight_decay 0.01 --epochs 50

echo ""
echo ">>> dinov2_base — B: freeze=False, no masks  [RETRAIN: lr=1e-4 caused severe overfit, AUC=0.68]"
$TRAIN --model dinov2_base --head_type mlp --dropout 0.5 \
       --lr 1e-5 --weight_decay 0.01 --epochs 100

# ── dinov2_large ─────────────────────────────────────────────────────────────
echo ""
echo ">>> dinov2_large — A: freeze=True, no masks  [NEW]"
$TRAIN --model dinov2_large --freeze_backbone --head_type mlp --dropout 0.3 \
       --lr 1e-5 --weight_decay 0.01 --epochs 50

echo ""
echo ">>> dinov2_large — B: freeze=False, no masks  [NEW]"
$TRAIN --model dinov2_large --head_type mlp --dropout 0.5 \
       --lr 1e-5 --weight_decay 0.01 --epochs 100

echo ""
echo ">>> dinov2_large — C: freeze=False, masks  [NEW]"
$TRAIN --model dinov2_large --head_type mlp --dropout 0.5 \
       --lr 1e-5 --weight_decay 0.01 --epochs 100 \
       --masks_dir "$MASKS_DIR"

# ── dinov3_base ───────────────────────────────────────────────────────────────
echo ""
echo ">>> dinov3_base — A: freeze=True, no masks  [RETRAIN: was linear head (1538 params), AUC=0.80]"
$TRAIN --model dinov3_base --freeze_backbone --head_type mlp --dropout 0.3 \
       --lr 1e-5 --weight_decay 0.01 --epochs 50

echo ""
echo ">>> dinov3_base — B: freeze=False, no masks  [NEW]"
$TRAIN --model dinov3_base --head_type mlp --dropout 0.5 \
       --lr 1e-5 --weight_decay 0.01 --epochs 100

# ── dinov3_large ──────────────────────────────────────────────────────────────
echo ""
echo ">>> dinov3_large — A: freeze=True, no masks  [NEW — previous run had no best.pt]"
$TRAIN --model dinov3_large --freeze_backbone --head_type mlp --dropout 0.3 \
       --lr 1e-5 --weight_decay 0.01 --epochs 50

echo ""
echo ">>> dinov3_large — B: freeze=False, no masks  [NEW]"
$TRAIN --model dinov3_large --head_type mlp --dropout 0.5 \
       --lr 1e-5 --weight_decay 0.01 --epochs 100

echo ""
echo "=========================================================="
echo "  All training jobs complete."
echo "  Next: update run dirs in scripts/run_all_evals.sh and"
echo "  run:  bash scripts/run_all_evals.sh"
echo "=========================================================="
