#!/usr/bin/env bash
# train_cnn.sh — CNN + CLIP subset of the comparison matrix.
# Covers: resnet18 (A/B/C), resnet50 (A/B/C), densenet121 (A only),
#         clip_vit_base (A/B)
# Estimated runtime: ~13 h  →  PBS walltime=16:00:00
#
# Usage:
#   bash scripts/train_cnn.sh
#   bash scripts/train_cnn.sh --masks_dir /path/to/masks

set -euo pipefail

MASKS_DIR="data/masks"
if [[ "${1:-}" == "--masks_dir" && -n "${2:-}" ]]; then
    MASKS_DIR="$2"
fi

TRAIN="uv run python scripts/train.py"

echo "=========================================================="
echo "  train_cnn.sh — CNN + CLIP training"
echo "  masks_dir = $MASKS_DIR"
echo "=========================================================="

# ── resnet18 ────────────────────────────────────────────────────────────────
echo ""
echo ">>> resnet18 — A: freeze=True, no masks  [grid-search optimal: lr=1e-3, wd=1e-5, bs=32]"
$TRAIN --model resnet18 --freeze_backbone --head_type mlp --dropout 0.3 \
       --lr 1e-3 --weight_decay 1e-5 --batch_size 32 --epochs 50

echo ""
echo ">>> resnet18 — B: freeze=False, no masks"
$TRAIN --model resnet18 --head_type mlp --dropout 0.5 \
       --lr 1e-4 --weight_decay 1e-4 --epochs 100

echo ""
echo ">>> resnet18 — C: freeze=False, masks"
$TRAIN --model resnet18 --head_type mlp --dropout 0.5 \
       --lr 1e-4 --weight_decay 1e-4 --epochs 100 \
       --masks_dir "$MASKS_DIR"

# ── resnet50 ─────────────────────────────────────────────────────────────────
echo ""
echo ">>> resnet50 — A: freeze=True, no masks  [grid-search optimal: lr=1e-3, wd=1e-4, bs=16]"
$TRAIN --model resnet50 --freeze_backbone --head_type mlp --dropout 0.3 \
       --lr 1e-3 --weight_decay 1e-4 --batch_size 16 --epochs 50

echo ""
echo ">>> resnet50 — B: freeze=False, no masks"
$TRAIN --model resnet50 --head_type mlp --dropout 0.5 \
       --lr 1e-4 --weight_decay 1e-4 --epochs 100

echo ""
echo ">>> resnet50 — C: freeze=False, masks"
$TRAIN --model resnet50 --head_type mlp --dropout 0.5 \
       --lr 1e-4 --weight_decay 1e-4 --epochs 100 \
       --masks_dir "$MASKS_DIR"

# ── densenet121 ──────────────────────────────────────────────────────────────
echo ""
echo ">>> densenet121 — A: freeze=True, no masks  [RETRAIN: grid-search optimal lr=5e-4, wd=1e-4, bs=32]"
$TRAIN --model densenet121 --freeze_backbone --head_type mlp --dropout 0.3 \
       --lr 5e-4 --weight_decay 1e-4 --batch_size 32 --epochs 50
# B (densenet121_20260228_184613, val_AUC=0.9166) and
# C (densenet121_20260228_193830, val_AUC=0.9283) are already good — skipped.

# ── clip_vit_base ─────────────────────────────────────────────────────────────
echo ""
echo ">>> clip_vit_base — A: freeze=True, no masks  [grid-search optimal: lr=1e-3, wd=1e-4, bs=16]"
$TRAIN --model clip_vit_base --freeze_backbone --head_type mlp --dropout 0.3 \
       --lr 1e-3 --weight_decay 1e-4 --batch_size 16 --epochs 50

echo ""
echo ">>> clip_vit_base — B: freeze=False, no masks  [fine-tuning; CLIP needs much lower lr]"
$TRAIN --model clip_vit_base --head_type mlp --dropout 0.5 \
       --lr 1e-5 --weight_decay 1e-4 --epochs 100

echo ""
echo "=========================================================="
echo "  train_cnn.sh complete."
echo "=========================================================="
