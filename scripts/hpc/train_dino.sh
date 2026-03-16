#!/usr/bin/env bash
# train_dino.sh — DINO subset of the comparison matrix.
# Covers: dinov2_base (A/B), dinov2_large (A/B/C),
#         dinov3_base (A/B), dinov3_large (A/B)
# Estimated runtime: ~26 h  →  PBS walltime=36:00:00
#
# Usage:
#   bash scripts/train_dino.sh
#   bash scripts/train_dino.sh --masks_dir /path/to/masks

set -euo pipefail

MASKS_DIR="data/masks"
if [[ "${1:-}" == "--masks_dir" && -n "${2:-}" ]]; then
    MASKS_DIR="$2"
fi

TRAIN="uv run python scripts/train.py"

echo "=========================================================="
echo "  train_dino.sh — DINO training"
echo "  masks_dir = $MASKS_DIR"
echo "=========================================================="

# ── dinov2_base ──────────────────────────────────────────────────────────────
echo ""
echo ">>> dinov2_base — A: freeze=True, no masks"
$TRAIN --model dinov2_base --freeze_backbone --head_type mlp --dropout 0.3 \
       --lr 1e-5 --weight_decay 0.01 --epochs 50

echo ""
echo ">>> dinov2_base — B: freeze=False, no masks  [RETRAIN: lr=1e-4 caused severe overfit]"
$TRAIN --model dinov2_base --head_type mlp --dropout 0.5 \
       --lr 1e-5 --weight_decay 0.01 --epochs 100

# ── dinov2_large ─────────────────────────────────────────────────────────────
echo ""
echo ">>> dinov2_large — A: freeze=True, no masks"
$TRAIN --model dinov2_large --freeze_backbone --head_type mlp --dropout 0.3 \
       --lr 1e-5 --weight_decay 0.01 --epochs 50

echo ""
echo ">>> dinov2_large — B: freeze=False, no masks"
$TRAIN --model dinov2_large --head_type mlp --dropout 0.5 \
       --lr 1e-5 --weight_decay 0.01 --epochs 100

echo ""
echo ">>> dinov2_large — C: freeze=False, masks"
$TRAIN --model dinov2_large --head_type mlp --dropout 0.5 \
       --lr 1e-5 --weight_decay 0.01 --epochs 100 \
       --masks_dir "$MASKS_DIR"

# ── dinov3_base ───────────────────────────────────────────────────────────────
echo ""
echo ">>> dinov3_base — A: freeze=True, no masks  [RETRAIN: was linear head, AUC=0.80]"
$TRAIN --model dinov3_base --freeze_backbone --head_type mlp --dropout 0.3 \
       --lr 1e-5 --weight_decay 0.01 --epochs 50

echo ""
echo ">>> dinov3_base — B: freeze=False, no masks"
$TRAIN --model dinov3_base --head_type mlp --dropout 0.5 \
       --lr 1e-5 --weight_decay 0.01 --epochs 100

# ── dinov3_large ──────────────────────────────────────────────────────────────
echo ""
echo ">>> dinov3_large — A: freeze=True, no masks  [RETRAIN: previous run had no best.pt]"
$TRAIN --model dinov3_large --freeze_backbone --head_type mlp --dropout 0.3 \
       --lr 1e-5 --weight_decay 0.01 --epochs 50

echo ""
echo ">>> dinov3_large — B: freeze=False, no masks"
$TRAIN --model dinov3_large --head_type mlp --dropout 0.5 \
       --lr 1e-5 --weight_decay 0.01 --epochs 100

echo ""
echo "=========================================================="
echo "  train_dino.sh complete."
echo "=========================================================="
