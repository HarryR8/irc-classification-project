#!/usr/bin/env bash
# Train the full comparison matrix: 9 models × 3 conditions (A/B/C).
#
# Condition A — freeze_backbone, no masks, 50 epochs, patience 15
# Condition B — full fine-tune,   no masks, 100 epochs, patience 20
# Condition C — full fine-tune,   masks,    100 epochs, patience 20
#
# All runs include --eval_test_every_epoch to produce epoch_test_preds.npz.
#
# Usage:
#   bash scripts/train_all.sh [--masks_dir /path/to/masks]
#
# On HPC this script is called by train_all_hpc.pbs which passes --masks_dir.

set -euo pipefail

# ── Parse optional --masks_dir argument ──────────────────────────────────────
MASKS_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --masks_dir)
            MASKS_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

TRAIN="uv run python scripts/train.py"
SPLIT_FILE="data/splits/splits.csv"
IMAGES_DIR="data/raw"
COMMON="--split_file $SPLIT_FILE --images_dir $IMAGES_DIR --eval_test_every_epoch --head_type mlp"

# ── Condition A — frozen backbone, 50 epochs, patience 15 ───────────────────
echo "=== Condition A: frozen backbone ==="

$TRAIN --model resnet18        $COMMON --freeze_backbone --epochs 50 --patience 15 --lr 1e-3  --weight_decay 1e-5 --batch_size 32 --dropout 0.3
$TRAIN --model resnet50        $COMMON --freeze_backbone --epochs 50 --patience 15 --lr 1e-3  --weight_decay 1e-4 --batch_size 16 --dropout 0.3
$TRAIN --model densenet121     $COMMON --freeze_backbone --epochs 50 --patience 15 --lr 5e-4  --weight_decay 1e-4 --batch_size 32 --dropout 0.3
$TRAIN --model efficientnet_b0 $COMMON --freeze_backbone --epochs 50 --patience 15 --lr 5e-4  --weight_decay 1e-5 --batch_size 16 --dropout 0.3
$TRAIN --model clip_vit_base   $COMMON --freeze_backbone --epochs 50 --patience 15 --lr 1e-3  --weight_decay 1e-4 --batch_size 16 --dropout 0.3
$TRAIN --model dinov2_base     $COMMON --freeze_backbone --epochs 50 --patience 15 --lr 1e-5  --weight_decay 1e-2 --batch_size 32 --dropout 0.3
$TRAIN --model dinov2_large    $COMMON --freeze_backbone --epochs 50 --patience 15 --lr 1e-5  --weight_decay 1e-2 --batch_size 16 --dropout 0.3
$TRAIN --model dinov3_base     $COMMON --freeze_backbone --epochs 50 --patience 15 --lr 1e-5  --weight_decay 1e-2 --batch_size 32 --dropout 0.3
$TRAIN --model dinov3_large    $COMMON --freeze_backbone --epochs 50 --patience 15 --lr 1e-5  --weight_decay 1e-2 --batch_size 16 --dropout 0.3

# ── Condition B — full fine-tune, no masks, 100 epochs, patience 20 ─────────
echo "=== Condition B: full fine-tune, no masks ==="

$TRAIN --model resnet18        $COMMON --epochs 100 --patience 20 --lr 1e-4 --weight_decay 1e-5 --batch_size 32 --dropout 0.5
$TRAIN --model resnet50        $COMMON --epochs 100 --patience 20 --lr 1e-4 --weight_decay 1e-5 --batch_size 16 --dropout 0.5
$TRAIN --model densenet121     $COMMON --epochs 100 --patience 20 --lr 1e-4 --weight_decay 1e-5 --batch_size 32 --dropout 0.5
$TRAIN --model efficientnet_b0 $COMMON --epochs 100 --patience 20 --lr 1e-4 --weight_decay 1e-5 --batch_size 16 --dropout 0.5
$TRAIN --model clip_vit_base   $COMMON --epochs 100 --patience 20 --lr 1e-5 --weight_decay 1e-2 --batch_size 16 --dropout 0.5
$TRAIN --model dinov2_base     $COMMON --epochs 100 --patience 20 --lr 1e-5 --weight_decay 1e-2 --batch_size 32 --dropout 0.5 --backbone_lr_scale 0.1
$TRAIN --model dinov2_large    $COMMON --epochs 100 --patience 20 --lr 1e-5 --weight_decay 1e-2 --batch_size 16 --dropout 0.5 --backbone_lr_scale 0.1
$TRAIN --model dinov3_base     $COMMON --epochs 100 --patience 20 --lr 1e-5 --weight_decay 1e-2 --batch_size 32 --dropout 0.5 --backbone_lr_scale 0.1
$TRAIN --model dinov3_large    $COMMON --epochs 100 --patience 20 --lr 1e-5 --weight_decay 1e-2 --batch_size 16 --dropout 0.5 --backbone_lr_scale 0.1

# ── Condition C — full fine-tune, with masks, 100 epochs, patience 20 ───────
echo "=== Condition C: full fine-tune, with masks ==="

if [[ -z "$MASKS_DIR" ]]; then
    echo "WARNING: --masks_dir not provided; skipping condition C runs."
else
    $TRAIN --model resnet18        $COMMON --masks_dir "$MASKS_DIR" --epochs 100 --patience 20 --lr 1e-4 --weight_decay 1e-5 --batch_size 32 --dropout 0.5
    $TRAIN --model resnet50        $COMMON --masks_dir "$MASKS_DIR" --epochs 100 --patience 20 --lr 1e-4 --weight_decay 1e-5 --batch_size 16 --dropout 0.5
    $TRAIN --model densenet121     $COMMON --masks_dir "$MASKS_DIR" --epochs 100 --patience 20 --lr 1e-4 --weight_decay 1e-5 --batch_size 32 --dropout 0.5
    $TRAIN --model efficientnet_b0 $COMMON --masks_dir "$MASKS_DIR" --epochs 100 --patience 20 --lr 1e-4 --weight_decay 1e-5 --batch_size 16 --dropout 0.5
    $TRAIN --model clip_vit_base   $COMMON --masks_dir "$MASKS_DIR" --epochs 100 --patience 20 --lr 1e-5 --weight_decay 1e-2 --batch_size 16 --dropout 0.5
    $TRAIN --model dinov2_base     $COMMON --masks_dir "$MASKS_DIR" --epochs 100 --patience 20 --lr 1e-5 --weight_decay 1e-2 --batch_size 32 --dropout 0.5 --backbone_lr_scale 0.1
    $TRAIN --model dinov2_large    $COMMON --masks_dir "$MASKS_DIR" --epochs 100 --patience 20 --lr 1e-5 --weight_decay 1e-2 --batch_size 16 --dropout 0.5 --backbone_lr_scale 0.1
    $TRAIN --model dinov3_base     $COMMON --masks_dir "$MASKS_DIR" --epochs 100 --patience 20 --lr 1e-5 --weight_decay 1e-2 --batch_size 32 --dropout 0.5 --backbone_lr_scale 0.1
    $TRAIN --model dinov3_large    $COMMON --masks_dir "$MASKS_DIR" --epochs 100 --patience 20 --lr 1e-5 --weight_decay 1e-2 --batch_size 16 --dropout 0.5 --backbone_lr_scale 0.1
fi

echo ""
echo "=== train_all.sh complete ==="
