"""Evaluate a trained BUS-BRA classifier on val or test split.

Example usage:
    python scripts/evaluate.py --run_dir runs/efficientnet_b0_20260228_192617
    python scripts/evaluate.py --run_dir runs/resnet18_20260228_143021 --split val
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from busbra.data.loaders import create_dataloaders
from busbra.models import create_model, get_preprocess_key
from busbra.training import evaluate


def find_threshold_at_sensitivity(tpr, thresholds, target: float = 0.90) -> float:
    """Return the lowest threshold that achieves >= target sensitivity."""
    valid = np.where(tpr >= target)[0]
    return float(thresholds[valid[0]]) if len(valid) > 0 else 0.5


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """Return accuracy, sensitivity, specificity, precision, and F1 for a given threshold."""
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    accuracy    = (tp + tn) / len(labels)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    precision   = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    denom = 2 * tp + fp + fn
    f1_score    = (2 * tp / denom) if denom > 0 else float("nan")
    return dict(
        tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
        accuracy=accuracy, sensitivity=sensitivity,
        specificity=specificity, precision=precision, f1_score=f1_score,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a BUS-BRA checkpoint")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to run directory (contains config.json and best.pt)")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"],
                        help="Split to evaluate on")
    parser.add_argument("--images_dir", type=str, default="data/raw",
                        help="Directory containing image files")
    parser.add_argument("--split_file", type=str, default="data/splits/splits.csv",
                        help="Path to splits CSV file")
    parser.add_argument("--masks_dir", type=str, default=None,
                        help="Directory containing mask_*.png segmentation masks "
                             "(enables lesion-crop preprocessing)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load run config ───────────────────────────────────────────────────────
    config_path = os.path.join(args.run_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    print(f"Run dir   : {args.run_dir}")
    print(f"Model     : {config['model']}")
    print(f"Split     : {args.split}")

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    preprocess_key = get_preprocess_key(config["model"])
    _, val_loader, test_loader = create_dataloaders(
        split_file=args.split_file,
        images_dir=args.images_dir,
        model_key=preprocess_key,
        batch_size=config.get("batch_size", 32),
        num_workers=config.get("num_workers", 4),
        masks_dir=args.masks_dir,
    )
    loader = val_loader if args.split == "val" else test_loader

    # ── Reconstruct model ─────────────────────────────────────────────────────
    model = create_model(
        config["model"],
        num_classes=2,
        pretrained=False,          # weights come from checkpoint, not ImageNet
        freeze_backbone=config.get("freeze_backbone", False),
        head_type=config.get("head_type", "linear"),
        head_dropout=config.get("dropout", 0.3),
    )

    ckpt_path = os.path.join(args.run_dir, "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val AUC during training: {ckpt['val_auc']:.4f})")

    # ── Inference ─────────────────────────────────────────────────────────────
    class_weights = torch.tensor([0.32, 0.68], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    results = evaluate(model, loader, criterion, device)
    labels = results["labels"]   # (N,) int
    probs  = results["probs"]    # (N,) float — P(malignant)

    # ── Metrics ───────────────────────────────────────────────────────────────
    auc = roc_auc_score(labels, probs)
    _, tpr, thresholds = roc_curve(labels, probs)

    threshold_90sens = find_threshold_at_sensitivity(tpr, thresholds, target=0.90)
    m = compute_metrics(labels, probs, threshold_90sens)
    tp, tn, fp, fn = m["tp"], m["tn"], m["fp"], m["fn"]
    accuracy, sensitivity, specificity = m["accuracy"], m["sensitivity"], m["specificity"]
    precision, f1_score = m["precision"], m["f1_score"]

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print(f"  Evaluation — {args.split} split")
    print("=" * 52)
    print(f"  AUC-ROC     : {auc:.4f}")
    print(f"  Accuracy    : {accuracy:.4f}  ({int(tp + tn)}/{len(labels)})")
    print(f"  Sensitivity : {sensitivity:.4f}  (malignant recall, TP={tp})")
    print(f"  Specificity : {specificity:.4f}  (benign recall,    TN={tn})")
    print()
    print("  Confusion matrix (rows=actual, cols=predicted):")
    print("                Pred benign  Pred malignant")
    print(f"  Act benign        {tn:5d}          {fp:5d}")
    print(f"  Act malignant     {fn:5d}          {tp:5d}")
    print("=" * 52)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "split": args.split,
        "checkpoint_epoch": int(ckpt["epoch"]),
        "auc_roc": round(float(auc), 6),
        "accuracy": round(float(accuracy), 6),
        "sensitivity": round(float(sensitivity), 6),
        "specificity": round(float(specificity), 6),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
        "n_samples": len(labels),
    }
    out_path = os.path.join(args.run_dir, f"eval_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
