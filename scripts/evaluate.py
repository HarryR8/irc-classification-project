"""Evaluate a trained BUS-BRA classifier on the val or test split.

Loads a checkpoint from a run directory produced by train.py, runs inference,
computes binary classification metrics at multiple probability thresholds, saves
a ROC curve PNG, a full threshold-sweep CSV, and a structured JSON report.

Typical usage
-------------
Evaluate the best checkpoint on the test split (default):
    uv run python scripts/evaluate.py --run_dir runs/resnet18_20260228_143021

Evaluate on the validation split instead:
    uv run python scripts/evaluate.py --run_dir runs/resnet18_20260228_143021 --split val

Use the test split both for evaluation and for threshold selection:
    uv run python scripts/evaluate.py --run_dir runs/resnet18_20260228_143021 \
        --split test --threshold_split same

Override data paths (e.g. on HPC):
    uv run python scripts/evaluate.py --run_dir runs/resnet18_20260228_143021 \
        --images_dir /path/to/data/raw --masks_dir /path/to/data/masks

CLI arguments
-------------
--run_dir           Path to the run directory (must contain config.json and best.pt).
--split             Split to evaluate: "val" or "test" (default: "test").
--images_dir        Directory containing the raw image files (default: data/raw).
--split_file        Path to the patient-level splits CSV (default: data/splits/splits.csv).
--masks_dir         Optional path to mask_*.png files; enables lesion-crop preprocessing.
--roc_png           Override the output path for the ROC curve PNG.
--threshold_split   Which split is used to select optimal thresholds:
                      "val"  — use validation set (default; avoids test-set leakage)
                      "test" — use the same split being evaluated
                      "same" — alias for the split passed to --split
--num_thresholds    Number of evenly-spaced thresholds to sweep in [0, 1] (default: 201).
--thresholds_csv    Override the output path for the threshold-sweep CSV.

Outputs written to --run_dir
----------------------------
eval_<split>.json                Structured results (AUC, metrics at 0.5, optimal thresholds).
eval_<split>_roc_curve.png       Publication-ready ROC curve at 300 dpi.
eval_<split>_threshold_sweep.csv Full per-threshold metrics table (num_thresholds rows).

Metrics reported
----------------
Baseline metrics are computed at threshold = 0.5.  Four additional operating points
are selected by optimising different clinical criteria from the threshold sweep:
  - ROC Youden J        max(TPR - FPR) on the ROC curve
  - Max F1              threshold that maximises F1 score
  - Sensitivity >= 0.95 highest specificity subject to sensitivity >= 0.95
  - Specificity >= 0.90 highest sensitivity subject to specificity >= 0.90

Metric library
--------------
Core metrics logic lives in src/busbra/training/metrics.py and is imported as:
    from busbra.training.metrics import metrics_at_threshold, find_optimal_thresholds
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Allow running this script directly while importing from the local src package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Use a writable local Matplotlib config/cache directory in all environments.
MPL_CONFIG_DIR = Path(__file__).resolve().parent.parent / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

# Optional API (newer sklearn): available point-wise confusion terms on score thresholds.
try:
    from sklearn.metrics import confusion_matrix_at_thresholds
except ImportError:
    confusion_matrix_at_thresholds = None

from busbra.data.loaders import create_dataloaders
from busbra.training.metrics import find_optimal_thresholds, metrics_at_threshold
from busbra.models import create_model, get_preprocess_key
from busbra.training import evaluate


def find_threshold_at_sensitivity(tpr, thresholds, target: float = 0.90) -> float:
    """Return the lowest threshold that achieves >= target sensitivity."""
    valid = np.where(tpr >= target)[0]
    return float(thresholds[valid[0]]) if len(valid) > 0 else 0.5


def parse_args():
    # Evaluation split and artifact/output controls.
    parser = argparse.ArgumentParser(description="Evaluate a BUS-BRA checkpoint")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--images_dir", type=str, default="data/raw")
    parser.add_argument("--split_file", type=str, default="data/splits/splits.csv")
    parser.add_argument("--masks_dir", type=str, default=None)
    parser.add_argument("--roc_png", type=str, default=None)
    parser.add_argument(
        "--threshold_split",
        type=str,
        default="val",
        choices=["val", "test", "same"],
        help="Split used to choose optimal thresholds. 'same' uses --split.",
    )
    parser.add_argument("--num_thresholds", type=int, default=201)
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader workers (overrides value stored in config.json)")
    parser.add_argument("--head_type", type=str, default=None,
                        choices=["linear", "mlp", "mlp_deep"],
                        help="Override head architecture from config.json")
    parser.add_argument(
        "--thresholds_csv",
        type=str,
        default=None,
        help="Optional path for threshold sweep CSV. Defaults to <run_dir>/eval_<split>_threshold_sweep.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a specific .pt checkpoint file (default: best.pt in run_dir)",
    )
    return parser.parse_args()


def resolve_run_artifacts(run_dir_arg: str, checkpoint_arg: str | None = None):
    # Resolve run directory and require the standard training artifacts.
    run_dir = Path(run_dir_arg).expanduser()
    run_dir = (Path.cwd() / run_dir).resolve() if not run_dir.is_absolute() else run_dir.resolve()
    config_path = run_dir / "config.json"
    if checkpoint_arg is not None:
        ckpt_path = Path(checkpoint_arg).expanduser().resolve()
    else:
        ckpt_path = run_dir / "best.pt"
    missing = [str(p) for p in (config_path, ckpt_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required file(s): {', '.join(missing)}")
    return run_dir, config_path, ckpt_path


def save_roc_curve(path: Path, split: str, fpr: np.ndarray, tpr: np.ndarray, auc: float):
    # Save a publication-ready ROC plot to disk.
    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=300)
    ax.plot(fpr, tpr, color="#1f77b4", lw=2.0, label=f"AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", lw=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"ROC Curve ({split} split)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _as_rounded_float(value: float) -> float | None:
    # JSON-friendly float conversion (NaN/inf -> None).
    value = float(value)
    if not np.isfinite(value):
        return None
    return round(value, 6)


def _serialize_metrics(metrics: dict[str, float | int]) -> dict[str, float | int | None]:
    # Keep confusion counts as integers, round metric floats.
    out: dict[str, float | int | None] = {}
    int_fields = {"tp", "fp", "tn", "fn"}
    for key, value in metrics.items():
        if key in int_fields:
            out[key] = int(value)
        else:
            out[key] = _as_rounded_float(float(value))
    return out


def _serialize_threshold_result(
    threshold: float | None, metrics: dict[str, float | int] | None
) -> dict[str, Any]:
    if threshold is None or metrics is None:
        return {"threshold": None, "metrics": None}
    # Threshold is already the key; avoid duplicating it inside metric payload.
    metrics_without_threshold = dict(metrics)
    metrics_without_threshold.pop("threshold", None)
    return {
        "threshold": _as_rounded_float(threshold),
        "metrics": _serialize_metrics(metrics_without_threshold),
    }


def main():
    args = parse_args()
    run_dir, config_path, ckpt_path = resolve_run_artifacts(args.run_dir, args.checkpoint)

    # Load model/training configuration tied to this run directory.
    with open(config_path) as f:
        config = json.load(f)

    print(f"Run dir   : {run_dir}")
    print(f"Model     : {config['model']}")
    print(f"Split     : {args.split}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}")

    # Build dataloaders with the same preprocessing key used for training.
    preprocess_key = get_preprocess_key(config["model"])
    _, val_loader, test_loader = create_dataloaders(
        split_file=args.split_file,
        images_dir=args.images_dir,
        model_key=preprocess_key,
        batch_size=config.get("batch_size", 32),
        num_workers=args.num_workers if args.num_workers is not None else config.get("num_workers", 4),
        masks_dir=args.masks_dir,
    )
    loader = val_loader if args.split == "val" else test_loader
    # Threshold search can be pinned to val/test/same (default: validation).
    threshold_source_split = args.split if args.threshold_split == "same" else args.threshold_split
    threshold_loader = val_loader if threshold_source_split == "val" else test_loader

    # Recreate model architecture and load the selected checkpoint.
    model = create_model(
        config["model"],
        num_classes=2,
        pretrained=False,
        freeze_backbone=config.get("freeze_backbone", False),
        head_type=args.head_type if args.head_type is not None else config.get("head_type", "linear"),
        head_dropout=config.get("dropout", 0.3),
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    checkpoint_val_auc = ckpt.get("val_auc")
    if checkpoint_val_auc is not None:
        print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val AUC during training: {checkpoint_val_auc:.4f})")
    else:
        print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    class_weights = torch.tensor([0.32, 0.68], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Primary evaluation on the requested output split.
    results = evaluate(model, loader, criterion, device)
    labels = np.asarray(results["labels"], dtype=np.int64)
    probs = np.asarray(results["probs"], dtype=np.float64)

    if threshold_source_split == args.split:
        # Reuse already computed probabilities if threshold source equals eval split.
        threshold_labels = labels
        threshold_probs = probs
    else:
        # Otherwise evaluate once more on threshold source split (usually validation).
        print(f"Computing threshold candidates from {threshold_source_split} split metrics...")
        threshold_results_eval = evaluate(model, threshold_loader, criterion, device)
        threshold_labels = np.asarray(threshold_results_eval["labels"], dtype=np.int64)
        threshold_probs = np.asarray(threshold_results_eval["probs"], dtype=np.float64)

    try:
        auc = float(roc_auc_score(labels, probs))
        fpr, tpr, _ = roc_curve(labels, probs)
    except ValueError:
        # Degenerate split with one class only: keep plotting/output resilient.
        auc = float("nan")
        fpr = np.array([0.0, 1.0], dtype=np.float64)
        tpr = np.array([0.0, 1.0], dtype=np.float64)

    # Keep standard binary metrics at the default operating threshold (0.5).
    baseline_metrics = metrics_at_threshold(labels, probs, threshold=0.5)
    accuracy = float(baseline_metrics["accuracy"])
    sensitivity = float(baseline_metrics["sensitivity"])
    specificity = float(baseline_metrics["specificity"])
    tn = int(baseline_metrics["tn"])
    fp = int(baseline_metrics["fp"])
    fn = int(baseline_metrics["fn"])
    tp = int(baseline_metrics["tp"])

    threshold_results = find_optimal_thresholds(
        threshold_labels, threshold_probs, num_thresholds=args.num_thresholds
    )
    metrics_df: pd.DataFrame = threshold_results["metrics_df"]
    threshold_candidates = {
        "by_roc_youden": threshold_results["by_roc_youden"],
        "by_max_f1": threshold_results["by_max_f1"],
        "by_target_sensitivity_95": threshold_results["by_target_sensitivity_95"],
        "by_target_specificity_90": threshold_results["by_target_specificity_90"],
    }
    threshold_candidate_metrics = {
        # Compute comparable metrics for each selected threshold candidate.
        key: (metrics_at_threshold(threshold_labels, threshold_probs, thr) if thr is not None else None)
        for key, thr in threshold_candidates.items()
    }

    # Save ROC figure for the evaluated split.
    roc_png = Path(args.roc_png).expanduser().resolve() if args.roc_png else (run_dir / f"eval_{args.split}_roc_curve.png")
    roc_png.parent.mkdir(parents=True, exist_ok=True)
    save_roc_curve(roc_png, args.split, fpr, tpr, auc)

    thresholds_csv = (
        Path(args.thresholds_csv).expanduser().resolve()
        if args.thresholds_csv
        else (run_dir / f"eval_{args.split}_threshold_sweep.csv")
    )
    thresholds_csv.parent.mkdir(parents=True, exist_ok=True)
    # Save the full threshold sweep table for analysis in notebooks/spreadsheets.
    metrics_df.to_csv(thresholds_csv, index=False)

    # Optional sklearn summary of confusion terms on score-derived thresholds.
    cm_threshold_points = None
    if confusion_matrix_at_thresholds is not None:
        try:
            _, _, _, _, cm_thresholds = confusion_matrix_at_thresholds(threshold_labels, threshold_probs)
            cm_threshold_points = int(cm_thresholds.shape[0])
        except ValueError:
            cm_threshold_points = None

    try:
        _, _, pr_thresholds = precision_recall_curve(labels, probs)
        pr_points = int(pr_thresholds.shape[0] + 1)
    except ValueError:
        pr_points = 0

    # Human-readable console summary.
    print("\n" + "=" * 52)
    print(f"  Evaluation - {args.split} split")
    print("=" * 52)
    print(f"  AUC-ROC     : {auc:.4f}")
    print(f"  Accuracy    : {accuracy:.4f}  ({int(tp + tn)}/{len(labels)})")
    print(f"  Sensitivity : {sensitivity:.4f}  (TP={tp})")
    print(f"  Specificity : {specificity:.4f}  (TN={tn})")
    print(f"  F1 @0.50    : {baseline_metrics['f1']:.4f}")
    print(f"  Optimal thresholds (from {threshold_source_split} metrics):")
    threshold_name_map = {
        "by_roc_youden": "ROC Youden J",
        "by_max_f1": "Max F1",
        "by_target_sensitivity_95": "Sensitivity >= 0.95",
        "by_target_specificity_90": "Specificity >= 0.90",
    }
    for key, label in threshold_name_map.items():
        thr = threshold_candidates[key]
        thr_metrics = threshold_candidate_metrics[key]
        if thr is None or thr_metrics is None:
            print(f"    - {label:<24}: not found")
            continue
        print(
            f"    - {label:<24}: thr={thr:.6f}  "
            f"sens={thr_metrics['sensitivity']:.4f}  "
            f"spec={thr_metrics['specificity']:.4f}  "
            f"f1={thr_metrics['f1']:.4f}"
        )
    print(f"  ROC curve saved          : {roc_png}")
    print(f"  Threshold sweep CSV      : {thresholds_csv}")
    print("=" * 52)

    best_threshold = threshold_candidates["by_roc_youden"]
    best_threshold_metrics = threshold_candidate_metrics["by_roc_youden"]
    optimal_thresholds_output = {
        # Persist each candidate threshold with its own metric snapshot.
        key: _serialize_threshold_result(threshold_candidates[key], threshold_candidate_metrics[key])
        for key in threshold_candidates
    }

    # Machine-readable output for downstream reporting.
    output = {
        "split": args.split,
        "threshold_source_split": threshold_source_split,
        "checkpoint_epoch": int(ckpt["epoch"]),
        "auc_roc": _as_rounded_float(auc),
        "accuracy": _as_rounded_float(accuracy),
        "sensitivity": _as_rounded_float(sensitivity),
        "specificity": _as_rounded_float(specificity),
        "precision": _as_rounded_float(float(baseline_metrics["precision"])),
        "npv": _as_rounded_float(float(baseline_metrics["npv"])),
        "f1": _as_rounded_float(float(baseline_metrics["f1"])),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "optimal_thresholds": optimal_thresholds_output,
        "best_threshold": _as_rounded_float(best_threshold) if best_threshold is not None else None,
        "best_threshold_sensitivity": (
            _as_rounded_float(float(best_threshold_metrics["sensitivity"]))
            if best_threshold_metrics is not None
            else None
        ),
        "best_threshold_specificity": (
            _as_rounded_float(float(best_threshold_metrics["specificity"]))
            if best_threshold_metrics is not None
            else None
        ),
        "roc_curve_png": str(roc_png),
        "threshold_sweep_csv": str(thresholds_csv),
        "threshold_grid_points": int(metrics_df.shape[0]),
        "roc_points": int(fpr.shape[0]),
        "pr_points": pr_points,
        "confusion_matrix_threshold_points": cm_threshold_points,
        "n_samples": int(len(labels)),
    }
    out_path = run_dir / f"eval_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
