"""
CLI entrypoint for training the BUS-BRA breast ultrasound classifier.

Example usage:
    python scripts/train.py --model resnet18 --epochs 30 --batch_size 32
    python scripts/train.py --model dinov2_base --epochs 30
    python scripts/train.py --model resnet18 --lr 5e-5 --epochs 30  # override default lr
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from busbra.data.loaders import create_dataloaders
from busbra.models import create_model, get_preprocess_key, count_parameters
from busbra.training import train_one_epoch, evaluate


# ── Per-model recommended training settings ────────────────────────────────────
# CLI arguments always override these when explicitly provided.
MODEL_TRAINING_CONFIGS = {
    "resnet18":        {"lr": 1e-4, "weight_decay": 1e-5, "warmup_epochs": 0, "freeze_backbone": False},
    "resnet50":        {"lr": 1e-4, "weight_decay": 1e-5, "warmup_epochs": 0, "freeze_backbone": False},
    "efficientnet_b0": {"lr": 1e-4, "weight_decay": 1e-5, "warmup_epochs": 0, "freeze_backbone": False},
    "densenet121":     {"lr": 1e-4, "weight_decay": 1e-5, "warmup_epochs": 0, "freeze_backbone": False},
    "dinov2_base":     {"lr": 1e-5, "weight_decay": 1e-2, "warmup_epochs": 5, "freeze_backbone": True},
    "clip_vit_base":   {"lr": 1e-5, "weight_decay": 1e-2, "warmup_epochs": 5, "freeze_backbone": True},
}
_DEFAULT_CONFIG = {"lr": 1e-4, "weight_decay": 1e-5, "warmup_epochs": 0, "freeze_backbone": False}


def resolve_config(args):
    """Fill None CLI args with per-model defaults. CLI always wins."""
    defaults = MODEL_TRAINING_CONFIGS.get(args.model, _DEFAULT_CONFIG)
    if args.lr is None:
        args.lr = defaults["lr"]
    if args.weight_decay is None:
        args.weight_decay = defaults["weight_decay"]
    if args.freeze_backbone is None:
        args.freeze_backbone = defaults["freeze_backbone"]
    args.warmup_epochs = defaults.get("warmup_epochs", 0)
    return args


def build_scheduler(optimizer, epochs, warmup_epochs):
    """Cosine annealing with optional linear warmup for ViT models."""
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train BUS-BRA classifier")

    parser.add_argument("--model", type=str, default="resnet18",
                        help="Model name from registry")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for all dataloaders")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate for AdamW (default: per-model config)")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="Weight decay for AdamW (default: per-model config)")
    parser.add_argument("--freeze_backbone", action="store_true", default=None,
                        help="Freeze backbone weights (only train head)")
    parser.add_argument("--head_type", type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="Head architecture on top of backbone")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save outputs. Defaults to runs/<model>_<timestamp>")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader worker processes")

    return parser.parse_args()


def main():
    args = parse_args()
    args = resolve_config(args)

    # ── Reproducibility ────────────────────────────────────────────────────────
    set_seed(args.seed)

    # ── Output directory ───────────────────────────────────────────────────────
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join("runs", f"{args.model}_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Device ─────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ── Data ───────────────────────────────────────────────────────────────────
    preprocess_key = get_preprocess_key(args.model)
    train_loader, val_loader, test_loader = create_dataloaders(
        split_file="data/splits/splits.csv",
        images_dir="data/raw",
        model_key=preprocess_key,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = create_model(
        args.model,
        num_classes=2,
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
        head_type=args.head_type if args.freeze_backbone else "linear",
    )
    model = model.to(device)

    param_info = count_parameters(model)
    print(f"Parameters — total: {param_info['total']:,}  |  trainable: {param_info['trainable']:,}")
    print(f"Config     — lr={args.lr}  weight_decay={args.weight_decay}  "
          f"freeze_backbone={args.freeze_backbone}  warmup_epochs={args.warmup_epochs}")

    # ── Loss, optimiser, scheduler ─────────────────────────────────────────────
    # Upweight malignant (minority class) to penalise missed cancers more heavily
    class_weights = torch.tensor([0.32, 0.68], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs)

    # ── Save config ────────────────────────────────────────────────────────────
    config = vars(args).copy()
    config["device"] = str(device)
    config["param_total"] = param_info["total"]
    config["param_trainable"] = param_info["trainable"]

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── Training loop ──────────────────────────────────────────────────────────
    history = []
    best_val_auc = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]

        epoch_record = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": round(train_metrics["loss"], 6),
            "train_auc": round(train_metrics["auc"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_auc": round(val_metrics["auc"], 6),
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | lr={current_lr:.2e} | "
            f"Train loss={train_metrics['loss']:.3f} auc={train_metrics['auc']:.3f} | "
            f"Val   loss={val_metrics['loss']:.3f} auc={val_metrics['auc']:.3f}"
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": best_val_auc,
                },
                os.path.join(args.output_dir, "best.pt"),
            )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_auc": val_metrics["auc"],
            },
            os.path.join(args.output_dir, "last.pt"),
        )

        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val AUC = {best_val_auc:.4f} at epoch {best_epoch}.")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
