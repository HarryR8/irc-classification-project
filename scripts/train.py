"""
CLI entrypoint for training the BUS-BRA breast ultrasound classifier.

Example usage:
    python scripts/train.py --model resnet18 --epochs 30 --batch_size 32 --lr 1e-4
    python scripts/train.py --model resnet18 --freeze_backbone --head_type mlp --epochs 2
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


def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Makes convolutions deterministic (slight speed cost)
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
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for AdamW")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone weights (only train head)")
    parser.add_argument("--head_type", type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="Head type when backbone is frozen")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save outputs. Defaults to runs/<model>_<timestamp>")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader worker processes")

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Reproducibility ────────────────────────────────────────────────────────
    set_seed(args.seed)

    # ── Output directory ───────────────────────────────────────────────────────
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join("runs", f"{args.model}_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Device ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    print(f"Parameters — total: {param_info['total']:,}  |  "
          f"trainable: {param_info['trainable']:,}")

    # ── Loss, optimiser, scheduler ─────────────────────────────────────────────
    # Class weights: upweight malignant (minority class)
    class_weights = torch.tensor([0.32, 0.68], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

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
        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Step scheduler after each epoch
        scheduler.step()

        # Record history (drop raw arrays — not JSON serialisable)
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_auc": round(train_metrics["auc"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_auc": round(val_metrics["auc"], 6),
        }
        history.append(epoch_record)

        # Print summary
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train loss={train_metrics['loss']:.3f} auc={train_metrics['auc']:.3f} | "
            f"Val   loss={val_metrics['loss']:.3f} auc={val_metrics['auc']:.3f}"
        )

        # Save best model
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

        # Always save latest checkpoint
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

        # Persist history after every epoch (safe against crashes)
        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val AUC = {best_val_auc:.4f} at epoch {best_epoch}.")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
