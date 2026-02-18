"""Sanity-check script for the BUS-BRA data pipeline.

Run from the repo root:
    python scripts/sanity_dataloader.py --model_key imagenet_cnn
    python scripts/sanity_dataloader.py --model_key clip
    python scripts/sanity_dataloader.py --model_key dinov2

For each split the script prints the batch tensor shape/dtype/range
and verifies that metadata lists have the expected length.
"""

import argparse
import sys
from pathlib import Path

# Allow running without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from busbra.data.loaders import create_dataloaders


def check_batch(batch: dict, split: str) -> None:
    """Print key statistics for one batch."""
    img    = batch["image"]
    label  = batch["label"]
    cases  = batch["case"]
    ids    = batch["image_id"]

    print(f"\n  [{split}]")
    print(f"    image  shape={tuple(img.shape)}  dtype={img.dtype}  "
          f"min={img.min():.3f}  max={img.max():.3f}")
    print(f"    label  shape={tuple(label.shape)}  dtype={label.dtype}  "
          f"unique={label.unique().tolist()}")
    print(f"    case   type={type(cases).__name__}  len={len(cases)}  "
          f"example={cases[0]!r}")
    print(f"    id     type={type(ids).__name__}   len={len(ids)}  "
          f"example={ids[0]!r}")

    # Assertions
    B = img.shape[0]
    assert label.shape == (B, 1),          f"label shape mismatch: {label.shape}"
    assert len(cases) == B,                f"case list length mismatch: {len(cases)}"
    assert len(ids)   == B,                f"image_id list length mismatch: {len(ids)}"
    assert img.is_floating_point(), "image tensor must be float"
    print(f"    ✓ assertions passed")


def main():
    parser = argparse.ArgumentParser(description="Sanity-check BUS-BRA dataloaders")
    parser.add_argument(
        "--model_key", default="imagenet_cnn",
        choices=["imagenet_cnn", "clip", "dinov2", "dinov3"],
        help="Backbone key to test",
    )
    parser.add_argument("--split_file",  default="data/splits/splits.csv")
    parser.add_argument("--images_dir",  default="data/raw")
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="0 = main process (easier to debug)")
    parser.add_argument("--size",        type=int, default=224)
    args = parser.parse_args()

    print(f"Testing model_key='{args.model_key}'  size={args.size}")
    print(f"split_file : {args.split_file}")
    print(f"images_dir : {args.images_dir}")

    train_loader, val_loader, test_loader = create_dataloaders(
        split_file=args.split_file,
        images_dir=args.images_dir,
        model_key=args.model_key,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        size=args.size,
    )

    for split, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        batch = next(iter(loader))
        check_batch(batch, split)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
