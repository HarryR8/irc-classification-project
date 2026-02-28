"""DataLoader creation for BUS-BRA.

This module wires together:
  - BUSBRADataset  (model-agnostic; returns PIL.Image + metadata)
  - get_preprocess (model-specific: resize / augment / normalize / tensorize)
  - make_collate_fn (applies preprocess inside the DataLoader worker,
                     stacks tensors, and collects metadata lists)

Typical usage
-------------
    from busbra.data.loaders import create_dataloaders

    train_loader, val_loader, test_loader = create_dataloaders(
        split_file="data/splits/splits.csv",
        images_dir="data/raw",
        model_key="imagenet_cnn",   # or "clip", "dinov2", "dinov3"
        batch_size=32,
        num_workers=4,
        size=224,
    )

    for batch in train_loader:
        images = batch["image"]     # (B, 3, H, W) float32
        labels = batch["label"]     # (B,)          int64
        cases  = batch["case"]      # list[str]
        ids    = batch["image_id"]  # list[str]
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from busbra.data.dataset import BUSBRADataset
from busbra.data.preprocessing import get_preprocess


# ---------------------------------------------------------------------------
# Collate function factory
# ---------------------------------------------------------------------------

# Collate function to apply model-specific preprocessing inside DataLoader workers.
def make_collate_fn(preprocess_fn: Callable) -> Callable:
    """Return a collate function that applies `preprocess_fn` to each sample.

    Parameters
    ----------
    preprocess_fn : Callable[[PIL.Image], torch.FloatTensor]
        Model-specific transform returned by get_preprocess(…).

    Returns
    -------
    Callable[list[dict], dict]
        A collate function compatible with torch DataLoader's
        ``collate_fn`` argument.  The returned batch dict has:
            "image"    : (B, 3, H, W)  float32 tensor
            "label"    : (B,)          int64 tensor
            "case"     : list[str]     length B
            "image_id" : list[str]     length B
    """

    # The collate function is called inside the DataLoader worker process, so it can apply the model-specific preprocessing to each PIL image and stack them into a batch tensor.  Metadata like "case" and "image_id" are collected into lists.
    def collate(samples: list[dict]) -> dict:
        # Apply model-specific preprocessing to each PIL image
        images = torch.stack([preprocess_fn(s["image"]) for s in samples])  # (B,3,H,W)
        labels = torch.tensor([s["label"] for s in samples], dtype=torch.long)  # (B,)
        cases    = [s["case"]     for s in samples]
        image_ids = [s["image_id"] for s in samples]

        return {
            "image":    images,
            "label":    labels,
            "case":     cases,
            "image_id": image_ids,
        }

    return collate


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    split_file: str,
    images_dir: str,
    model_key: str = "imagenet_cnn",
    batch_size: int = 32,
    num_workers: int = 4,
    size: int = 224,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train / val / test DataLoaders.

    Parameters
    ----------
    split_file : str
        Path to splits.csv produced by prepare_data.py.
    images_dir : str
        Directory containing <ID>.png image files.
    model_key : str
        Backbone key passed to get_preprocess.
        One of "imagenet_cnn", "clip", "dinov2", "dinov3".
    batch_size : int
        Mini-batch size.
    num_workers : int
        DataLoader worker processes.
    size : int
        Target image size (used by imagenet_cnn pipeline).

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    # Build model-specific preprocessors per split
    preprocess_train = get_preprocess(model_key, split="train", size=size)
    preprocess_val   = get_preprocess(model_key, split="val",   size=size)
    preprocess_test  = get_preprocess(model_key, split="test",  size=size)

    # Datasets return raw PIL.Image; no transform stored inside Dataset
    datasets = {
        split: BUSBRADataset(split_file, images_dir, split)
        for split in ["train", "val", "test"]
    }

    # Weighted sampler on training set to compensate for class imbalance
    train_labels = datasets["train"].df["label"].values
    class_counts  = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(
        datasets["train"],
        sampler=sampler,
        drop_last=True,
        collate_fn=make_collate_fn(preprocess_train),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        datasets["val"],
        collate_fn=make_collate_fn(preprocess_val),
        **loader_kwargs,
    )
    test_loader = DataLoader(
        datasets["test"],
        collate_fn=make_collate_fn(preprocess_test),
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader
