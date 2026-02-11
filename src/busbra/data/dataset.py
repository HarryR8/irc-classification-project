"""PyTorch Dataset for BUS-BRA."""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class BUSBRADataset(Dataset):
    """BUS-BRA breast ultrasound dataset."""
    
    def __init__(
        self,
        split_file: str | Path,
        images_dir: str | Path,
        split: str,
        transform: Optional[Callable] = None,
    ):
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        df = pd.read_csv(split_file)
        self.df = df[df["split"] == split].reset_index(drop=True)
        
        if len(self.df) == 0:
            raise ValueError(f"No samples for split '{split}'")
        print(f"Loaded {split}: {len(self.df)} images")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load grayscale and convert to 3-channel for ImageNet-pretrained models
        img_path = self.images_dir / f"{row['ID']}.png"
        image = np.array(Image.open(img_path).convert("RGB"))

        # Apply transforms
        if self.transform:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return {
            "image": image,
            "label": int(row["label"]),
            "case": row["Case"],
            "image_id": row["ID"],
        }


def get_transforms(split: str, size: int = 224):
    """Albumentations transforms."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    # letterboxing, black padding - handle input image aspect ratio
    letterbox = [
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, value=0),
    ]

    if split == "train":
        return A.Compose([
            *letterbox,
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=0, value=0, p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    return A.Compose([
        *letterbox,
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def create_dataloaders(split_file, images_dir, batch_size=32, num_workers=4, size=224):
    """Create train/val/test dataloaders."""
    datasets = {
        split: BUSBRADataset(split_file, images_dir, split, get_transforms(split, size))
        for split in ["train", "val", "test"]
    }

    # Weighted sampler from training labels only to handle class imbalance
    train_labels = datasets["train"].df["label"].values
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    loader_kwargs = dict(
        batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return (
        DataLoader(datasets["train"], sampler=sampler, drop_last=True, **loader_kwargs),
        DataLoader(datasets["val"], **loader_kwargs),
        DataLoader(datasets["test"], **loader_kwargs),
    )
