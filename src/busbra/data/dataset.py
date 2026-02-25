"""PyTorch Dataset for BUS-BRA.

Model-agnostic: __getitem__ returns raw PIL.Image + metadata.
Preprocessing (resize, normalize, augment) is handled externally via
preprocessing.py so any backbone (ImageNet CNN, CLIP, DINOv2, …) can
inject its own transform through the collate function.
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# define Dataset class
# PyTorch training expects:
#   __len__()       = number of samples
#   __getitem__(idx) = sample at index idx


class BUSBRADataset(Dataset):
    """BUS-BRA breast ultrasound dataset.

    Returns raw PIL.Image objects so that model-specific preprocessing
    can be applied outside the Dataset (see preprocessing.py).
    """

    def __init__(
        self,
        split_file: str | Path,  # path to splits.csv (ID, Case, label, split, …)
        images_dir: str | Path,  # directory containing <ID>.png files
        split: str,              # one of "train", "val", "test"
    ):
        self.images_dir = Path(images_dir)

        df = pd.read_csv(split_file)
        self.df = df[df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No samples for split '{split}'")
        print(f"Loaded {split}: {len(self.df)} images")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """Return a single sample.

        Returns
        -------
        dict with keys:
            "image"    : PIL.Image in RGB mode  (no resize/normalize)
            "label"    : int  — 0 benign, 1 malignant
            "case"     : str  — patient Case ID
            "image_id" : str  — image ID (used to build the filename)
        """
        row = self.df.iloc[idx]

        img_path = self.images_dir / row["filename"]
        # Convert to RGB so every backbone receives a 3-channel image
        image = Image.open(img_path).convert("RGB")

        # dataset.py returns PIL.Image (raw, unprocessed) + metadata; preprocessing.py applies model-specific transforms
        return {
            "image": image,
            "label": int(row["label"]),
            "case": str(row["Case"]),
            "image_id": str(row["ID"]),
        }
