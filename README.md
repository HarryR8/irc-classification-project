# _SonoNet_ - IRC Classification Project
**Classification of benign/malignant tumours from medical images (breast ultrasound).**

## Goal
Build a **reproducible** ML pipeline to classify **benign vs malignant** tumours, with honest evaluation (patient-level splits where applicable).

## Datasets
  - **BUS-BRA (primary)** - 1875 images / 1064 patients, BI-RADS + tumour delineations. **CC BY 4.0**
  Links: https://github.com/wgomezf/BUS-BRA | https://zenodo.org/records/8231412
- **BUS-UCLM (extension: cross-dataset generalisation)** - 683 images / 38 patients, normal/benign/malignant + masks. **CC BY 4.0**
  Links: https://github.com/noeliavallez/BUS-UCLM-Dataset

## Approach (preliminary)
- Baseline: transfer learning CNN (ResNet18, EfficientNet-B0, DenseNet121)
- ViT backbones: DINOv2 (Base / Large) and CLIP via HuggingFace
- Mask-aware (extension): ROI-cropped classification (ROI-only vs ROI+context)
- Evaluation: ROC-AUC/PR-AUC/F1 + confusion matrix (focus on malignant recall)

## Repo structure
```
irc-classification-project/
├── pyproject.toml          # Dependencies + uv config
├── .gitignore              # Data, checkpoints, envs excluded
├── README.md               # Setup instructions, usage guide
├── scripts/
│   └── sanity_dataloader.py  # ✅ Verify batch shapes/dtypes for any backbone
├── src/busbra/
│   ├── train.py            # Training loop with early stopping (empty)
│   ├── data/
│   │   ├── prepare_data.py   # ✅ Load CSVs, create patient-level splits
│   │   ├── dataset.py        # ✅ Model-agnostic PyTorch Dataset (returns PIL.Image)
│   │   ├── preprocessing.py  # ✅ Backbone-specific preprocessing registry
│   │   └── loaders.py        # ✅ Collate functions + DataLoader factory
│   ├── models/
│   │   └── model.py        # CNN model (empty)
│   └── evaluation/
│       └── evaluate.py     # (empty)
└── data/
    ├── raw/            ← 🚨 put dataset (BUS-BRA) here
    │   ├── bus_data.csv
    │   ├── bus_0001-l.png
    │   └── ...
    └── splits/         ← ✅ created by prepare_data.py
        ├── patient_splits.csv  # patient ID + split assignment (for auditing)
        ├── split_info.json     # split metadata
        └── splits.csv          # ID + Case + label + split (primary file used downstream)
```

## Data pipeline

The data pipeline is **model-agnostic**: the `BUSBRADataset` returns raw `PIL.Image` objects and metadata, and all backbone-specific preprocessing (resize, normalize, augment) is applied at DataLoader collation time.

### Supported backbones (`model_key`)

| `model_key` | Backbone | Preprocessing | Train augmentation |
|---|---|---|---|
| `imagenet_cnn` | ResNet18, EfficientNet-B0, DenseNet121 | Letterbox → ImageNet norm | Flip, rotate ±15°, brightness |
| `clip` | CLIP ViT (`openai/clip-vit-base-patch32`) | HuggingFace `CLIPProcessor` | Horizontal flip |
| `dinov2` | DINOv2 Base (`facebook/dinov2-base`) | HuggingFace `AutoImageProcessor` | None |
| `dinov3` | DINOv2 Large (`facebook/dinov2-large`) | HuggingFace `AutoImageProcessor` | None |

### Creating DataLoaders

```python
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
    labels = batch["label"]     # (B, 1)       float32
    cases  = batch["case"]      # list[str] — patient IDs
    ids    = batch["image_id"]  # list[str] — image filenames
```

### Sanity-checking the pipeline

```bash
# ImageNet CNN (no HuggingFace download required)
uv run python scripts/sanity_dataloader.py --model_key imagenet_cnn \
  --split_file data/splits/splits.csv \
  --images_dir data/raw

# CLIP (downloads ~1 GB checkpoint on first run, cached afterward)
uv run python scripts/sanity_dataloader.py --model_key clip \
  --split_file data/splits/splits.csv \
  --images_dir data/raw

# DINOv2 (downloads ~300 MB checkpoint on first run, cached afterward)
uv run python scripts/sanity_dataloader.py --model_key dinov2 \
  --split_file data/splits/splits.csv \
  --images_dir data/raw
```

Expected output for each split:
```
[train]
  image  shape=(4, 3, 224, 224)  dtype=torch.float32  min=-2.118  max=2.640
  label  shape=(4, 1)  dtype=torch.float32  unique=[0.0, 1.0]
  case   type=list  len=4  example='42'
  id     type=list  len=4   example='bus_0042-l'
  ✓ assertions passed
```

## Setup
### 1) Clone repository
```bash
git clone https://github.com/HarryR8/irc-classification-project.git
cd irc-classification-project
```
### 2) Install `uv` package & project manager (one-time)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
### 3) Create virtual environment + install dependencies
```bash
uv sync
```

## Usage
### 1) Activate the venv
```bash
# macOS/Linux:
source .venv/bin/activate

# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1
```
### 2) Prepare data splits
```bash
uv run python -m busbra.data.prepare_data
```
### 3) Verify the pipeline
```bash
uv run python scripts/sanity_dataloader.py --model_key imagenet_cnn \
  --split_file data/splits/splits.csv \
  --images_dir data/raw
```

## Team
Zhuo Jin • Charlie Lam • Harry Reeve • Karolina Zvonickova (Advisor: James DesLauriers)

## Reference
Gómez-Flores et al. (2024). BUS-BRA: A Breast Ultrasound Dataset for Assessing CAD Systems. *Medical Physics*, 51(4), 3110-3123.

## License
MIT
