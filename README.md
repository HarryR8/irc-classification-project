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
│   ├── train.py                          # ✅ CLI training entrypoint
│   ├── evaluate.py                       # ✅ Evaluate a checkpoint
│   ├── search.py                         # ✅ Grid search over hyperparameters
│   ├── sanity_dataloader.py              # ✅ Verify batch shapes/dtypes
│   ├── train_efficientnet_hpc.pbs        # HPC job: train single EfficientNet-B0 config
│   ├── evaluate_efficientnet_hpc.pbs     # HPC job: evaluate best EfficientNet-B0 config
│   ├── search_densenet121_hpc.pbs        # HPC job: DenseNet-121 grid search (18 configs)
│   ├── search_resnet18_hpc.pbs           # HPC job: ResNet-18 grid search (18 configs)
│   ├── search_resnet50_hpc.pbs           # HPC job: ResNet-50 grid search (18 configs)
│   ├── search_dinov3_large_hpc_part1.pbs # HPC job: DINOv3-Large grid search (configs 1–6)
│   ├── search_dinov3_large_hpc_part2.pbs # HPC job: DINOv3-Large grid search (configs 7–12)
│   ├── search_dinov3_large_hpc_part3.pbs # HPC job: DINOv3-Large grid search (configs 13–18)
│   └── train_all.sh                      # Convenience shell script: train all models
├── src/busbra/
│   ├── data/
│   │   ├── prepare_data.py   # ✅ Load CSVs, create patient-level splits
│   │   ├── dataset.py        # ✅ Model-agnostic PyTorch Dataset (returns PIL.Image)
│   │   ├── preprocessing.py  # ✅ Backbone-specific preprocessing registry
│   │   └── loaders.py        # ✅ Collate functions + DataLoader factory
│   ├── models/
│   │   ├── factory.py        # ✅ Model registry + create_model / create_backbone
│   │   ├── heads.py          # ✅ Classification head architectures (linear, mlp, mlp_deep)
│   │   └── __init__.py       # ✅ Public exports
│   └── training/
│       ├── trainer.py        # ✅ train_one_epoch + evaluate functions
│       ├── metrics.py        # ✅ Pure metrics library (metrics_at_threshold, sweep_thresholds, find_optimal_thresholds)
│       └── __init__.py       # ✅ Public exports
└── data/
    ├── raw/            ← 🚨 put dataset (BUS-BRA) here
    │   ├── bus_data.csv
    │   ├── bus_0001-l.png
    │   └── ...
    └── splits/         ← ✅ created by prepare_data.py
        ├── patient_splits.csv  # patient ID + split assignment (for auditing)
        ├── split_info.json     # split metadata
        └── splits.csv          # ID + Case + label + filename + split
```

## Data pipeline

The data pipeline is **model-agnostic**: the `BUSBRADataset` returns raw `PIL.Image` objects and metadata, and all backbone-specific preprocessing (resize, normalize, augment) is applied at DataLoader collation time.

### Supported backbones (`model_key`)

| `model_key` | Backbone family | Preprocessing | Train augmentation |
|---|---|---|---|
| `imagenet_cnn` | ResNet, EfficientNet, DenseNet | Letterbox → 224px → ImageNet norm | H/V flip, rotate ±30°, brightness/contrast, Gaussian blur, elastic & grid distortion |
| `clip` | CLIP ViT-B/32 | HuggingFace `CLIPProcessor` | Horizontal flip |
| `dinov2` | DINOv2 (small/base/large, ±registers) | HuggingFace `AutoImageProcessor` | None |
| `dinov3` | DINOv3 (small/base/large) | HuggingFace `AutoImageProcessor` | None |

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
    labels = batch["label"]     # (B,)          int64
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
  label  shape=(4,)  dtype=torch.int64  unique=[0, 1]
  case   type=list  len=4  example='42'
  id     type=list  len=4   example='bus_0042-l'
  ✓ assertions passed
```

## Model factory

The model factory (`busbra.models`) provides a consistent interface for creating classifiers, with support for three transfer learning modes.

### Model registry

| Model name | Backbone | `preprocess_key` | Embed dim | Notes |
|---|---|---|---|---|
| `resnet18` | ResNet-18 | `imagenet_cnn` | 512 | |
| `resnet50` | ResNet-50 | `imagenet_cnn` | 2048 | |
| `efficientnet_b0` | EfficientNet-B0 | `imagenet_cnn` | 1280 | |
| `densenet121` | DenseNet-121 | `imagenet_cnn` | 1024 | |
| `dinov2_small` | DINOv2 ViT-S/14 | `dinov2` | 384 | |
| `dinov2_base` | DINOv2 ViT-B/14 | `dinov2` | 768 | |
| `dinov2_large` | DINOv2 ViT-L/14 | `dinov2` | 1024 | |
| `dinov2_base_reg` | DINOv2 ViT-B/14 + registers | `dinov2` | 768 | |
| `dinov2_large_reg` | DINOv2 ViT-L/14 + registers | `dinov2` | 1024 | |
| `dinov3_small` | DINOv3 ViT-S/16 | `dinov3` | 384 | |
| `dinov3_base` | DINOv3 ViT-B/16 | `dinov3` | 768 | |
| `dinov3_large` | DINOv3 ViT-L/16 | `dinov3` | 1024 | |
| `clip_vit_base` | CLIP ViT-B/32 | `clip` | 512 | Requires `uv sync --extra clip` |

### Usage modes

```python
from busbra.models import create_model, create_backbone, get_preprocess_key, count_parameters

# 1. Full fine-tuning — all ~11 M params trainable
model = create_model("resnet18", num_classes=2, pretrained=True)

# 2. Frozen backbone + custom head — only head params trainable
model = create_model(
    "resnet18",
    freeze_backbone=True,
    head_type="mlp",        # "linear" | "mlp" | "mlp_deep"
    head_hidden_dim=256,
    head_dropout=0.3,
)
print(count_parameters(model))
# {'total': 11308354, 'trainable': 131842, 'frozen': 11176512}

# 3. Backbone only — for pre-computing and caching embeddings
backbone, embed_dim = create_backbone("resnet18", pretrained=True)
backbone.eval()
with torch.no_grad():
    features = backbone(images)  # (B, 512)

# Link model name → preprocessing key for DataLoader
preprocess_key = get_preprocess_key("resnet18")  # "imagenet_cnn"
```

### Classification heads

| `head_type` | Architecture | Trainable params (resnet18 backbone) |
|---|---|---|
| `linear` | `Linear(512 → 2)` | 1,026 |
| `mlp` | `Linear → ReLU → Dropout → Linear` | 131,842 |
| `mlp_deep` | `Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear` | 164,482 |

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
### 4) Train a model
```bash
uv run python scripts/train.py --model resnet18 --epochs 30 --batch_size 32 --lr 1e-4
```

Key CLI arguments for `train.py`:

| Argument | Default | Description |
|---|---|---|
| `--model` | `resnet18` | Model name (see registry table) |
| `--epochs` | 30 | Number of training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--weight_decay` | `1e-4` | L2 regularisation |
| `--dropout` | `0.3` | Dropout rate for MLP head |
| `--freeze_backbone` | off | Freeze backbone weights, train head only |
| `--head_type` | `linear` | Head architecture (`linear`, `mlp`, `mlp_deep`) |
| `--split_file` | `data/splits/splits.csv` | Path to splits CSV |
| `--images_dir` | `data/raw` | Directory containing image files |
| `--patience` | 0 | Early stopping patience; 0 disables |
| `--output_dir` | `runs/<model>_<timestamp>` | Custom output directory |

### 5) Evaluate a trained model
```bash
# Evaluate on test split (default)
uv run python scripts/evaluate.py --run_dir runs/<model>_<timestamp>

# Evaluate on validation split
uv run python scripts/evaluate.py --run_dir runs/<model>_<timestamp> --split val

# Evaluate with lesion-crop preprocessing
uv run python scripts/evaluate.py --run_dir runs/<model>_<timestamp> --masks_dir data/masks
```

Key CLI arguments for `evaluate.py`:

| Argument | Default | Description |
|---|---|---|
| `--run_dir` | required | Run directory containing `config.json` and `best.pt` |
| `--split` | `test` | Split to evaluate: `val` or `test` |
| `--threshold_split` | `val` | Split used to select optimal thresholds (`val`, `test`, or `same`) |
| `--masks_dir` | `None` | Mask directory for lesion-crop preprocessing |
| `--num_thresholds` | `201` | Number of threshold grid points swept in [0, 1] |

Three files are written to `--run_dir`:

| Output file | Contents |
|---|---|
| `eval_<split>.json` | AUC, per-metric results at threshold 0.5, four optimal threshold candidates |
| `eval_<split>_roc_curve.png` | Publication-ready ROC curve at 300 dpi |
| `eval_<split>_threshold_sweep.csv` | Full per-threshold metrics table (`num_thresholds` rows) |

Four clinically motivated threshold candidates are reported (selected from the sweep):

| Criterion | Strategy |
|---|---|
| ROC Youden J | max(TPR − FPR) |
| Max F1 | maximise F1 score |
| Sensitivity ≥ 0.95 | highest specificity subject to sensitivity ≥ 0.95 |
| Specificity ≥ 0.90 | highest sensitivity subject to specificity ≥ 0.90 |

### 6) Grid search

`search.py` sweeps all combinations of `lr`, `weight_decay`, and `batch_size` from a fixed grid for one or more models.

```bash
# Quick local test (5 epochs, all configs for resnet18)
uv run python scripts/search.py --models resnet18 --epochs 5 --head_type mlp

# Large model: split into 3 partitions (as done in HPC PBS scripts)
# Run partition 1 of 3 (6 configs out of 18)
uv run python scripts/search.py --models dinov3_large --epochs 100 \
    --head_type mlp --dropout 0.5 --patience 10 --freeze_backbone \
    --part 1 --num_parts 3 --output_dir runs/search_dinov3_large
```

Results are saved to `<output_dir>/grid_search_results.json`. HPC PBS scripts for each model are in `scripts/`.

Key CLI arguments for `search.py`:

| Argument | Default | Description |
|---|---|---|
| `--models` | `efficientnet_b0 dinov3_base` | One or more model names |
| `--epochs` | 10 | Epochs per config |
| `--head_type` | `linear` | Head type (`linear`, `mlp`, `mlp_deep`) |
| `--dropout` | 0.3 | Dropout for MLP head |
| `--patience` | 0 | Early stopping (0 = disabled) |
| `--freeze_backbone` | off | Freeze backbone, train head only |
| `--part` | 1 | Partition index (1-indexed) |
| `--num_parts` | 1 | Total partitions (1 = no split) |
| `--output_dir` | `runs/search` | Base directory for outputs |
| `--images_dir` | None | Path to images (required on HPC) |

The metrics library (`busbra.training.metrics`) can also be used standalone:

```python
from busbra.training.metrics import metrics_at_threshold, find_optimal_thresholds
import numpy as np

m = metrics_at_threshold(y_true, y_score, threshold=0.5)
# {"sensitivity": ..., "specificity": ..., "precision": ..., "npv": ..., "f1": ..., "accuracy": ..., "tp": ..., ...}

result = find_optimal_thresholds(y_true, y_score)
print(result["by_roc_youden"])   # best Youden J threshold
```

## Team
Zhuo Jin • Charlie Lam • Harry Reeve • Karolina Zvonickova (Advisor: Jay DesLauriers)

## Reference
Gómez-Flores et al. (2024). BUS-BRA: A Breast Ultrasound Dataset for Assessing CAD Systems. *Medical Physics*, 51(4), 3110-3123.

## License
MIT
