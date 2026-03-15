# _SonoNet_ - IRC Classification Project
**Classification of benign/malignant tumours from medical images (breast ultrasound).**

## Goal
Build a **reproducible** ML pipeline to classify **benign vs malignant** tumours, with honest evaluation (patient-level splits where applicable).

## Introduction
Breast cancer is the most common cancer in women globally and the 2nd most
prevalent cancer overall worldwide. Early detection of breast cancer with
ultrasound is critical as it identifies small, non-palpable tumours that might be
missed by X-ray mammography, which can significantly improve clinical
outcomes. The 5-year survival rate of Stage 1 breast cancer is approximately
99%, which severely drops to only about 30% in Stage 4 metastatic breast cancer,
highlighting the pressing need for early breast tumour detection. The use of deep
learning models such as Bre-Net (implemented transfer learning with
CNNs) has previously demonstrated an improvement in the diagnostic
performance of radiologists in the identification of malignant breast
carcinomas from ultrasonography images. Therefore, it would be worthwhile to
investigate whether an alternative deep learning model could achieve a similar
diagnostic accuracy on an entirely novel, independent dataset.  

## Rationale
Ultrasound serves as a superior breast cancer imaging modality in younger women
and women with greater breast density, and it does not subject patients to ionising
radiation or intravenous contrast, unlike X-ray mammography. Breast Imaging
Reporting and Data System (BI-RADS) represents the current gold standard as a
risk assessment tool and universal reporting system for breast cancer screening,
including ultrasound, mammography and MRI. According to current clinical
guidelines, any patient that has been assigned a BI-RADS category of 4A or above
(i.e. has a likelihood of malignancy as low as >2% to 10%), might be subjected to a
biopsy. However, only 9–11% of biopsies performed following ultrasound screening
are malignant, increasing healthcare expenses and patient anxiety. Therefore,
alternative non-invasive approaches for malignant breast tumour detection are
necessary. 

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
│   ├── train.py              # ✅ CLI training entrypoint (model, epochs, lr, etc.)
│   ├── evaluate.py           # ✅ Evaluate a checkpoint (AUC, accuracy, sensitivity, specificity)
│   └── sanity_dataloader.py  # ✅ Verify batch shapes/dtypes for any backbone
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

| `model_key` | Backbone | Preprocessing | Train augmentation |
|---|---|---|---|
| `imagenet_cnn` | ResNet18, EfficientNet-B0, DenseNet121 | Letterbox → ImageNet norm | H/V flip, rotate ±30°, brightness/contrast, Gaussian blur, elastic & grid distortion |
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

| Model name | Backbone | `preprocess_key` | Embed dim | Status |
|---|---|---|---|---|
| `resnet18` | ResNet-18 | `imagenet_cnn` | 512 | ✅ |
| `resnet50` | ResNet-50 | `imagenet_cnn` | 2048 | ✅ |
| `efficientnet_b0` | EfficientNet-B0 | `imagenet_cnn` | 1280 | ✅ |
| `densenet121` | DenseNet-121 | `imagenet_cnn` | 1024 | ✅ |
| `dinov2_base` | DINOv2 Base | `dinov2` | 768 | ✅ |
| `clip_vit_base` | CLIP ViT-B/32 | `clip` | 512 | ✅ |

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
