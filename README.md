# IRC-Classification-Project (Team 6)
>**Classification of benign/malignant tumours from medical images (breast ultrasound; optional MS MRI).**

## Goal
Build a **reproducible** ML pipeline to classify **benign vs malignant** tumours, with honest evaluation (patient-level splits where applicable).

## Datasets
  - **BUS-BRA (primary)** - 1875 images / 1064 patients, BI-RADS + tumour delineations. **CC BY 4.0**  
  Links: https://github.com/wgomezf/BUS-BRA | https://zenodo.org/records/8231412  
- **BUS-UCLM (extension: cross-dataset generalisation)** - 683 images / 38 patients, normal/benign/malignant + masks. **CC BY 4.0**  
  Links: https://github.com/noeliavallez/BUS-UCLM-Dataset

## Approach (preliminary)
- Baseline: transfer learning CNN (e.g., ResNet/EfficientNet)
- Mask-aware (extension): ROI-cropped classification (ROI-only vs ROI+context)
- Evaluation: ROC-AUC/PR-AUC/F1 + confusion matrix (focus on malignant recall)

## Repo structure 
```
irc-classification-project/
├── pyproject.toml          # Dependencies + uv config
├── .gitignore              # Data, checkpoints, envs excluded
├── README.md               # Setup instructions, usage guide
├── src/busbra/
│   ├── train.py            # Training loop with early stopping (empty)
│   ├── data/
│   │   ├── prepare_data.py # Load CSVs, create patient-level splits
│   │   ├── dataset.py      # PyTorch Dataset + dataloaders 
│   │   ├── splitting.py    # Patient-level splits (leakage prevention) 
│   │   └── transforms.py   # Albumentations augmentations 
│   ├── models/
│   │   └── model.py        # CNN model (empty)
│   └── evaluation/
│       └── evaluate.py     # (empty)
└── data/
    ├── raw/            ← put dataset (BUS-BRA) here
    │   ├── bus_data.csv
    │   ├── bus_0001-l.png
    │   └── ...
    └── splits/         ← created by prepare_data.py
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

## Team
Zhuo Jin • Charlie Lam • Harry Reeve • Karolina Zvonickova (Advisor: Jay DesLauriers)

## Reference
Gómez-Flores et al. (2024). BUS-BRA: A Breast Ultrasound Dataset for Assessing CAD Systems. *Medical Physics*, 51(4), 3110-3123.

## License
MIT

