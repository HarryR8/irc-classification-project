# IRC-Classification-Project (Team 6)
>**Classification of benign/malignant tumours from medical images (breast ultrasound; optional MS MRI).**

## Goal
Build a **reproducible** ML pipeline to classify **benign vs malignant** tumours, with honest evaluation (patient-level splits where applicable).

## Datasets
  - **BUS-BRA (primary)** - 1875 images / 1064 patients, BI-RADS + tumour delineations. **CC BY 4.0**  
  Links: https://github.com/wgomezf/BUS-BRA | https://zenodo.org/records/8231412  
- **BUS-UCLM (optional / cross-dataset)** - 683 images / 38 patients, normal/benign/malignant + masks. **CC BY 4.0**  
  Links: https://github.com/noeliavallez/BUS-UCLM-Dataset
- **MS MRI lesion segmentation (optional)** - T1/T2/FLAIR + lesion masks (60 patients). **CC BY 4.0**  
  Link: https://www.kaggle.com/datasets/orvile/multiple-sclerosis-brain-mri-lesion-segmentation

## Approach (preliminary)
- Baseline: transfer learning CNN (e.g., ResNet/EfficientNet)
- Mask-aware option: ROI-cropped classification (ROI-only vs ROI+context)
- Evaluation: ROC-AUC/PR-AUC/F1 + confusion matrix (focus on malignant recall)

## Repo structure 
```
bus-bra-classifier/
├── pyproject.toml
├── .gitignore
├── README.md
├── src/
│   ├── prepare_data.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
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
### 4) (Optional) Activate the venv
```bash 
# macOS/Linux:
source .venv/bin/activate

# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1
```

## Running (soon)


## Team
Zhuo Jin • Charlie Lam • Harry Reeve • Karolina Zvonickova (Advisor: Jay DesLauriers)

## License
MIT

