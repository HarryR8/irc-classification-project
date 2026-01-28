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

## Repo structure (soon)


## Install (soon)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running (soon)


## Team
Zhuo Jin • Charlie Lam • Harry Reeve • Karolina Zvonickova (Advisor: Jay DesLauriers)

## Lisence
MIT

