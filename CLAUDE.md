# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All commands must be run with `uv run` from the repo root. The package is installed as `busbra` (source in `src/`).

```bash
uv sync                   # install/sync dependencies
uv sync --extra dev       # include pytest + jupyter
```

## Common Commands

```bash
# Prepare patient-level splits (run once after placing data in data/raw/)
uv run python -m busbra.data.prepare_data

# Verify the data pipeline for a given backbone
uv run python scripts/sanity_dataloader.py --model_key imagenet_cnn \
  --split_file data/splits/splits.csv --images_dir data/raw

# Train (DO NOT run locally without explicit instruction — intended for remote/HPC)
uv run python scripts/train.py --model resnet18 --epochs 30 --batch_size 32

# Evaluate a saved run
uv run python scripts/evaluate.py --run_dir runs/<model>_<timestamp>
uv run python scripts/evaluate.py --run_dir runs/<model>_<timestamp> --split val

# Run tests
uv run pytest
uv run pytest tests/test_specific.py::test_name   # single test
```

## Architecture

The pipeline is intentionally **model-agnostic at the dataset level**:

1. `BUSBRADataset` (`src/busbra/data/dataset.py`) — returns raw `PIL.Image` + metadata dict. No transforms stored inside the dataset.
2. `get_preprocess(model_key, split)` (`src/busbra/data/preprocessing.py`) — returns a callable that converts PIL → float32 tensor. The `split` argument controls train augmentation vs. deterministic val/test transforms.
3. `make_collate_fn(preprocess_fn)` (`src/busbra/data/loaders.py`) — wraps the preprocessor inside the DataLoader collate function so transforms run in worker processes.
4. `create_dataloaders(...)` (`src/busbra/data/loaders.py`) — factory that wires dataset + preprocessor + weighted sampler (class-imbalance compensation) into three DataLoaders. Batch dict keys: `"image"` (B,3,H,W), `"label"` (B,) int64, `"case"` list[str], `"image_id"` list[str].

Model creation goes through `src/busbra/models/factory.py`:
- `create_model(name, freeze_backbone, head_type)` — returns a full classifier. timm models with `freeze_backbone=False` use the native timm head; all DINO/CLIP models and any frozen-backbone timm model use `BackboneWithHead`.
- `create_backbone(name)` — returns `(backbone, embed_dim)` for pre-computing embeddings offline.
- `get_preprocess_key(model_name)` — links a model name to the correct preprocessing key for the DataLoader.
- `BackboneWithHead` wraps any backbone + head; `get_features(x)` returns raw embeddings.
- CLIP uses `open_clip` (not HuggingFace). Requires the `clip` optional extra: `uv sync --extra clip`.

Training (`src/busbra/training/trainer.py`) uses `train_one_epoch` / `evaluate`, both returning loss and AUC. The `evaluate` function also returns raw labels, probabilities, and image IDs for threshold analysis.

`scripts/train.py` applies per-model recommended hyperparameters from `MODEL_TRAINING_CONFIGS` (lr, weight_decay, warmup_epochs, freeze_backbone) that CLI flags override. Outputs go to `runs/<model>_<timestamp>/`: `config.json`, `history.json`, `best.pt` (best val AUC), `last.pt`.

`scripts/evaluate.py` loads a run directory, runs inference, and writes `eval_<split>.json`, `eval_<split>_roc_curve.png`, and `eval_<split>_threshold_sweep.csv`. Key options: `--split` (val/test), `--threshold_split` (which split to derive optimal thresholds from), `--masks_dir` (enables lesion-crop preprocessing). The threshold candidates (Youden J, max F1, sensitivity >= 0.95, specificity >= 0.90) come from `busbra.training.metrics.find_optimal_thresholds` and can be used standalone.

## Key Invariants

- Labels are `(B,)` int64 — correct for `nn.CrossEntropyLoss`. Never `(B, 1)`.
- All DINO models load via `AutoModel.from_pretrained(hf_name)` — no torch.hub.
- DINOv3 uses `pooler_output`; DINOv2 uses CLS token `last_hidden_state[:, 0, :]`.
- albumentations 2.x: use `fill=0` (not `value=` or `fill_value=`) in `PadIfNeeded`/`Rotate`.
- Loss uses fixed class weights `[0.32, 0.68]` (benign/malignant) to upweight the minority class. Evaluation must use the same weights.
- `pin_memory=True` produces a harmless warning on MPS (Apple Silicon) — expected.
- Data splits are patient-level stratified: train 1316 / val 285 / test 274 images.
