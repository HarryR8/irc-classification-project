# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All commands must be run with `uv run` from the repo root. The package is installed as `busbra` (source in `src/`).

```bash
uv sync                   # install/sync dependencies
uv sync --extra dev       # include pytest + jupyter
uv sync --extra clip      # also install open_clip (required for clip_vit_base model)
```

**DINOv3 models require a HuggingFace token** (gated checkpoints). Set `HF_TOKEN` as an env var, or write it to `~/.hf_token` (PBS scripts load it automatically).

## Common Commands

```bash
# Prepare patient-level splits (run once after placing data in data/raw/)
uv run python -m busbra.data.prepare_data

# Verify the data pipeline for a given backbone
uv run python scripts/sanity_dataloader.py --model_key imagenet_cnn \
  --split_file data/splits/splits.csv --images_dir data/raw

# Train (DO NOT run locally without explicit instruction — intended for remote/HPC)
uv run python scripts/train.py --model resnet18 --epochs 30 --batch_size 32
# Key train.py flags: --freeze_backbone, --head_type {linear,mlp,mlp_deep}, --patience N (early stop),
#   --dropout F (head dropout, default 0.3 frozen / 0.5 unfrozen in HPC scripts),
#   --backbone_lr_scale F (e.g. 0.02 → backbone LR 2% of head LR, for BackboneWithHead models only)

# Evaluate a saved run
uv run python scripts/evaluate.py --run_dir runs/<model>_<timestamp>
uv run python scripts/evaluate.py --run_dir runs/<model>_<timestamp> --split val

# Grid search over lr/weight_decay/batch_size (DO NOT run locally — intended for HPC)
uv run python scripts/search.py --models resnet18 --epochs 5 --head_type mlp

# Ensemble inference across multiple checkpoints
uv run python scripts/ensemble_eval.py \
  --run_dirs runs/<model_a>_<timestamp> runs/<model_b>_<timestamp>

# Aggregate all evaluated runs into results/summary.csv + ROC/CM/training-curve files
uv run python scripts/collect_results.py
uv run python scripts/collect_results.py --poster_only          # 6 poster-key runs only
uv run python scripts/collect_results.py --runs runs/resnet18_* # specific runs

# Plot per-epoch ROC curves from epoch_test_preds.npz (written by train.py each epoch)
uv run python scripts/plot_epoch_roc.py --run_dir runs/<model>_<timestamp>
uv run python scripts/plot_epoch_roc.py \
  --run_dirs runs/<model_a>_<timestamp> runs/<model_b>_<timestamp>  # comparison plot

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

**Per-model training defaults** (CLI flags always win; `--freeze_backbone` is `store_true` so must be passed explicitly even when the model default is `True`):

| Model family         | lr   | weight_decay | warmup_epochs | freeze_backbone default |
|----------------------|------|--------------|---------------|------------------------|
| resnet18/50, efficientnet_b0, densenet121 | 1e-4 | 1e-5 | 0 | False (full fine-tune) |
| dinov2_*, dinov3_*, clip_vit_base | 1e-5 | 1e-2 | 5 | True (frozen backbone) |

Optimizer: **AdamW**. Scheduler: **cosine annealing** (`CosineAnnealingLR`); ViT/DINO/CLIP also get a 5-epoch linear warmup prepended via `SequentialLR`.

`scripts/evaluate.py` loads a run directory, runs inference, and writes `eval_<split>.json`, `eval_<split>_roc_curve.png`, and `eval_<split>_threshold_sweep.csv`. Key options: `--split` (val/test), `--threshold_split` (which split to derive optimal thresholds from), `--masks_dir` (enables lesion-crop preprocessing). The threshold candidates (Youden J, max F1, sensitivity >= 0.95, specificity >= 0.90) come from `busbra.training.metrics.find_optimal_thresholds` and can be used standalone.

`scripts/ensemble_eval.py` averages softmax probabilities from multiple checkpoints and writes combined AUC and threshold metrics to `runs/ensemble_<timestamp>/`. `scripts/search.py` sweeps lr/weight_decay/batch_size grids; supports `--part`/`--num_parts` for parallelising across HPC nodes. HPC PBS scripts live in `scripts/hpc/`.

The full comparison matrix uses three training conditions:
- **A** — `--freeze_backbone`, no masks (head-only training)
- **B** — full fine-tune, no masks
- **C** — full fine-tune, `--masks_dir` (lesion-crop preprocessing)

HPC results are synced to `results_bundle/` (treated as equally authoritative as `runs/`). `scripts/hpc/collect_results.sh` aggregates evaluation outputs.

## Key Invariants

- Labels are `(B,)` int64 — correct for `nn.CrossEntropyLoss`. Never `(B, 1)`.
- All DINO models load via `AutoModel.from_pretrained(hf_name)` — no torch.hub.
- DINOv3 uses `pooler_output` (via `_DINOv3Wrapper`); DINOv2 uses CLS token `last_hidden_state[:, 0, :]`.
- albumentations 2.x: use `fill=0` (not `value=` or `fill_value=`) in `PadIfNeeded`/`Rotate`.
- Loss uses fixed class weights `[0.32, 0.68]` (benign/malignant) to upweight the minority class. Evaluation must use the same weights.
- `pin_memory=True` produces a harmless warning on MPS (Apple Silicon) — expected.
- Data splits are patient-level stratified: train 1316 / val 285 / test 274 images.
- `--freeze_backbone` in `train.py` is `store_true` (default `False`); the per-model `MODEL_TRAINING_CONFIGS` entry for `freeze_backbone` is **not** automatically applied via CLI — pass `--freeze_backbone` explicitly for DINO/CLIP frozen training.

## Extension Points

**New backbone** — add an entry to `MODEL_REGISTRY` in `src/busbra/models/factory.py` (keys: `type`, `timm_name`/`hf_name`, `embedding_dim`, `preprocess_key`), add a builder branch to `create_backbone()` if the type is new, and add per-model hyperparameter defaults to `MODEL_TRAINING_CONFIGS` in `scripts/train.py`.

**New head** — define an `nn.Module` in `src/busbra/models/heads.py` and add a branch to `create_head()` keyed by the `head_type` string. Pass `--head_type <name>` at the CLI.

**New preprocessing** — write a factory function in `src/busbra/data/preprocessing.py` and add a branch to `get_preprocess()`. Reference the new key in the model's `preprocess_key` registry entry.
