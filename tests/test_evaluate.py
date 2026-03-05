"""Tests for scripts/evaluate.py — helper functions and main() integration."""
import json
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Make the scripts directory importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import evaluate as eval_module


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_labels_probs():
    """Labels and probs for a perfect classifier."""
    labels = np.array([0, 0, 0, 1, 1, 1])
    probs  = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    return labels, probs


@pytest.fixture
def known_labels_probs():
    """Labels and probs with a known 2×2 confusion matrix at threshold=0.5.

    Applying threshold 0.5:  preds = [0, 0, 1, 0, 1, 1, 1]
    labels                          = [0, 0, 0, 1, 1, 1, 1]
    TN=2, FP=1, FN=1, TP=3  → 7 samples
    """
    labels = np.array([0, 0, 0, 1, 1, 1, 1])
    probs  = np.array([0.1, 0.2, 0.6, 0.3, 0.7, 0.8, 0.9])
    return labels, probs


@pytest.fixture
def dummy_loader():
    """Small DataLoader returning dicts (matches real loader contract)."""
    images = torch.randn(8, 3, 224, 224)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
    image_ids = [f"img_{i:03d}" for i in range(8)]

    class DictDataset(TensorDataset):
        def __init__(self, images, labels, image_ids):
            super().__init__(images, labels)
            self.image_ids = image_ids

        def __getitem__(self, index):
            img, label = super().__getitem__(index)
            return {"image": img, "label": label, "image_id": self.image_ids[index]}

    return DataLoader(DictDataset(images, labels, image_ids), batch_size=4)


@pytest.fixture
def tmp_run_dir(tmp_path):
    """Minimal run directory with config.json and a stub best.pt checkpoint."""
    config = {
        "model": "resnet18",
        "freeze_backbone": True,
        "head_type": "linear",
        "dropout": 0.3,
        "batch_size": 4,
        "num_workers": 0,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


# ─── Group 1: find_threshold_at_sensitivity ──────────────────────────────────

class TestFindThresholdAtSensitivity:

    def test_threshold_found_at_target(self):
        tpr        = np.array([0.0, 0.5, 0.9, 1.0])
        thresholds = np.array([1.0, 0.8, 0.6, 0.4])
        result = eval_module.find_threshold_at_sensitivity(tpr, thresholds, target=0.90)
        assert result == pytest.approx(0.6)

    def test_threshold_fallback_to_0_5_when_not_achievable(self):
        tpr        = np.array([0.0, 0.5, 0.8])
        thresholds = np.array([1.0, 0.8, 0.6])
        result = eval_module.find_threshold_at_sensitivity(tpr, thresholds, target=0.90)
        assert result == pytest.approx(0.5)

    def test_threshold_picks_first_valid_not_last(self):
        # Multiple thresholds achieve target; should return the first (most conservative)
        tpr        = np.array([0.0, 0.91, 0.95, 1.0])
        thresholds = np.array([1.0, 0.7,  0.5,  0.3])
        result = eval_module.find_threshold_at_sensitivity(tpr, thresholds, target=0.90)
        assert result == pytest.approx(0.7)

    def test_exact_target_counts_as_valid(self):
        tpr        = np.array([0.0, 0.90, 1.0])
        thresholds = np.array([1.0, 0.6,  0.3])
        result = eval_module.find_threshold_at_sensitivity(tpr, thresholds, target=0.90)
        assert result == pytest.approx(0.6)


# ─── Group 2: compute_metrics ────────────────────────────────────────────────

class TestComputeMetrics:

    def test_perfect_classifier(self, perfect_labels_probs):
        labels, probs = perfect_labels_probs
        m = eval_module.compute_metrics(labels, probs, threshold=0.5)
        assert m["accuracy"]    == pytest.approx(1.0)
        assert m["sensitivity"] == pytest.approx(1.0)
        assert m["specificity"] == pytest.approx(1.0)
        assert m["precision"]   == pytest.approx(1.0)
        assert m["f1_score"]    == pytest.approx(1.0)

    def test_known_values(self, known_labels_probs):
        labels, probs = known_labels_probs
        # At threshold=0.5: TP=3, TN=2, FP=1, FN=1
        m = eval_module.compute_metrics(labels, probs, threshold=0.5)
        assert m["tp"] == 3
        assert m["tn"] == 2
        assert m["fp"] == 1
        assert m["fn"] == 1
        assert m["accuracy"]    == pytest.approx(5 / 7)
        assert m["sensitivity"] == pytest.approx(3 / 4)   # tp/(tp+fn) = 3/4
        assert m["specificity"] == pytest.approx(2 / 3)   # tn/(tn+fp) = 2/3
        assert m["precision"]   == pytest.approx(3 / 4)   # tp/(tp+fp) = 3/4
        # F1 = 2*3 / (2*3+1+1) = 6/8 = 3/4
        assert m["f1_score"]    == pytest.approx(3 / 4)

    def test_all_predicted_negative_gives_zero_sensitivity(self):
        labels = np.array([0, 1, 0, 1])
        probs  = np.array([0.1, 0.2, 0.1, 0.2])
        m = eval_module.compute_metrics(labels, probs, threshold=0.5)
        assert m["tp"] == 0
        assert m["sensitivity"] == pytest.approx(0.0)
        assert m["f1_score"]    == pytest.approx(0.0)

    def test_all_predicted_positive_gives_zero_specificity(self):
        labels = np.array([0, 1, 0, 1])
        probs  = np.array([0.9, 0.9, 0.9, 0.9])
        m = eval_module.compute_metrics(labels, probs, threshold=0.5)
        assert m["tn"] == 0
        assert m["specificity"] == pytest.approx(0.0)

    def test_sensitivity_nan_when_no_positive_labels(self):
        labels = np.array([0, 0, 0, 0])
        probs  = np.array([0.1, 0.2, 0.3, 0.4])
        m = eval_module.compute_metrics(labels, probs, threshold=0.5)
        assert math.isnan(m["sensitivity"])

    def test_specificity_nan_when_no_negative_labels(self):
        labels = np.array([1, 1, 1, 1])
        probs  = np.array([0.6, 0.7, 0.8, 0.9])
        m = eval_module.compute_metrics(labels, probs, threshold=0.5)
        assert math.isnan(m["specificity"])

    def test_f1_nan_when_no_positive_predictions_or_labels(self):
        # All labels 0, all predicted 0 → tp=0, fp=0, fn=0 → denom=0
        labels = np.array([0, 0, 0, 0])
        probs  = np.array([0.1, 0.2, 0.1, 0.2])
        m = eval_module.compute_metrics(labels, probs, threshold=0.5)
        assert math.isnan(m["f1_score"])

    def test_confusion_matrix_values_are_int(self, known_labels_probs):
        labels, probs = known_labels_probs
        m = eval_module.compute_metrics(labels, probs, threshold=0.5)
        for key in ("tp", "tn", "fp", "fn"):
            assert isinstance(m[key], int), f"{key} should be int"


# ─── Group 3: main() integration ─────────────────────────────────────────────

def _make_fake_evaluate_fn(n=8):
    """Return a mock for busbra.training.evaluate that returns canned arrays."""
    rng = np.random.default_rng(0)
    labels = np.array([0, 1] * (n // 2))
    probs  = rng.uniform(0.1, 0.9, size=n)
    # Ensure both classes present so roc_auc_score doesn't raise
    probs[labels == 1] = np.clip(probs[labels == 1], 0.5, 0.95)
    probs[labels == 0] = np.clip(probs[labels == 0], 0.05, 0.49)
    return MagicMock(return_value={"labels": labels, "probs": probs, "loss": 0.4, "auc": 0.85})


def _make_fake_model():
    """Tiny linear model whose state_dict can be loaded."""
    return nn.Linear(10, 2)


class TestMainIntegration:

    def _run_main(self, monkeypatch, tmp_run_dir, dummy_loader, split="test"):
        """Patch all I/O-heavy dependencies and call main()."""
        fake_model = _make_fake_model()
        fake_ckpt  = {
            "model_state_dict": fake_model.state_dict(),
            "epoch": 5,
            "val_auc": 0.88,
        }

        monkeypatch.setattr("sys.argv", ["evaluate.py", "--run_dir", str(tmp_run_dir),
                                         "--split", split])
        with (
            patch.object(eval_module, "create_dataloaders",
                         return_value=(dummy_loader, dummy_loader, dummy_loader)),
            patch.object(eval_module, "create_model", return_value=fake_model),
            patch.object(eval_module, "evaluate", _make_fake_evaluate_fn()),
            patch("torch.load", return_value=fake_ckpt),
            patch.object(fake_model, "load_state_dict"),
        ):
            eval_module.main()

    def test_main_creates_eval_json(self, monkeypatch, tmp_run_dir, dummy_loader):
        self._run_main(monkeypatch, tmp_run_dir, dummy_loader, split="test")
        out = tmp_run_dir / "eval_test.json"
        assert out.exists(), "eval_test.json should be created"

    def test_main_val_split_writes_correct_split_key(self, monkeypatch, tmp_run_dir, dummy_loader):
        self._run_main(monkeypatch, tmp_run_dir, dummy_loader, split="val")
        data = json.loads((tmp_run_dir / "eval_val.json").read_text())
        assert data["split"] == "val"

    def test_main_json_has_required_keys(self, monkeypatch, tmp_run_dir, dummy_loader):
        self._run_main(monkeypatch, tmp_run_dir, dummy_loader, split="test")
        data = json.loads((tmp_run_dir / "eval_test.json").read_text())
        for key in ("split", "checkpoint_epoch", "auc_roc", "accuracy",
                    "sensitivity", "specificity", "confusion_matrix", "n_samples"):
            assert key in data, f"Missing key: {key}"

    def test_main_json_types_are_correct(self, monkeypatch, tmp_run_dir, dummy_loader):
        self._run_main(monkeypatch, tmp_run_dir, dummy_loader, split="test")
        data = json.loads((tmp_run_dir / "eval_test.json").read_text())
        assert isinstance(data["auc_roc"],    float)
        assert isinstance(data["accuracy"],   float)
        assert isinstance(data["sensitivity"], float)
        assert isinstance(data["specificity"], float)
        assert isinstance(data["checkpoint_epoch"], int)
        cm = data["confusion_matrix"]
        for key in ("tn", "fp", "fn", "tp"):
            assert isinstance(cm[key], int), f"confusion_matrix.{key} should be int"
