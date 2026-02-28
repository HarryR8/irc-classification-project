#!/usr/bin/env python
"""Test checkpoint loading and inference."""
import torch
from busbra.models.factory import create_model

print("=" * 60)
print("CHECKPOINT LOADING TEST")
print("=" * 60)

# Test 1: Load full fine-tuning checkpoint
print("\n=== Test 1: Load Full Fine-tuning Checkpoint ===")
checkpoint_path = "/tmp/test_training/full_finetune/best.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
print(f"Checkpoint keys: {list(checkpoint.keys())}")
print(f"Epoch: {checkpoint['epoch']}")
print(f"Val AUC: {checkpoint['val_auc']:.4f}")

# Recreate model and load state
model = create_model("resnet18", num_classes=2, freeze_backbone=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("✓ Model state loaded successfully")

# Test forward pass
dummy_input = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    output = model(dummy_input)
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Output (logits sample): {output[0]}")
assert output.shape == (2, 2), "Output shape should be (batch_size, num_classes)"
print("✓ Forward pass successful")

# Test 2: Load frozen backbone checkpoint
print("\n=== Test 2: Load Frozen Backbone Checkpoint ===")
checkpoint_path = "/tmp/test_training/frozen_backbone/best.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
print(f"Epoch: {checkpoint['epoch']}")
print(f"Val AUC: {checkpoint['val_auc']:.4f}")

# Recreate model with same architecture
model = create_model("resnet18", num_classes=2, freeze_backbone=True, head_type="mlp")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("✓ Model state loaded successfully")

# Test forward pass
with torch.no_grad():
    output = model(dummy_input)
print(f"Output shape: {output.shape}")
assert output.shape == (2, 2), "Output shape should be (batch_size, num_classes)"
print("✓ Forward pass successful")

# Test 3: Verify softmax probabilities
print("\n=== Test 3: Verify Softmax Probabilities ===")
with torch.no_grad():
    logits = model(dummy_input)
    probs = torch.softmax(logits, dim=1)
print(f"Logits: {logits[0]}")
print(f"Probabilities: {probs[0]}")
print(f"Sum of probabilities: {probs[0].sum():.4f}")
assert torch.allclose(probs[0].sum(), torch.tensor(1.0)), "Probabilities should sum to 1"
assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities should be in [0, 1]"
print("✓ Probabilities valid")

# Test 4: Check malignant probability extraction
print("\n=== Test 4: Extract Malignant Probabilities ===")
malignant_probs = probs[:, 1].cpu().numpy()
print(f"Malignant probabilities for batch: {malignant_probs}")
assert len(malignant_probs) == 2, "Should have one probability per sample"
assert (malignant_probs >= 0).all() and (malignant_probs <= 1).all(), "Probs in [0, 1]"
print("✓ Malignant probability extraction successful")

print("\n" + "=" * 60)
print("ALL CHECKPOINT LOADING TESTS PASSED ✓")
print("=" * 60)
