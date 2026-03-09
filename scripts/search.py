"""
Grid search functionality for hyperparameter tuning of BUS-BRA classifier.

This module can be imported into train.py or run as a standalone script.

Example usage:
    # As standalone script
    python scripts/search.py --models efficientnet_b0 dinov3_base --epochs 10

    # Import into train.py (modify train.py to use search functionality)
    from scripts.search import run_grid_search
    results = run_grid_search(models=['efficientnet_b0'], param_grid=param_grid, epochs=10)
"""

import argparse
import json
import os
import subprocess
import sys
from itertools import product
from pathlib import Path


def get_default_param_grids():
    """Get default parameter grids for grid search based on model type."""
    return {
        "efficientnet_b0": {
            "lr": [1e-4, 5e-4, 1e-3],
            "weight_decay": [1e-5, 1e-4, 1e-3],
            "batch_size": [16, 32],
        },
        "dinov3_base": {
            "lr": [1e-5, 5e-5, 1e-4],
            "weight_decay": [1e-2, 5e-3, 1e-3],
            "batch_size": [16, 32],
        }
    }


def run_single_training(model, params, epochs, output_base_dir="runs/search", seed=42):
    """Run a single training run with given parameters."""
    # Create unique output directory
    param_str = "_".join([f"{k}={v}" for k, v in params.items()])
    output_dir = os.path.join(output_base_dir, f"{model}_{param_str}")

    # Build command
    cmd = [
        "uv", "run", "python", "scripts/train.py",
        "--model", model,
        "--epochs", str(epochs),
        "--batch_size", str(params["batch_size"]),
        "--lr", str(params["lr"]),
        "--weight_decay", str(params["weight_decay"]),
        "--output_dir", output_dir,
        "--seed", str(seed)
    ]

    print(f"Running: {' '.join(cmd)}")

    # Run training
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        success = result.returncode == 0

        # Extract best val AUC from output
        best_auc = None
        for line in result.stdout.split('\n'):
            if "Best val AUC =" in line:
                best_auc = float(line.split("Best val AUC = ")[1].split()[0])
                break

        return {
            "model": model,
            "params": params,
            "output_dir": output_dir,
            "success": success,
            "best_val_auc": best_auc,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except Exception as e:
        return {
            "model": model,
            "params": params,
            "output_dir": output_dir,
            "success": False,
            "best_val_auc": None,
            "error": str(e)
        }


def run_grid_search(models, param_grid=None, epochs=10, output_base_dir="runs/search", seed=42):
    """Run grid search over specified models and parameter combinations."""
    if param_grid is None:
        param_grid = get_default_param_grids()

    results = []

    for model in models:
        if model not in param_grid:
            print(f"Warning: No parameter grid defined for {model}, skipping")
            continue

        grid = param_grid[model]
        param_names = list(grid.keys())
        param_values = list(grid.values())

        # Generate all combinations
        for combination in product(*param_values):
            params = dict(zip(param_names, combination))

            print(f"\n--- Running {model} with params: {params} ---")
            result = run_single_training(model, params, epochs, output_base_dir, seed)
            results.append(result)

            if result["success"] and result["best_val_auc"] is not None:
                print(f"Best AUC: {result['best_val_auc']:.4f}")
            else:
                print(f"Failed: {result.get('error', 'Unknown error')}")

    # Save summary results
    os.makedirs(output_base_dir, exist_ok=True)
    summary_file = os.path.join(output_base_dir, "grid_search_results.json")

    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print("GRID SEARCH SUMMARY")
    print(f"{'='*50}")

    successful_runs = [r for r in results if r["success"] and r["best_val_auc"] is not None]
    if successful_runs:
        best_result = max(successful_runs, key=lambda x: x["best_val_auc"])
        print(f"Best AUC: {best_result['best_val_auc']:.4f}")
        print(f"Best params: {best_result['params']}")
        print(f"Model: {best_result['model']}")
        print(f"Output dir: {best_result['output_dir']}")
    else:
        print("No successful runs completed.")

    print(f"Total runs: {len(results)}")
    print(f"Successful runs: {len(successful_runs)}")
    print(f"Results saved to: {summary_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run grid search for BUS-BRA classifier")
    parser.add_argument("--models", nargs="+", default=["efficientnet_b0", "dinov3_base"],
                        help="Models to include in grid search")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs per training run")
    parser.add_argument("--output_dir", type=str, default="runs/search",
                        help="Base directory for search outputs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    print(f"Starting grid search for models: {args.models}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Output base dir: {args.output_dir}")

    run_grid_search(
        models=args.models,
        epochs=args.epochs,
        output_base_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()