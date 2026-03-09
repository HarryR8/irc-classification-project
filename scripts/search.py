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
            "lr": [1e-5, 5e-5],
            "weight_decay": [1e-2, 5e-3, 1e-3],
            "batch_size": [16, 32],
        }
    }


def run_single_training(model, params, epochs, output_base_dir="runs/search", seed=42, python_cmd="", images_dir=None, masks_dir=None):
    """Run a single training run with given parameters."""
    # Create unique output directory
    param_str = "_".join([f"{k}={v}" for k, v in params.items()])
    output_dir = os.path.join(output_base_dir, f"{model}_{param_str}")

    # Build command — default to sys.executable so the subprocess uses the same
    # venv that is already active, avoiding "No module named 'busbra'" errors.
    cmd = (python_cmd.split() if python_cmd else [sys.executable]) + [
        "scripts/train.py",
        "--model", model,
        "--epochs", str(epochs),
        "--batch_size", str(params["batch_size"]),
        "--lr", str(params["lr"]),
        "--weight_decay", str(params["weight_decay"]),
        "--output_dir", output_dir,
        "--seed", str(seed)
    ]

    if images_dir:
        cmd += ["--images_dir", images_dir]
    if masks_dir:
        cmd += ["--masks_dir", masks_dir]

    print(f"Running: {' '.join(cmd)}")

    # Inject PYTHONPATH so subprocesses can import busbra from src/ layout
    repo_root = Path(__file__).parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")

    # Run training, streaming output to terminal
    try:
        result = subprocess.run(cmd, cwd=repo_root, env=env)
        success = result.returncode == 0

        # Read best val AUC from history.json
        best_auc = None
        history_path = os.path.join(output_dir, "history.json")
        if success and os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
            best_auc = max(e["val_auc"] for e in history)

        return {
            "model": model,
            "params": params,
            "output_dir": output_dir,
            "success": success,
            "best_val_auc": best_auc,
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


def run_grid_search(models, param_grid=None, epochs=10, output_base_dir="runs/search", seed=42, python_cmd="", images_dir=None, masks_dir=None):
    """Run grid search over specified models and parameter combinations."""
    if param_grid is None:
        param_grid = get_default_param_grids()

    results = []

    # Pre-compute total number of runs for progress display
    total_runs = sum(
        len(list(product(*param_grid[m].values())))
        for m in models if m in param_grid
    )
    run_idx = 0

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
            run_idx += 1

            print(f"\n{'='*50}")
            print(f"Run {run_idx}/{total_runs} — {model} | {params}")
            print(f"{'='*50}")
            result = run_single_training(model, params, epochs, output_base_dir, seed, python_cmd, images_dir, masks_dir)
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
    parser.add_argument("--python_cmd", type=str, default="",
                        help="Python command to invoke train.py (default: sys.executable). "
                             "Override on HPC with e.g. 'python3'.")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="Path to images directory (required on HPC)")
    parser.add_argument("--masks_dir", type=str, default=None,
                        help="Path to masks directory (required on HPC)")

    args = parser.parse_args()

    print(f"Starting grid search for models: {args.models}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Output base dir: {args.output_dir}")
    print(f"Python command: {args.python_cmd or sys.executable}")

    run_grid_search(
        models=args.models,
        epochs=args.epochs,
        output_base_dir=args.output_dir,
        seed=args.seed,
        python_cmd=args.python_cmd,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
    )


if __name__ == "__main__":
    main()