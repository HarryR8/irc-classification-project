#!/usr/bin/env bash
set -euo pipefail

RUNS_DIR="runs"
OUTPUT_DIR="results_bundle"
MAKE_TAR=false

usage() {
    echo "Usage: bash scripts/collect_results.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --runs_dir   DIR   Source directory to search  (default: runs)"
    echo "  --output_dir DIR   Staging directory to write to  (default: results_bundle)"
    echo "  --tar              Also create <output_dir>.tar.gz after copying"
    echo "  -h, --help         Show this help message"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs_dir)   RUNS_DIR="$2";   shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --tar)        MAKE_TAR=true;   shift   ;;
        -h|--help)    usage; exit 0             ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ ! -d "$RUNS_DIR" ]]; then
    echo "Error: runs directory '$RUNS_DIR' not found." >&2
    exit 1
fi

RESULT_FILES=(config.json history.json)
EVAL_GLOBS=("eval_*.json" "eval_*_roc_curve.png" "eval_*_threshold_sweep.csv")

total=0
skipped=0

while IFS= read -r config_file; do
    run_dir=$(dirname "$config_file")
    rel_dir="${run_dir#"$RUNS_DIR"/}"
    dest="$OUTPUT_DIR/$rel_dir"

    files=()
    for name in "${RESULT_FILES[@]}"; do
        [[ -f "$run_dir/$name" ]] && files+=("$run_dir/$name")
    done
    for glob in "${EVAL_GLOBS[@]}"; do
        for f in "$run_dir"/$glob; do
            [[ -f "$f" ]] && files+=("$f")
        done
    done

    if [[ ${#files[@]} -eq 0 ]]; then
        skipped=$((skipped + 1))
        continue
    fi

    # Count eval json files for summary
    eval_count=0
    for f in "${files[@]}"; do
        [[ "$f" == */eval_*.json ]] && eval_count=$((eval_count + 1))
    done

    mkdir -p "$dest"
    cp "${files[@]}" "$dest/"

    printf "  ✓  %-55s  %2d files  (%d eval JSON)\n" \
        "$rel_dir" "${#files[@]}" "$eval_count"
    total=$((total + 1))

done < <(find "$RUNS_DIR" -name "config.json" | sort)

echo ""
echo "Collected $total run(s) into '$OUTPUT_DIR/'  ($skipped skipped — no result files)."

if $MAKE_TAR; then
    archive="${OUTPUT_DIR}.tar.gz"
    tar -czf "$archive" "$OUTPUT_DIR"
    abs_path="$(cd "$(dirname "$archive")" && pwd)/$(basename "$archive")"
    echo ""
    echo "Archive created: $abs_path"
    echo ""
    echo "Download with:"
    echo "  scp <user>@<hpc>:$abs_path ."
fi
