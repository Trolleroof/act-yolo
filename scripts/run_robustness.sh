#!/bin/bash
set -e

# Resolve scripts directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

cd "$PROJECT_ROOT"

echo "=== Starting Robustness Evaluation Sweep ==="

for mode in baseline yolo_guided; do
  echo "Evaluating mode: $mode"
  for sev in 0 1 2 3; do
    echo "Running evaluation: mode=$mode, severity=$sev"
    python scripts/evaluate.py --mode "$mode" --corruption_severity "$sev" --num_rollouts 50
  done
done

echo "=== Generating Robustness Curves ==="
python scripts/plot_robustness.py

echo "Sweep complete. Plots generated in data/ folder."
