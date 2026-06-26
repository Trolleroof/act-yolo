#!/bin/bash
set -e

# Resolve scripts directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

cd "$PROJECT_ROOT"

echo "=== Starting Robustness Evaluation Sweep ==="

# 100 rollouts/cell gives enough power to detect ~15pp success-rate gaps.
# Both modes are reseeded per-rollout to the same layouts (paired), so the
# significance test is McNemar's — see scripts/significance.py.
NUM_ROLLOUTS="${NUM_ROLLOUTS:-100}"

for mode in baseline yolo_guided; do
  echo "Evaluating mode: $mode"
  for sev in 0 1 2 3; do
    echo "Running evaluation: mode=$mode, severity=$sev"
    python scripts/evaluate.py --mode "$mode" --corruption_severity "$sev" --num_rollouts "$NUM_ROLLOUTS"
  done
done

echo "=== Statistical Significance (baseline vs yolo_guided) ==="
python scripts/significance.py --mode_a baseline --mode_b yolo_guided

echo "=== Generating Robustness Curves ==="
python scripts/plot_robustness.py

echo "Sweep complete. Plots + significance table generated in data/ folder."
