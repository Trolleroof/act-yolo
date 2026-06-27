#!/bin/bash
# Resumable end-to-end pipeline for act-yolo on RunPod.
# Safe to re-run after any pod pause/restart — each step checks if its
# output already exists and skips if so.
#
# Usage (first time or after resume):
#   tmux new -s pipeline
#   bash /workspace/act-yolo/scripts/runpod_pipeline.sh 2>&1 | tee /workspace/pipeline.log

set -euo pipefail

PROJECT_ROOT="/workspace/act-yolo"
DEMOS_DIR="$PROJECT_ROOT/data/demos"
NUM_DEMOS=150

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 1. System deps (idempotent) ───────────────────────────────────────────────
log "=== Installing system deps ==="
apt-get update -q 2>&1 | tail -1
apt-get install -y -q libegl1 libgl1 libglvnd0 2>&1 | tail -2

log "=== Installing Python deps ==="
pip install -q \
  mujoco dm_control \
  ultralytics \
  h5py opencv-python-headless \
  matplotlib tqdm ipython einops packaging \
  pyquaternion pyyaml rospkg --break-system-packages 2>&1 | tail -3

# ── 2. Clone / update repo ────────────────────────────────────────────────────
if [ ! -d "$PROJECT_ROOT/.git" ]; then
  log "=== Cloning repo ==="
  git clone https://github.com/Trolleroof/act-yolo.git "$PROJECT_ROOT"
else
  log "=== Repo already present, pulling latest ==="
  cd "$PROJECT_ROOT" && git pull --ff-only 2>/dev/null || true
fi
cd "$PROJECT_ROOT"

# ── 3. YOLO data generation ───────────────────────────────────────────────────
if [ ! -f "data/yolo_dataset/dataset.yaml" ]; then
  log "=== Generating YOLO training data ==="
  python detection/generate_yolo_data.py --n_frames 5000 --out data/yolo_dataset
else
  log "=== YOLO dataset already exists, skipping ==="
fi

# ── 4. YOLO training ──────────────────────────────────────────────────────────
if [ ! -f "weights/yolov8n_pickplace.pt" ]; then
  log "=== Training YOLOv8 ==="
  python detection/train_yolo.py \
    --data data/yolo_dataset/dataset.yaml \
    --name pickplace_full_aug \
    --epochs 50 --imgsz 480 --batch 64 --corrupt_aug --corrupt_p 0.85

  BEST="$(find weights runs -name best.pt -path '*pickplace_full_aug*' 2>/dev/null | head -1)"
  if [ -z "$BEST" ]; then log "ERROR: YOLO best.pt not found"; exit 1; fi
  cp "$BEST" weights/yolov8n_pickplace.pt
  log "YOLO weights installed -> weights/yolov8n_pickplace.pt"
else
  log "=== YOLO weights already exist, skipping training ==="
fi

# ── 5. YOLO sanity check ──────────────────────────────────────────────────────
log "=== YOLO sanity gate ==="
python scripts/yolo_sanity.py --weights weights/yolov8n_pickplace.pt --n 100

# ── 6. Demo collection (parallel, resumable) ──────────────────────────────────
log "=== Demo collection ==="
mkdir -p "$DEMOS_DIR"

# Determine resume points for each worker by checking highest existing episode
# in each worker's range, then starting from the next missing one.
resume_start() {
  local range_start=$1
  local range_end=$2
  local next=$range_start
  for i in $(seq $range_start $range_end); do
    if [ -f "$DEMOS_DIR/episode_${i}.hdf5" ]; then
      next=$((i + 1))
    else
      break
    fi
  done
  echo $next
}

W0_START=$(resume_start 0 49)
W1_START=$(resume_start 50 99)
W2_START=$(resume_start 100 149)

W0_REMAINING=$((50 - (W0_START - 0)))
W1_REMAINING=$((50 - (W1_START - 50)))
W2_REMAINING=$((50 - (W2_START - 100)))

log "Demo workers: W0 from ep $W0_START ($W0_REMAINING remaining), W1 from $W1_START ($W1_REMAINING remaining), W2 from $W2_START ($W2_REMAINING remaining)"

PIDS=()
if [ $W0_REMAINING -gt 0 ]; then
  python -u scripts/collect_demos.py --num_episodes $W0_REMAINING --start_idx $W0_START > /workspace/demos_0.log 2>&1 &
  PIDS+=($!)
fi
if [ $W1_REMAINING -gt 0 ]; then
  python -u scripts/collect_demos.py --num_episodes $W1_REMAINING --start_idx $W1_START > /workspace/demos_1.log 2>&1 &
  PIDS+=($!)
fi
if [ $W2_REMAINING -gt 0 ]; then
  python -u scripts/collect_demos.py --num_episodes $W2_REMAINING --start_idx $W2_START > /workspace/demos_2.log 2>&1 &
  PIDS+=($!)
fi

if [ ${#PIDS[@]} -gt 0 ]; then
  log "Waiting for ${#PIDS[@]} demo workers (PIDs: ${PIDS[*]})..."
  for pid in "${PIDS[@]}"; do wait $pid || true; done
  log "Demo workers done"
else
  log "All demo episodes already collected"
fi

# Count and verify demos
TOTAL_HDF5=$(ls "$DEMOS_DIR"/episode_*.hdf5 2>/dev/null | wc -l)
log "Total HDF5 files: $TOTAL_HDF5"

python3 - << 'PYEOF'
import h5py, glob
demos = sorted(glob.glob('/workspace/act-yolo/data/demos/episode_*.hdf5'))
successes = sum(1 for f in demos if h5py.File(f,'r').attrs.get('success', False))
print(f'Successful demos: {successes}/{len(demos)}')
if demos:
    with h5py.File(demos[0], 'r') as f:
        for k in ['top_rgb','wrist_rgb','cube_boxes','target_boxes','qpos','actions']:
            if k in f: print(f'  {k}: {f[k].shape}')
        cb = f['cube_boxes'][:]
        tb = f['target_boxes'][:]
        print(f'  cube_box detection rate: {(cb.sum(axis=1)!=0).mean():.1%}')
        print(f'  target_box detection rate: {(tb.sum(axis=1)!=0).mean():.1%}')
PYEOF

# ── 7. ACT training (parallel baseline + yolo_guided) ────────────────────────
BASELINE_CKPT="checkpoints/baseline/policy_last.ckpt"
YOLO_CKPT="checkpoints/yolo_guided/policy_last.ckpt"

log "=== ACT Training ==="
TRAIN_PIDS=()

if [ ! -f "$BASELINE_CKPT" ]; then
  log "Starting baseline ACT training..."
  python scripts/train.py --mode baseline --num_epochs 2000 \
    > /workspace/train_baseline.log 2>&1 &
  TRAIN_PIDS+=($!)
else
  log "Baseline checkpoint exists, skipping"
fi

if [ ! -f "$YOLO_CKPT" ]; then
  log "Starting yolo_guided ACT training..."
  python scripts/train.py --mode yolo_guided --num_epochs 2000 \
    > /workspace/train_yolo.log 2>&1 &
  TRAIN_PIDS+=($!)
else
  log "yolo_guided checkpoint exists, skipping"
fi

if [ ${#TRAIN_PIDS[@]} -gt 0 ]; then
  log "Waiting for ${#TRAIN_PIDS[@]} ACT training job(s)..."
  for pid in "${TRAIN_PIDS[@]}"; do wait $pid || log "WARNING: training job $pid exited non-zero"; done
  log "ACT training done"
fi

# ── 8. Robustness eval sweep (all 8 cells in parallel) ───────────────────────
log "=== Robustness eval sweep ==="
mkdir -p data/eval_results
EVAL_PIDS=()

for MODE in baseline yolo_guided; do
  for SEV in 0 1 2 3; do
    RESULT_FILE="data/eval_results/results_${MODE}_${SEV}.json"
    if [ ! -f "$RESULT_FILE" ]; then
      log "  Launching eval: $MODE sev=$SEV"
      python scripts/evaluate.py --mode $MODE --corruption_severity $SEV \
        --num_rollouts 50 \
        > /workspace/eval_${MODE}_${SEV}.log 2>&1 &
      EVAL_PIDS+=($!)
    else
      log "  Eval $MODE sev=$SEV already done, skipping"
    fi
  done
done

if [ ${#EVAL_PIDS[@]} -gt 0 ]; then
  log "Waiting for ${#EVAL_PIDS[@]} eval jobs..."
  for pid in "${EVAL_PIDS[@]}"; do wait $pid || log "WARNING: eval job $pid exited non-zero"; done
fi

# ── 9. Significance testing ───────────────────────────────────────────────────
log "=== Significance testing ==="
python scripts/significance.py --results_dir data/eval_results

# ── 10. Rsync results back locally ───────────────────────────────────────────
log "=== ALL DONE ==="
log "Results in data/eval_results/"
log "To download: rsync -avz -e 'ssh -i ~/.ssh/id_ed25519' <pod-proxy>:/workspace/act-yolo/data/ ~/act-yolo-results/"
