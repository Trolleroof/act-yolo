#!/bin/bash
# RunPod setup + training script for act-yolo.
# Run once on a fresh pod: bash scripts/setup_runpod.sh
set -e

REPO="https://github.com/Trolleroof/act-yolo.git"
PROJECT_ROOT="/workspace/act-yolo"

# MuJoCo rendering on RunPod is headless; EGL avoids GLFW/X11 DISPLAY errors.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# ── 1. Clone ──────────────────────────────────────────────────────────────────
if [ ! -d "$PROJECT_ROOT" ]; then
  git clone "$REPO" "$PROJECT_ROOT"
fi
cd "$PROJECT_ROOT"

# ── 2. Install deps ───────────────────────────────────────────────────────────
apt-get update
apt-get install -y --no-install-recommends \
  libegl1 \
  libgl1 \
  libglvnd0 \
  libgles2

pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -q \
  mujoco dm_control \
  ultralytics \
  h5py opencv-python-headless \
  matplotlib tqdm ipython einops packaging \
  pyquaternion pyyaml rospkg

# ── 3. Generate YOLO training data (5k frames) ────────────────────────────────
echo "=== Generating YOLO data ==="
python detection/generate_yolo_data.py --n_frames 5000 --out data/yolo_dataset

# ── 4. Train YOLO detector (corruption-aug / Option A, on by default) ─────────
echo "=== Training YOLOv8 (corruption augmentation enabled) ==="
python detection/train_yolo.py \
  --data data/yolo_dataset/dataset.yaml \
  --name pickplace_full_aug \
  --epochs 50 --imgsz 480 --batch 64 --corrupt_aug --corrupt_p 0.85

# Ultralytics writes to <runs>/<name>/weights/best.pt — copy it to the single
# path collect_demos.py and evaluate.py load. Without this, the rest of the
# pipeline silently falls back to ZERO boxes.
BEST="$(find weights runs /opt -name best.pt -path '*pickplace_full_aug*' 2>/dev/null | head -1)"
if [ -z "$BEST" ]; then echo "ERROR: trained YOLO best.pt not found"; exit 1; fi
cp "$BEST" weights/yolov8n_pickplace.pt
echo "Installed YOLO weights -> weights/yolov8n_pickplace.pt (from $BEST)"

# ── 4b. Sanity gate — must pass before spending GPU on ACT ────────────────────
echo "=== YOLO rendered-frame sanity gate ==="
python scripts/yolo_sanity.py --weights weights/yolov8n_pickplace.pt --n 100

# ── 5. Collect demos ──────────────────────────────────────────────────────────
echo "=== Collecting demos (batch 1) ==="
python scripts/collect_demos.py --num_episodes 100 --out data/demos --seed 0

echo "=== Collecting demos (batch 2) ==="
python scripts/collect_demos.py --num_episodes 100 --out data/demos --start_idx 100 --seed 100

echo "Demo count:"
python -c "
from utils import get_successful_episode_ids
ids = get_successful_episode_ids('data/demos')
print(f'  Successful: {len(ids)}')
"

# ── 6. Train ACT ──────────────────────────────────────────────────────────────
echo "=== Training baseline ACT ==="
python scripts/train.py --mode baseline --num_epochs 2000

echo "=== Training yolo_guided ACT ==="
python scripts/train.py --mode yolo_guided --num_epochs 2000

# ── 7. Robustness eval sweep ──────────────────────────────────────────────────
echo "=== Running robustness eval sweep ==="
bash scripts/run_robustness.sh

echo ""
echo "All done. Results in data/robustness_results.json and data/robustness_curves.png"
echo "Download with: rsync -avz <pod-ip>:~/act-yolo/data/ ./data/"
