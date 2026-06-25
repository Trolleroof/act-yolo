#!/bin/bash
# RunPod setup + training script for act-yolo.
# Run once on a fresh pod: bash scripts/setup_runpod.sh
set -e

REPO="https://github.com/Trolleroof/act-yolo.git"
PROJECT_ROOT="$HOME/act-yolo"

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

# ── 4. Train YOLO detector ────────────────────────────────────────────────────
echo "=== Training YOLOv8 ==="
python detection/train_yolo.py
# weights saved to weights/yolov8n_pickplace.pt

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
