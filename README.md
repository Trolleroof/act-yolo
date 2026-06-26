# ACT-YOLO: Is a robot policy more robust to camera corruption when guided by object detections?

A controlled study on a simulated MuJoCo pick-and-place task. We compare two
[ACT](https://github.com/tonyzhaozh/act) (Action Chunking Transformer) policies
under increasing visual corruption:

- **`baseline`** — ACT from camera images + proprioception only.
- **`yolo_guided`** — same, plus YOLOv8 object/target bounding boxes appended to
  the proprioception vector.

**Hypothesis:** as the cameras degrade, the detection boxes give the policy a
stable signal that raw pixels no longer do, so `yolo_guided` should degrade more
gracefully than `baseline`.

## Why this is set up the way it is

The comparison is only meaningful if neither side is sandbagged, so three
train/eval gaps are closed **symmetrically**:

1. **Detector robustness.** YOLO is trained on clean renders but evaluated on
   corrupted frames. A clean-only detector collapses on the small `cube` at
   medium/high corruption. Fixed by training YOLO with the *exact* corruption
   operator used at eval (`detection/corruption_aug.py`). Controlled result
   (same data, same epochs, augmentation the only variable):

   | severity | clean-only | corruption-aug |
   |----------|-----------|----------------|
   | 0        | 1.00      | 1.00           |
   | 1        | 0.95      | 1.00           |
   | 2        | 0.17      | 1.00           |
   | 3        | 0.03      | 0.98           |

   (cube recall @ IoU≥0.5; `target_zone` stays ~1.00 throughout.)

2. **Policy robustness.** Both ACT modes train with identical image augmentation
   (`--image_aug`) so the baseline isn't penalized merely for never having seen
   corruption. The *only* difference between modes is the box channel.

3. **Box realism.** `yolo_guided` also trains with box jitter/dropout
   (`--box_aug`) so the policy tolerates the imperfect detections a real YOLO
   produces on corrupted frames.

Comparisons are reported with a **paired McNemar test** (both modes face the same
per-rollout scene seeds) plus Wilson confidence intervals — see
`scripts/significance.py`.

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| Generate YOLO data | `detection/generate_yolo_data.py` | `data/yolo_dataset/` (auto-labeled from MuJoCo segmentation) |
| Train detector | `detection/train_yolo.py --corrupt_aug` | `weights/yolov8n_pickplace.pt` |
| **Gate detector** | `scripts/yolo_sanity.py` | recall@IoU per severity (pass before spending GPU on ACT) |
| Collect demos | `scripts/collect_demos.py` | `data/demos/*.hdf5` (scripted IK pick-place) |
| Train ACT | `scripts/train.py --mode {baseline,yolo_guided}` | `checkpoints/<mode>/` |
| Eval under corruption | `scripts/evaluate.py --mode <m> --corruption_severity <0-3>` | `data/eval_results/*.json` |
| Sweep + stats + plots | `scripts/run_robustness.sh` | significance table + robustness curves |

Corruption severities (`vision/corruption.py`): `0` clean, `1` low, `2` medium,
`3` high — additive noise, blur, brightness/contrast shift, and JPEG compression.

## Quick start

```bash
# Detector
python detection/generate_yolo_data.py --n_frames 5000 --out data/yolo_dataset
python detection/train_yolo.py --data data/yolo_dataset/dataset.yaml --corrupt_aug
cp <runs>/.../best.pt weights/yolov8n_pickplace.pt        # see train_yolo.py output
python scripts/yolo_sanity.py --weights weights/yolov8n_pickplace.pt --n 100

# Demos + policies
python scripts/collect_demos.py --num_episodes 200 --out data/demos
python scripts/train.py --mode baseline     --num_epochs 2000
python scripts/train.py --mode yolo_guided  --num_epochs 2000

# Compare under corruption (100 rollouts/cell, paired McNemar + CIs + plots)
NUM_ROLLOUTS=100 bash scripts/run_robustness.sh
```

On a fresh GPU box, `scripts/setup_runpod.sh` runs all of the above end-to-end.

## Repo layout

- `detection/` — YOLO data generation, training (with corruption aug), detector wrapper.
- `vision/corruption.py` — the corruption operator (single source of truth for train aug and eval).
- `envs/`, `sim_env.py`, `assets/` — MuJoCo pick-place task.
- `detr/`, `policy.py`, `imitate_episodes.py`, `utils.py` — ACT model + training.
- `scripts/` — demo collection, training, evaluation, significance, plotting, RunPod setup.
