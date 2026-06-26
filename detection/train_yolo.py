"""
Fine-tune YOLOv8n on auto-labeled sim frames.

Usage (run on RunPod after generating data):
    python detection/train_yolo.py --data data/yolo_dataset/dataset.yaml
"""
import argparse
import os
import sys

# Ensure project root is importable when run as `python detection/train_yolo.py`.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='data/yolo_dataset/dataset.yaml')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--imgsz', type=int, default=480)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--weights', default='yolov8n.pt')
parser.add_argument('--name', default='yolov8n_pickplace')
parser.add_argument('--corrupt_aug', dest='corrupt_aug', action='store_true', default=True,
                    help='Domain-randomize with eval-matched corruption (Option A, default on).')
parser.add_argument('--no_corrupt_aug', dest='corrupt_aug', action='store_false',
                    help='Disable corruption augmentation (clean-only baseline).')
parser.add_argument('--corrupt_p', type=float, default=0.85,
                    help='Probability a training image is corrupted.')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

# Enable corruption augmentation BEFORE building the dataset/trainer so the
# monkeypatch is in place when Ultralytics constructs its dataloaders.
if args.corrupt_aug:
    from detection.corruption_aug import enable_corruption_aug
    enable_corruption_aug(p=args.corrupt_p, seed=args.seed)

from ultralytics import YOLO

os.makedirs('weights', exist_ok=True)

model = YOLO(args.weights)
model.train(
    data=args.data,
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=args.batch,
    project='weights',
    name=args.name,
    seed=args.seed,
    exist_ok=True,
    # Keep Ultralytics' own color/geometry augs off: the validated recipe is
    # clean geometry + our eval-matched corruption as the only randomization.
    mosaic=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    fliplr=0.0, scale=0.0, translate=0.0, degrees=0.0,
)

print(f"Training complete. Best weights at: weights/{args.name}/weights/best.pt")
print("Copy to: weights/yolov8n_pickplace.pt")
