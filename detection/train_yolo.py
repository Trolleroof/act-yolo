"""
Fine-tune YOLOv8n on auto-labeled sim frames.

Usage (run on RunPod after generating data):
    python detection/train_yolo.py --data data/yolo_dataset/dataset.yaml
"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='data/yolo_dataset/dataset.yaml')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--imgsz', type=int, default=480)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--weights', default='yolov8n.pt')
args = parser.parse_args()

from ultralytics import YOLO

os.makedirs('weights', exist_ok=True)

model = YOLO(args.weights)
model.train(
    data=args.data,
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=args.batch,
    project='weights',
    name='yolov8n_pickplace',
    exist_ok=True,
)

print("Training complete. Best weights at: weights/yolov8n_pickplace/weights/best.pt")
print("Copy to: weights/yolov8n_pickplace.pt")
