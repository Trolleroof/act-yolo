"""
Visualize generated YOLO labels overlaid on training images.

Usage:
    python scripts/verify_yolo_data.py --data /tmp/yolo_test --n 5
    # then: open /tmp/yolo_verify/
"""
import argparse
import os
import cv2
import numpy as np

COLORS = {0: (255, 50, 50), 1: (50, 220, 50)}   # BGR: red=cube, green=target_zone
NAMES  = {0: 'cube', 1: 'target_zone'}

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='data/yolo_dataset')
parser.add_argument('--n', type=int, default=8, help='number of images to visualize')
parser.add_argument('--out', default='/tmp/yolo_verify')
args = parser.parse_args()

img_dir = os.path.join(args.data, 'images', 'train')
lbl_dir = os.path.join(args.data, 'labels', 'train')
os.makedirs(args.out, exist_ok=True)

stems = sorted(os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.jpg'))[:args.n]

for stem in stems:
    img = cv2.imread(os.path.join(img_dir, f"{stem}.jpg"))
    H, W = img.shape[:2]

    lbl_path = os.path.join(lbl_dir, f"{stem}.txt")
    if not os.path.exists(lbl_path):
        continue

    with open(lbl_path) as f:
        for line in f:
            cls, cx, cy, bw, bh = map(float, line.split())
            cls = int(cls)
            x1 = int((cx - bw / 2) * W)
            y1 = int((cy - bh / 2) * H)
            x2 = int((cx + bw / 2) * W)
            y2 = int((cy + bh / 2) * H)
            color = COLORS[cls]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, NAMES[cls], (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    out_path = os.path.join(args.out, f"{stem}.jpg")
    cv2.imwrite(out_path, img)

print(f"Saved {len(stems)} annotated frames to {args.out}")
print(f"  open {args.out}")
