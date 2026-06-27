"""Re-run YOLO on stored top_rgb frames and update cube_boxes/target_boxes in-place.

Run after re-training YOLO when demos were collected with bad weights.
"""
import argparse, glob, os, sys
import numpy as np
import h5py

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from detection.yolo_detector import YOLODetector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demos_dir', default='data/demos')
    parser.add_argument('--weights', default='weights/yolov8n_pickplace.pt')
    parser.add_argument('--conf', type=float, default=0.001)
    args = parser.parse_args()

    detector = YOLODetector(weights=args.weights, conf=args.conf)
    files = sorted(glob.glob(os.path.join(args.demos_dir, 'episode_*.hdf5')))
    print(f'Repopulating boxes in {len(files)} demo files...')

    cube_detected, target_detected, total_frames = 0, 0, 0
    for i, path in enumerate(files):
        with h5py.File(path, 'r+') as f:
            frames = f['top_rgb'][:]  # (T, H, W, 3)
            T = frames.shape[0]
            cube_boxes = np.zeros((T, 5), dtype=np.float32)
            target_boxes = np.zeros((T, 5), dtype=np.float32)

            for t in range(T):
                dets = detector.detect(frames[t])
                if dets['cube'] is not None:
                    cube_boxes[t] = dets['cube']
                    cube_detected += 1
                if dets['target_zone'] is not None:
                    target_boxes[t] = dets['target_zone']
                    target_detected += 1
                total_frames += 1

            f['cube_boxes'][:] = cube_boxes
            f['target_boxes'][:] = target_boxes

        if (i + 1) % 20 == 0 or i == len(files) - 1:
            print(f'  [{i+1}/{len(files)}] cube: {cube_detected/total_frames:.1%}, '
                  f'target: {target_detected/total_frames:.1%}')

    print(f'Done. Cube detection: {cube_detected/total_frames:.1%}, '
          f'Target detection: {target_detected/total_frames:.1%}')

if __name__ == '__main__':
    main()
