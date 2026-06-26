"""
Rendered-frame YOLO sanity gate.

YOLO's own validation mAP is not trusted here: the failure mode we care about
only shows up on freshly rendered MuJoCo frames under the eval-time corruption,
not on the held-out training split. This script:

  1. renders top-camera frames at random cube positions,
  2. gets ground-truth boxes from MuJoCo segmentation,
  3. corrupts each frame at severities 0-3 (the exact eval operator),
  4. runs YOLODetector.detect,
  5. reports recall @ IoU>=0.5 and mean IoU per class per severity.

This is the gate to pass BEFORE spending GPU on ACT.

Usage:
    python scripts/yolo_sanity.py --weights weights/yolov8n_pickplace.pt --n 100
"""
import argparse
import os
import sys

import numpy as np
import mujoco

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from detection.generate_yolo_data import (
    XML_PATH, CUBE_GEOM, TARGET_GEOM, CUBE_SPAWN_X, CUBE_SPAWN_Y, CUBE_Z,
    random_arm_pose, get_bbox_from_mask,
)
from detection.yolo_detector import YOLODetector
from vision.corruption import corrupt_frame


def iou_xywhn(a, b):
    """IoU of two normalized [cx,cy,w,h] boxes."""
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='weights/yolov8n_pickplace.pt')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--severities', type=int, nargs='+', default=[0, 1, 2, 3])
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=12345)
    args = ap.parse_args()

    weights = args.weights if os.path.isabs(args.weights) else os.path.join(_PROJECT_ROOT, args.weights)
    detector = YOLODetector(weights=weights)

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    cube_gid = model.geom(CUBE_GEOM).id
    target_gid = model.geom(TARGET_GEOM).id
    cam_id = model.camera('top').id
    renderer = mujoco.Renderer(model, height=480, width=480)
    seg = mujoco.Renderer(model, height=480, width=480)
    seg.enable_segmentation_rendering()

    rng = np.random.default_rng(args.seed)
    # Pre-generate clean frames + GT once; corrupt per severity at eval time.
    frames, gts = [], []
    for _ in range(args.n):
        cx, cy = rng.uniform(*CUBE_SPAWN_X), rng.uniform(*CUBE_SPAWN_Y)
        arm = random_arm_pose(rng)
        mujoco.mj_resetData(model, data)
        data.qpos[:8] = arm
        data.ctrl[:8] = arm
        data.qpos[8:11] = [cx, cy, CUBE_Z]
        data.qpos[11:15] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam_id)
        rgb = renderer.render().copy()
        seg.update_scene(data, camera=cam_id)
        gid = seg.render()[:, :, 0]
        gt = {
            'cube': get_bbox_from_mask((gid == cube_gid).astype(np.uint8)),
            'target_zone': get_bbox_from_mask((gid == target_gid).astype(np.uint8)),
        }
        frames.append(rgb)
        gts.append(gt)
    renderer.close()
    seg.close()

    print("severity,class,recall@iou,mean_iou,seen,hits")
    pass_lines = []
    for sev in args.severities:
        agg = {c: {'seen': 0, 'hits': 0, 'iou_sum': 0.0} for c in ('cube', 'target_zone')}
        crng = np.random.default_rng(args.seed + 1000 + sev)
        for rgb, gt in zip(frames, gts):
            frame = corrupt_frame(rgb, sev, rng=crng) if sev > 0 else rgb
            dets = detector.detect(frame)
            for c in ('cube', 'target_zone'):
                if gt[c] is None:
                    continue
                agg[c]['seen'] += 1
                pred = dets.get(c)
                if pred is not None:
                    iou = iou_xywhn(np.asarray(pred[:4]), np.asarray(gt[c]))
                    if iou >= args.iou:
                        agg[c]['hits'] += 1
                        agg[c]['iou_sum'] += iou
        for c in ('cube', 'target_zone'):
            s, h = agg[c]['seen'], agg[c]['hits']
            recall = h / s if s else 0.0
            miou = agg[c]['iou_sum'] / h if h else 0.0
            print(f"{sev},{c},{recall:.3f},{miou:.3f},{s},{h}")
            pass_lines.append((sev, c, recall))

    # Pass criteria: detector must be reliable at clean/low corruption before
    # we trust it to guide the policy.
    def rec(sev, c):
        return next((r for s, cc, r in pass_lines if s == sev and cc == c), 0.0)

    print("\n--- gate ---")
    gate0 = rec(0, 'cube') >= 0.95 and rec(0, 'target_zone') >= 0.95
    gate1 = rec(1, 'cube') >= 0.90 and rec(1, 'target_zone') >= 0.95
    print(f"severity0 (cube>=.95 & target>=.95): {'PASS' if gate0 else 'FAIL'}")
    print(f"severity1 (cube>=.90 & target>=.95): {'PASS' if gate1 else 'FAIL'}")
    if 2 in args.severities:
        print(f"severity2 cube recall: {rec(2,'cube'):.3f}  (goal: lift well above clean-only ~0.34)")
    if 3 in args.severities:
        print(f"severity3 cube recall: {rec(3,'cube'):.3f}  (goal: lift well above clean-only ~0.12)")


if __name__ == '__main__':
    main()
