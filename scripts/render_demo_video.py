"""
Render a short demo video of the pick-place env with a scripted motion.

Usage:
    python scripts/render_demo_video.py --out /tmp/demo.mp4
    open /tmp/demo.mp4
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import mujoco
from detection.generate_yolo_data import get_bbox_from_mask, CUBE_GEOM, TARGET_GEOM

XML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'pick_place.xml')

START_POSE = np.array([0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239])

parser = argparse.ArgumentParser()
parser.add_argument('--out', default='/tmp/demo.mp4')
parser.add_argument('--steps', type=int, default=300)
parser.add_argument('--fps', type=int, default=30)
args = parser.parse_args()

model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

mujoco.mj_resetData(model, data)
data.qpos[:8] = START_POSE
data.ctrl[:8] = START_POSE
data.qpos[8:11]  = [-0.05, 0.55, 0.05]   # cube start
data.qpos[11:15] = [1, 0, 0, 0]
mujoco.mj_forward(model, data)

top_cam   = model.camera('top').id
wrist_cam = model.camera('left_wrist').id

renderer_top   = mujoco.Renderer(model, height=480, width=480)
renderer_wrist = mujoco.Renderer(model, height=480, width=480)
renderer_seg_top   = mujoco.Renderer(model, height=480, width=480)
renderer_seg_top.enable_segmentation_rendering()
renderer_seg_wrist = mujoco.Renderer(model, height=480, width=480)
renderer_seg_wrist.enable_segmentation_rendering()

cube_geom_id   = model.geom(CUBE_GEOM).id
target_geom_id = model.geom(TARGET_GEOM).id

BOX_COLORS = {0: (50, 50, 255), 1: (50, 200, 50)}   # BGR: red=cube, green=target_zone
BOX_NAMES  = {0: 'cube', 1: 'target_zone'}

# Side-by-side: top | wrist
out_w, out_h = 960, 480
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(args.out, fourcc, args.fps, (out_w, out_h))

# Simple scripted motion: sweep the arm joints with sinusoids
def scripted_ctrl(t):
    ctrl = START_POSE.copy()
    ctrl[0] += 0.4 * np.sin(t * 0.03)          # waist sweep
    ctrl[1] += 0.2 * np.sin(t * 0.02 + 0.5)    # shoulder dip
    ctrl[2] += 0.15 * np.sin(t * 0.025)         # elbow
    ctrl[4] += 0.2 * np.sin(t * 0.04)           # wrist angle
    # gripper: open then close
    gripper = 0.05 if t < args.steps // 2 else 0.022
    ctrl[6] = gripper
    ctrl[7] = -gripper
    return ctrl

print(f"Rendering {args.steps} steps → {args.out}")

for t in range(args.steps):
    data.ctrl[:8] = scripted_ctrl(t)
    for _ in range(4):   # 4 sim substeps per frame (80 Hz sim, 20 Hz video)
        mujoco.mj_step(model, data)

    renderer_top.update_scene(data, camera=top_cam)
    renderer_wrist.update_scene(data, camera=wrist_cam)

    top_rgb   = renderer_top.render()
    wrist_rgb = renderer_wrist.render()

    top_bgr   = cv2.cvtColor(top_rgb,   cv2.COLOR_RGB2BGR)
    wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)

    def draw_boxes(frame_bgr, cam_id, seg_renderer):
        seg_renderer.update_scene(data, camera=cam_id)
        seg = seg_renderer.render()
        geom_ids = seg[:, :, 0]
        H, W = frame_bgr.shape[:2]
        for cls_id, geom_id in enumerate([cube_geom_id, target_geom_id]):
            mask = (geom_ids == geom_id).astype(np.uint8)
            bbox = get_bbox_from_mask(mask)
            if bbox is None:
                continue
            cx, cy, bw, bh = bbox
            x1 = int((cx - bw / 2) * W)
            y1 = int((cy - bh / 2) * H)
            x2 = int((cx + bw / 2) * W)
            y2 = int((cy + bh / 2) * H)
            color = BOX_COLORS[cls_id]
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_bgr, BOX_NAMES[cls_id], (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    draw_boxes(top_bgr,   top_cam,   renderer_seg_top)
    draw_boxes(wrist_bgr, wrist_cam, renderer_seg_wrist)

    frame = np.concatenate([top_bgr, wrist_bgr], axis=1)

    # Labels
    cv2.putText(frame, 'top',   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame, 'wrist', (490, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    writer.write(frame)

    if t % 50 == 0:
        print(f"  {t}/{args.steps}")

writer.release()
print(f"Done. Open with:\n  open {args.out}")
