"""
Auto-generate YOLO training data from MuJoCo GT segmentation.

Usage:
    python detection/generate_yolo_data.py --n_frames 5000 --out data/yolo_dataset
"""
import argparse
import os
import sys

import cv2
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

XML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'pick_place.xml')

# Geom names from MJCF — resolved at runtime, not hard-coded as ints
CUBE_GEOM = 'cube_geom'
TARGET_GEOM = 'target_zone_geom'

# Cube spawn range (must match envs/pick_place.py)
CUBE_SPAWN_X = (-0.15, 0.05)
CUBE_SPAWN_Y = (0.50, 0.60)
CUBE_Z = 0.05

# Arm joint random range for pose diversity
ARM_JOINT_NOISE = 0.3  # radians


def random_arm_pose(rng):
    base = np.array([0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239])
    noise = rng.uniform(-ARM_JOINT_NOISE, ARM_JOINT_NOISE, size=6)
    pose = base.copy()
    pose[:6] += noise
    return pose


def get_bbox_from_mask(mask):
    """Return YOLO-format [cx, cy, w, h] normalized, or None if object invisible."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 4:  # skip tiny blips
        return None
    x, y, w, h = cv2.boundingRect(cnt)
    H, W = mask.shape
    cx = (x + w / 2) / W
    cy = (y + h / 2) / H
    return cx, cy, w / W, h / H


def generate(n_frames: int, out_dir: str, seed: int = 0):
    img_dir = os.path.join(out_dir, 'images', 'train')
    lbl_dir = os.path.join(out_dir, 'labels', 'train')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    # Resolve geom IDs once — names come from MJCF, never hard-coded ints
    cube_geom_id   = model.geom(CUBE_GEOM).id
    target_geom_id = model.geom(TARGET_GEOM).id

    renderer     = mujoco.Renderer(model, height=480, width=480)
    renderer_seg = mujoco.Renderer(model, height=480, width=480)
    renderer_seg.enable_segmentation_rendering()

    cam_id = model.camera('top').id

    rng = np.random.default_rng(seed)
    saved = 0
    skipped = 0

    print(f"Generating {n_frames} frames → {out_dir}")

    for i in range(n_frames):
        # Randomize cube position and arm pose for visual diversity
        cx = rng.uniform(*CUBE_SPAWN_X)
        cy = rng.uniform(*CUBE_SPAWN_Y)
        arm_pose = random_arm_pose(rng)

        mujoco.mj_resetData(model, data)
        data.qpos[:8] = arm_pose
        data.ctrl[:8] = arm_pose
        # cube free joint starts at qpos index 8
        data.qpos[8:11] = [cx, cy, CUBE_Z]
        data.qpos[11:15] = [1, 0, 0, 0]  # upright quaternion
        mujoco.mj_forward(model, data)

        # Render RGB
        renderer.update_scene(data, camera=cam_id)
        rgb = renderer.render().copy()

        # Render segmentation — channel 0 = geom id, channel 1 = object type
        renderer_seg.update_scene(data, camera=cam_id)
        seg = renderer_seg.render()
        geom_ids = seg[:, :, 0]

        labels = []
        for cls_id, geom_id in enumerate([cube_geom_id, target_geom_id]):
            mask = (geom_ids == geom_id).astype(np.uint8)
            bbox = get_bbox_from_mask(mask)
            if bbox is None:
                continue
            bx, by, bw, bh = bbox
            labels.append(f"{cls_id} {bx:.4f} {by:.4f} {bw:.4f} {bh:.4f}")

        # Skip frames where neither object is visible
        if not labels:
            skipped += 1
            continue

        stem = f"{saved:05d}"
        cv2.imwrite(os.path.join(img_dir, f"{stem}.jpg"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        with open(os.path.join(lbl_dir, f"{stem}.txt"), 'w') as f:
            f.write('\n'.join(labels))

        saved += 1
        if saved % 500 == 0:
            print(f"  {saved}/{n_frames} saved  (skipped {skipped} invisible frames)")

    print(f"Done. {saved} frames saved, {skipped} skipped.")

    # Write dataset.yaml for YOLO training
    yaml_path = os.path.join(out_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(out_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/train\n")  # use same split for now; swap in a held-out set if desired
        f.write("nc: 2\n")
        f.write("names: ['cube', 'target_zone']\n")
    print(f"Written {yaml_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_frames', type=int, default=5000)
    parser.add_argument('--out', default='data/yolo_dataset')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    generate(args.n_frames, args.out, args.seed)
