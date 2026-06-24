"""
Quick sanity check for the pick-place env.

Usage:
    # Save camera frames as PNGs (no display needed):
    python scripts/verify_env.py --save

    # Open the MuJoCo interactive viewer (needs a display):
    python scripts/verify_env.py --viewer

    # Both:
    python scripts/verify_env.py --save --viewer
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault('MUJOCO_GL', 'glfw')  # use GLFW on Mac; set EGL on headless Linux

parser = argparse.ArgumentParser()
parser.add_argument('--save', action='store_true', help='Save camera frames to /tmp')
parser.add_argument('--viewer', action='store_true', help='Launch MuJoCo interactive viewer')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

if not args.save and not args.viewer:
    args.save = True  # default to saving

from envs.pick_place import make_pick_place_env, PickPlaceTask, TARGET_ZONE_POS

env = make_pick_place_env(random_seed=args.seed)
ts = env.reset()
obs = ts.observation

print(f"qpos (7):    {obs['qpos'].round(3)}")
print(f"env_state:   cube pos={obs['env_state'][:3].round(3)}")
print(f"target zone: {TARGET_ZONE_POS}")
print(f"top frame:   {obs['top'].shape}  dtype={obs['top'].dtype}")
print(f"wrist frame: {obs['wrist'].shape} dtype={obs['wrist'].dtype}")

# ── Save frames ────────────────────────────────────────────────────────────────
if args.save:
    import cv2
    out_dir = '/tmp/act_yolo_verify'
    os.makedirs(out_dir, exist_ok=True)

    top_bgr   = cv2.cvtColor(obs['top'],   cv2.COLOR_RGB2BGR)
    wrist_bgr = cv2.cvtColor(obs['wrist'], cv2.COLOR_RGB2BGR)

    top_path   = os.path.join(out_dir, 'top.png')
    wrist_path = os.path.join(out_dir, 'wrist.png')
    cv2.imwrite(top_path,   top_bgr)
    cv2.imwrite(wrist_path, wrist_bgr)

    print(f"\nSaved frames:")
    print(f"  open {top_path}")
    print(f"  open {wrist_path}")

# ── Interactive viewer ─────────────────────────────────────────────────────────
if args.viewer:
    import mujoco
    import mujoco.viewer

    XML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'pick_place.xml')
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)

    print("\nLaunching viewer — close window to exit.")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Run sim forward slowly so you can watch
        for _ in range(10_000):
            mujoco.mj_step(m, d)
            viewer.sync()
            if not viewer.is_running():
                break
