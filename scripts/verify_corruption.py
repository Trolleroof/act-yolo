"""Visualize the corruption pipeline at all 4 severity levels.

Renders a clean 480x480 frame from MuJoCo, applies corruption at severities
0-3, and saves a 2x2 grid to /tmp/corruption_viz.png.
"""
import sys
import os

# Make sure project root is on the path so vision.corruption is importable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import cv2
import mujoco

from vision.corruption import corrupt_frame

XML_PATH = os.path.join(project_root, "assets", "pick_place.xml")
OUTPUT_PATH = "/tmp/corruption_viz.png"

LABELS = ["clean", "low", "medium", "high"]


def render_clean_frame() -> np.ndarray:
    """Render a single 480x480 RGB frame from the top camera."""
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # Set arm pose
    data.qpos[:8] = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]
    data.ctrl[:8] = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

    # Place cube
    data.qpos[8:11] = [-0.05, 0.55, 0.05]
    data.qpos[11:15] = [1, 0, 0, 0]

    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=480)
    renderer.update_scene(data, camera=model.camera("top").id)
    rgb = renderer.render()

    return rgb  # uint8 (H, W, 3) RGB


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    """Overlay a label in the top-left corner of an image."""
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    padding = 6

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    # Dark semi-transparent background rectangle
    cv2.rectangle(
        img,
        (padding, padding),
        (padding + tw + padding, padding + th + baseline + padding),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(
        img,
        text,
        (padding * 2, padding + th),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return img


def main():
    print("Rendering clean frame from MuJoCo...")
    clean_rgb = render_clean_frame()
    print(f"  Frame shape: {clean_rgb.shape}, dtype: {clean_rgb.dtype}")

    rng = np.random.default_rng(42)

    print("Applying corruption at severities 0, 1, 2, 3...")
    corrupted = []
    for severity in range(4):
        frame = corrupt_frame(clean_rgb, severity=severity, rng=rng)
        frame = add_label(frame, LABELS[severity])
        corrupted.append(frame)
        print(f"  severity={severity} ({LABELS[severity]}): done")

    # Build 2x2 grid: row-major order  [clean, low / medium, high]
    top_row = np.concatenate([corrupted[0], corrupted[1]], axis=1)  # (480, 960, 3)
    bot_row = np.concatenate([corrupted[2], corrupted[3]], axis=1)
    grid = np.concatenate([top_row, bot_row], axis=0)               # (960, 960, 3)

    # cv2 expects BGR
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    cv2.imwrite(OUTPUT_PATH, grid_bgr)
    print(f"\nSaved 2x2 grid ({grid.shape[1]}x{grid.shape[0]}) to {OUTPUT_PATH}")
    print(f"open {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
