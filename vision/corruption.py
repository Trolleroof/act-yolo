import cv2
import numpy as np

# severity index for plotting: 0=clean, 1=low, 2=medium, 3=high
SEVERITY_PRESETS = {
    0: dict(noise_std=0.0,  blur_ksize=0, brightness_delta=0.0,  contrast_scale=1.0,  jpeg_quality=100),
    1: dict(noise_std=8.0,  blur_ksize=3, brightness_delta=0.08, contrast_scale=0.92, jpeg_quality=80),
    2: dict(noise_std=18.0, blur_ksize=5, brightness_delta=0.15, contrast_scale=0.85, jpeg_quality=55),
    3: dict(noise_std=30.0, blur_ksize=7, brightness_delta=0.25, contrast_scale=0.75, jpeg_quality=35),
}

SEVERITY_ALIASES = {'clean': 0, 'low': 1, 'medium': 2, 'high': 3, 'light': 1, 'heavy': 3}


def corrupt_frame(rgb: np.ndarray, severity: int | str = 0,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply visual corruption to a single frame. Eval time only — never call during training.

    Args:
        rgb: (H, W, 3) uint8 RGB frame from MuJoCo render.
        severity: 0–3 or alias ('clean', 'low', 'medium', 'high', 'light', 'heavy').
        rng: optional RNG for reproducible eval seeds.

    Returns:
        Corrupted uint8 RGB frame, same shape as input.
    """
    if isinstance(severity, str):
        severity = SEVERITY_ALIASES[severity.lower()]

    if severity == 0:
        return rgb

    p = SEVERITY_PRESETS[severity]
    rng = rng or np.random.default_rng()
    out = rgb.astype(np.float32)

    if p['noise_std'] > 0:
        out += rng.normal(0, p['noise_std'], out.shape)

    out = np.clip(out, 0, 255).astype(np.uint8)

    k = p['blur_ksize']
    if k >= 3:
        out = cv2.GaussianBlur(out, (k, k), 0)

    out = out.astype(np.float32)
    out = out * p['contrast_scale'] + p['brightness_delta'] * 255.0
    out = np.clip(out, 0, 255).astype(np.uint8)

    if p['jpeg_quality'] < 100:
        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), p['jpeg_quality']])
        out = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    return out


def corrupt_obs_images(obs: dict, severity: int | str, rng=None) -> dict:
    """Corrupt top and wrist camera keys in an observation dict. Eval time only."""
    obs = dict(obs)
    for cam in ('top', 'wrist'):
        if cam in obs:
            obs[cam] = corrupt_frame(obs[cam], severity, rng=rng)
    return obs
