"""
Corruption-augmented YOLO training (Option A — in-pipeline domain randomization).

The eval pipeline (`vision/corruption.py`) corrupts camera frames at severities
0-3 before they ever reach YOLO. A YOLO model trained only on clean MuJoCo
renders therefore sees a hard train/eval distribution gap and collapses on the
small `cube` object at severity 2/3 (blur + heavy JPEG erase its high-frequency
detail while the large flat `target_zone` survives).

This module closes that gap by reusing the *exact same* `corrupt_frame` operator
the evaluator uses, applied on-the-fly to every training image during YOLO
training. Object positions are unchanged by corruption, so the auto-generated
labels remain valid.

It is implemented as an idempotent monkeypatch of Ultralytics'
`BaseDataset.load_image` so it needs no extra dependency (albumentations is not
installed) and slots cleanly into the existing `train_yolo.py` flow. Corruption
is only applied when the dataset is in augmentation mode (`self.augment`), so
validation frames stay clean.
"""
import os
import sys

import numpy as np

# Reuse the evaluator's corruption operator verbatim — single source of truth.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.corruption import corrupt_frame, SEVERITY_PRESETS  # noqa: E402

_PATCHED = False


def enable_corruption_aug(severities=(0, 1, 2, 3),
                          weights=(0.25, 0.25, 0.25, 0.25),
                          p: float = 0.85,
                          seed: int | None = None) -> None:
    """Monkeypatch Ultralytics so training images are randomly corrupted.

    Args:
        severities: severity levels to sample from (must be keys of SEVERITY_PRESETS).
        weights:    sampling weights over `severities` (normalized internally).
        p:          probability a given training image is corrupted at all.
        seed:       optional RNG seed for reproducible augmentation.

    Idempotent: calling twice is a no-op. Only affects datasets with
    `self.augment is True` (i.e. the training split, not validation).
    """
    global _PATCHED
    if _PATCHED:
        return

    for s in severities:
        if s not in SEVERITY_PRESETS:
            raise ValueError(f"Unknown severity {s}; valid: {list(SEVERITY_PRESETS)}")

    from ultralytics.data.base import BaseDataset

    severities = np.asarray(severities)
    probs = np.asarray(weights, dtype=np.float64)
    probs = probs / probs.sum()
    rng = np.random.default_rng(seed)

    _orig_load_image = BaseDataset.load_image

    def load_image_corrupt(self, i, *args, **kwargs):
        im, hw0, hw = _orig_load_image(self, i, *args, **kwargs)
        # Only corrupt the training split; never touch val/inference frames.
        if getattr(self, "augment", False) and im is not None and im.ndim == 3:
            if rng.random() < p:
                sev = int(rng.choice(severities, p=probs))
                if sev > 0:
                    # corrupt_frame is channel-order agnostic (noise/blur/
                    # brightness/contrast are symmetric; the JPEG round-trip
                    # preserves channel order), so applying it to Ultralytics'
                    # BGR images is safe.
                    im = corrupt_frame(im, sev, rng=rng)
        return im, hw0, hw

    BaseDataset.load_image = load_image_corrupt
    _PATCHED = True
    print(f"[corruption_aug] Enabled: severities={[int(s) for s in severities]} "
          f"weights={probs.round(3).tolist()} p={p}")
