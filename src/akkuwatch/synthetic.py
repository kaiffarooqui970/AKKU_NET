"""Synthetic videos for SMOKE TESTS ONLY.

These contain random block patterns, not batteries. They exist purely to
check that the pipeline runs end to end. Never train or evaluate on them.
"""
from pathlib import Path

import cv2
import numpy as np


def _scene(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    """A random coarse block pattern: sharp edges, and every scene looks different."""
    blocks = rng.integers(0, 256, size=(height // 16, width // 16, 3), dtype=np.uint8)
    return cv2.resize(blocks, (width, height), interpolation=cv2.INTER_NEAREST)


def make_synthetic_video(
    path: Path,
    n_scenes: int = 4,
    frames_per_scene: int = 5,
    n_blurry_scenes: int = 1,
    size: tuple[int, int] = (320, 240),
    fps: int = 10,
    seed: int = 0,
) -> None:
    """Write a video made of `n_scenes` distinct still scenes, each held for
    `frames_per_scene` identical frames (-> near-duplicates), followed by
    `n_blurry_scenes` heavily blurred scenes (-> should be removed as blurry).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for i in range(n_scenes + n_blurry_scenes):
        frame = _scene(rng, width, height)
        if i >= n_scenes:
            frame = cv2.GaussianBlur(frame, (51, 51), 0)
        for _ in range(frames_per_scene):
            writer.write(frame)
    writer.release()
