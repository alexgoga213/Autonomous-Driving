"""Shared constants and helpers used by the training, evaluation and inference scripts."""

from __future__ import annotations

import random
from pathlib import Path

# Class order must match the dataset's data.yaml.
CLASS_NAMES: tuple[str, ...] = (
    "Green Light",
    "Red Light",
    "Speed Limit 10",
    "Speed Limit 100",
    "Speed Limit 110",
    "Speed Limit 120",
    "Speed Limit 20",
    "Speed Limit 30",
    "Speed Limit 40",
    "Speed Limit 50",
    "Speed Limit 60",
    "Speed Limit 70",
    "Speed Limit 80",
    "Speed Limit 90",
    "Stop",
)

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_IMG_SIZE = 640
DEFAULT_CONF = 0.25

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"


def select_device(preferred: str | None = None) -> str:
    """Return the best available device: CUDA GPU, then Apple MPS, then CPU.

    `preferred` (e.g. "cpu", "0", "mps") overrides auto-detection.
    """
    if preferred:
        return preferred
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int = 42) -> None:
    """Make runs reproducible across Python, NumPy and PyTorch."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
