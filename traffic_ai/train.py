"""Fine-tune a YOLOv8 model on the traffic sign & light dataset.

Example:
    python -m traffic_ai.train --data configs/data.yaml
    python -m traffic_ai.train --data configs/data.yaml --model yolov8s.pt --epochs 100 --device 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from traffic_ai.config import DEFAULT_MODEL, RUNS_DIR, seed_everything, select_device


def load_hyperparameters(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="dataset YAML (see configs/data.example.yaml)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="pretrained weights, e.g. yolov8n.pt / yolov8s.pt")
    p.add_argument("--config", type=Path, default=Path("configs/train.yaml"), help="hyperparameter YAML")
    p.add_argument("--epochs", type=int, help="override epochs from the config")
    p.add_argument("--imgsz", type=int, help="override image size from the config")
    p.add_argument("--batch", type=int, help="override batch size (-1 = auto)")
    p.add_argument("--device", help="cpu, 0 (first GPU), mps ... (default: auto)")
    p.add_argument("--name", default="traffic_signs", help="run name inside runs/")
    p.add_argument("--resume", type=Path, metavar="LAST_PT", help="resume an interrupted run from its last.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise SystemExit(f"Dataset config not found: {args.data}")

    hyp = load_hyperparameters(args.config if args.config.exists() else None)
    for key in ("epochs", "imgsz", "batch"):
        value = getattr(args, key)
        if value is not None:
            hyp[key] = value

    seed_everything(int(hyp.get("seed", 42)))
    device = select_device(args.device)
    print(f"[train] model={args.model} device={device} hyperparameters={hyp}")

    from ultralytics import YOLO  # imported late so `--help` works without the heavy dependency

    if args.resume:
        # Ultralytics restores data, hyperparameters and optimizer state from the checkpoint.
        model = YOLO(str(args.resume))
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(
            data=str(args.data),
            device=device,
            project=str(RUNS_DIR),
            name=args.name,
            exist_ok=False,
            **hyp,
        )

    best = Path(model.trainer.best)
    print(f"[train] done. Best weights: {best}")
    print(f"[train] evaluate with: python -m traffic_ai.evaluate --weights {best} --data {args.data}")


if __name__ == "__main__":
    main()
