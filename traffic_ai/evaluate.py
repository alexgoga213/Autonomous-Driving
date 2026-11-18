"""Evaluate trained weights and export a per-class report (CSV + Markdown).

Example:
    python -m traffic_ai.evaluate --weights runs/traffic_signs/weights/best.pt --data configs/data.yaml
    python -m traffic_ai.evaluate --weights best.pt --data configs/data.yaml --split test
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from traffic_ai.config import DEFAULT_IMG_SIZE, RUNS_DIR, select_device


@dataclass
class ClassReport:
    name: str
    precision: float
    recall: float
    f1: float
    map50: float
    map50_95: float


def f1_score(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def to_markdown(rows: list[ClassReport], overall: ClassReport) -> str:
    lines = [
        "| Class | Precision | Recall | F1 | mAP@50 | mAP@50-95 |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: r.map50, reverse=True):
        lines.append(
            f"| {r.name} | {r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} | {r.map50:.3f} | {r.map50_95:.3f} |"
        )
    o = overall
    lines.append(
        f"| **all** | **{o.precision:.3f}** | **{o.recall:.3f}** | **{o.f1:.3f}** "
        f"| **{o.map50:.3f}** | **{o.map50_95:.3f}** |"
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE)
    p.add_argument("--conf", type=float, default=0.001, help="low conf for mAP; raise it to mimic deployment")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold (Ultralytics default)")
    p.add_argument("--device")
    p.add_argument("--out", type=Path, default=RUNS_DIR / "eval")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=select_device(args.device),
        plots=True,
        project=str(args.out.parent),
        name=args.out.name,
        exist_ok=True,
    )

    box = metrics.box
    rows: list[ClassReport] = []
    for i, class_id in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        rows.append(ClassReport(metrics.names[int(class_id)], p, r, f1_score(p, r), ap50, ap))

    overall = ClassReport("all", box.mp, box.mr, f1_score(box.mp, box.mr), box.map50, box.map)

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / f"per_class_{args.split}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(overall)))
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows + [overall])

    md = to_markdown(rows, overall)
    (args.out / f"per_class_{args.split}.md").write_text(md, encoding="utf-8")

    print(md)
    print(f"\n[evaluate] reports saved in {args.out}")
    weakest = min(rows, key=lambda r: r.map50, default=None)
    if weakest:
        print(f"[evaluate] weakest class: {weakest.name} (mAP@50 {weakest.map50:.3f}): consider more data/augmentation")


if __name__ == "__main__":
    main()
