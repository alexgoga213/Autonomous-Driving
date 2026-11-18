"""Run the detector + decision engine on a video, a webcam, an image or a folder of images.

The output video/images contain the YOLO boxes plus a HUD (heads-up display) with the current
driving command, the active speed limit and the FPS.

Examples:
    python -m traffic_ai.predict --weights best.pt --source video.avi
    python -m traffic_ai.predict --weights best.pt --source 0 --show          # webcam
    python -m traffic_ai.predict --weights best.pt --source path/to/images/
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from traffic_ai.config import DEFAULT_CONF, DEFAULT_IMG_SIZE, RUNS_DIR, select_device
from traffic_ai.decision import Action, DecisionConfig, DecisionEngine, Detection, DrivingDecision

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ACTION_COLORS = {  # BGR
    Action.STOP: (40, 40, 220),
    Action.PROCEED: (60, 190, 60),
    Action.CRUISE: (200, 160, 40),
}


# ---------------------------------------------------------------------------- helpers
def to_detections(result) -> list[Detection]:
    """Convert an Ultralytics `Results` object into plain `Detection`s."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    return [
        Detection(result.names[c], float(p), tuple(float(v) for v in b))
        for b, p, c in zip(xyxy, confs, classes)
    ]


def draw_hud(frame: np.ndarray, decision: DrivingDecision, fps: float | None = None) -> np.ndarray:
    """Draw a semi-transparent panel with the driving command on top of the frame."""
    h, w = frame.shape[:2]
    scale = max(0.5, w / 1280)
    panel_h = int(70 * scale)
    color = ACTION_COLORS[decision.action]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    cv2.rectangle(frame, (0, panel_h - int(4 * scale)), (w, panel_h), color, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, decision.action.value, (int(15 * scale), int(45 * scale)), font,
                1.3 * scale, color, max(2, int(3 * scale)), cv2.LINE_AA)
    limit = f"{decision.speed_limit} km/h" if decision.speed_limit is not None else "--"
    cv2.putText(frame, f"LIMIT {limit}", (int(220 * scale), int(30 * scale)), font,
                0.7 * scale, (255, 255, 255), max(1, int(2 * scale)), cv2.LINE_AA)
    cv2.putText(frame, decision.reason, (int(220 * scale), int(58 * scale)), font,
                0.55 * scale, (210, 210, 210), 1, cv2.LINE_AA)
    if fps is not None:
        text = f"{fps:4.1f} FPS"
        (tw, _), _ = cv2.getTextSize(text, font, 0.6 * scale, 1)
        cv2.putText(frame, text, (w - tw - int(15 * scale), int(30 * scale)), font,
                    0.6 * scale, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def is_image_source(source: str) -> bool:
    path = Path(source)
    return path.is_dir() or path.suffix.lower() in IMAGE_SUFFIXES


# ---------------------------------------------------------------------------- runners
def run_images(model, source: Path, args: argparse.Namespace, out_dir: Path) -> None:
    files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) if source.is_dir() else [source]
    # Single images have no temporal context: confirm on the first sighting, no stop hold.
    engine = DecisionEngine(DecisionConfig(window=1, min_hits=1, stop_hold_frames=0))
    for path in files:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"[predict] skipped unreadable file {path}")
            continue
        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        engine.reset()
        decision = engine.update(to_detections(result), frame_size=(frame.shape[1], frame.shape[0]))
        annotated = draw_hud(result.plot(line_width=2), decision)
        cv2.imwrite(str(out_dir / f"{path.stem}_pred.jpg"), annotated)
        print(f"{path.name:<40} {decision}")


def run_stream(model, source: str, args: argparse.Namespace, out_dir: Path) -> None:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source: {source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    name = "webcam" if source.isdigit() else Path(source).stem
    out_path = out_dir / f"{name}_pred.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height))

    engine = DecisionEngine(DecisionConfig(stop_hold_frames=int(fps_in * args.stop_seconds)))
    actions: Counter[Action] = Counter()
    last_action: Action | None = None
    smoothed_fps: float | None = None
    start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
            decision = engine.update(to_detections(result), frame_size=(width, height))

            dt = time.perf_counter() - t0
            inst = 1.0 / dt if dt > 0 else 0.0
            smoothed_fps = inst if smoothed_fps is None else 0.9 * smoothed_fps + 0.1 * inst

            annotated = draw_hud(result.plot(line_width=2), decision, smoothed_fps)
            writer.write(annotated)
            actions[decision.action] += 1

            if decision.action != last_action:  # log only state changes, not every frame
                print(f"[frame {engine.frame_index:>5}] {decision}")
                last_action = decision.action

            if args.show:
                cv2.imshow("traffic_ai", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    frames = engine.frame_index
    elapsed = time.perf_counter() - start
    print(f"\n[predict] {frames} frames in {elapsed:.1f}s ({frames / max(elapsed, 1e-9):.1f} FPS avg)")
    for action, n in actions.most_common():
        print(f"[predict]   {action.value:<8} {n:>6} frames ({100 * n / max(frames, 1):.1f}%)")
    print(f"[predict] annotated video saved to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--source", required=True, help="video file, webcam index (0), image or image folder")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF, help="detector confidence (decision layer filters further)")
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE)
    p.add_argument("--device")
    p.add_argument("--stop-seconds", type=float, default=1.0, help="how long to hold a mandatory stop")
    p.add_argument("--show", action="store_true", help="display a live window (press q to quit)")
    p.add_argument("--out", type=Path, default=RUNS_DIR / "predict")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.device = select_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    if is_image_source(args.source):
        run_images(model, Path(args.source), args, args.out)
    else:
        run_stream(model, args.source, args, args.out)


if __name__ == "__main__":
    main()
