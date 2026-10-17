"""Part B3: live demo. Webcam or video -> YOLO + ByteTrack -> boxes, track IDs, alerts.

Usage:
    python -m akkunet.app                                  # webcam 0, weights from config.yaml
    python -m akkunet.app --source clip.mp4 --record out.mp4
    python -m akkunet.app --weights yolo11n.pt --alert-classes person   # check with a stock COCO model

Keys in the window: q or Esc quits.

Alert rule (config `demo`): a track alerts ONCE, when it has been seen in at
least `alert_min_frames` frames with confidence >= `alert_conf`. Frames need
not be consecutive: ByteTrack keeps the ID through short gaps, and a brief
dip in confidence shouldn't reset the count.
"""
import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from akkunet.config import load_config
from akkunet.device import get_device

REPO_ROOT = Path(__file__).resolve().parents[2]
BANNER_TEXT = "Possible battery detected"
RED = (0, 0, 230)
# BGR colours per class ID; alerted tracks are always drawn red.
PALETTE = [(255, 160, 0), (0, 200, 255), (200, 0, 200), (0, 200, 0), (255, 255, 0), (128, 128, 255)]


@dataclass
class Detection:
    box: tuple[int, int, int, int]   # x0, y0, x1, y1 in pixels
    cls: int
    conf: float
    track_id: int | None             # None while ByteTrack has not confirmed the track


@dataclass
class Alert:
    track_id: int
    cls: int
    conf: float
    frame_index: int


@dataclass
class AlertTracker:
    """Counts qualifying frames per track and fires one alert per track."""
    conf_threshold: float
    min_frames: int
    alert_classes: set[int] | None = None          # None = all classes
    hits: dict[int, int] = field(default_factory=dict)
    alerted: set[int] = field(default_factory=set)

    def update(self, detections: list[Detection], frame_index: int) -> list[Alert]:
        new = []
        for d in detections:
            if d.track_id is None or d.conf < self.conf_threshold:
                continue
            if self.alert_classes is not None and d.cls not in self.alert_classes:
                continue
            self.hits[d.track_id] = self.hits.get(d.track_id, 0) + 1
            if self.hits[d.track_id] >= self.min_frames and d.track_id not in self.alerted:
                self.alerted.add(d.track_id)
                new.append(Alert(d.track_id, d.cls, d.conf, frame_index))
        return new


def detections_from_result(result) -> list[Detection]:
    """Ultralytics tracking result -> list of Detection."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
    return [Detection(tuple(int(v) for v in xyxy), int(c), float(p), tid)
            for xyxy, c, p, tid in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), ids)]


def draw(frame: np.ndarray, detections: list[Detection], names: dict[int, str], alerted: set[int],
         banner: str | None, status: str) -> np.ndarray:
    """Boxes with '#id class conf', a red banner while an alert is active, and a status line."""
    out = frame.copy()
    h, w = out.shape[:2]
    thick = max(2, w // 400)
    for d in detections:
        colour = RED if d.track_id in alerted else PALETTE[d.cls % len(PALETTE)]
        x0, y0, x1, y1 = d.box
        cv2.rectangle(out, (x0, y0), (x1, y1), colour, thick)
        label = f"{'#' + str(d.track_id) + ' ' if d.track_id is not None else ''}{names.get(d.cls, d.cls)} {d.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(y0, th + 6)
        cv2.rectangle(out, (x0, ty - th - 6), (x0 + tw + 6, ty), colour, -1)
        cv2.putText(out, label, (x0 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if banner:
        bh = max(40, h // 12)
        cv2.rectangle(out, (0, 0), (w, bh), RED, -1)
        cv2.putText(out, banner, (12, int(bh * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, bh / 50, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, status, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def parse_source(source):
    """'0' -> webcam 0; anything else is a file path."""
    return int(source) if str(source).isdigit() else str(source)


def resolve_classes(names: dict[int, str], wanted: list[str]) -> set[int] | None:
    if not wanted:
        return None
    lookup = {v: k for k, v in names.items()}
    unknown = [c for c in wanted if c not in lookup]
    if unknown:
        raise ValueError(f"Unknown alert classes {unknown}. Model classes: {list(names.values())}")
    return {lookup[c] for c in wanted}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=REPO_ROOT / "config.yaml")
    parser.add_argument("--weights", help="overrides demo.weights")
    parser.add_argument("--source", help="webcam index or video path (overrides demo.source)")
    parser.add_argument("--record", help="write the annotated output to this .mp4")
    parser.add_argument("--alert-conf", type=float)
    parser.add_argument("--alert-min-frames", type=int)
    parser.add_argument("--alert-classes", nargs="*", help="class names that may alert (default: config/all)")
    parser.add_argument("--device", help="cpu / mps / cuda (default: config device)")
    parser.add_argument("--no-window", action="store_true", help="don't open a window (for files/tests)")
    parser.add_argument("--max-frames", type=int, help="stop after this many frames")
    args = parser.parse_args(argv)

    from ultralytics import YOLO  # slow import

    cfg = load_config(args.config)
    demo = cfg["demo"]
    weights = Path(args.weights) if args.weights else cfg["config_path"].parent / demo["weights"]
    if args.weights is None and not weights.exists():
        print(f"Weights not found: {weights}. Train first or pass --weights.")
        return 1
    model = YOLO(str(weights))
    names = model.names
    try:
        alert_classes = resolve_classes(names, args.alert_classes if args.alert_classes is not None
                                        else demo.get("alert_classes") or [])
    except ValueError as e:
        print(e)
        return 1
    alerts = AlertTracker(args.alert_conf if args.alert_conf is not None else demo["alert_conf"],
                          args.alert_min_frames or demo["alert_min_frames"], alert_classes)
    device = get_device(args.device or cfg["device"])

    source = parse_source(args.source if args.source is not None else demo["source"])
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open source {source!r}. On macOS, allow camera access for your terminal "
              "(System Settings > Privacy & Security > Camera).")
        return 1
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    frame_index, banner_until, banner = 0, 0.0, None
    t_last, fps = time.time(), 0.0
    print(f"Model {weights.name} ({len(names)} classes) on {device}; alert at conf >= {alerts.conf_threshold} "
          f"for >= {alerts.min_frames} frames. Press q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = model.track(frame, persist=True, tracker=demo["tracker"], conf=demo["detect_conf"],
                                 imgsz=demo["imgsz"], device=device, verbose=False)[0]
            detections = detections_from_result(result)
            for a in alerts.update(detections, frame_index):
                banner = f"{BANNER_TEXT}: {names[a.cls]} #{a.track_id}"
                banner_until = time.time() + demo["banner_seconds"]
                print(f"{time.strftime('%H:%M:%S')} ALERT frame {a.frame_index}: {names[a.cls]} "
                      f"track #{a.track_id} conf {a.conf:.2f}", flush=True)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6) if fps else 1 / max(now - t_last, 1e-6)
            t_last = now
            status = f"{fps:4.1f} fps | alert conf>={alerts.conf_threshold} x{alerts.min_frames} frames | " \
                     f"alerts so far: {len(alerts.alerted)}"
            shown = draw(frame, detections, names, alerts.alerted, banner if now < banner_until else None, status)

            if args.record:
                if writer is None:
                    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), fps_in,
                                             (shown.shape[1], shown.shape[0]))
                writer.write(shown)
            if not args.no_window:
                cv2.imshow("AkkuNet demo", shown)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            frame_index += 1
            if args.max_frames and frame_index >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_window:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
    print(f"{frame_index} frames, {len(alerts.alerted)} alert(s)."
          + (f" Recording: {args.record}" if args.record else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
