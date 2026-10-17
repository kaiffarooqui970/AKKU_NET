"""Record a webcam session to an MP4 (no model, CPU only).

Made for the hard-negatives session: faces, empty hands and lookalikes
(Tic Tac box, unlit lighter, phone, marker, USB stick, glue stick) with NO
batteries in view. The file lands in data/raw_negatives/<session>/, which is
git-ignored (it shows people) and is only ever used as TRAINING negatives.

Usage:
    python scripts/record_session.py                       # session negatives_<today>, max 10 min
    python scripts/record_session.py --minutes 8 --session negatives_2026-10-17_desk

Window keys: q or Esc stops early. A red "REC mm:ss" shows the elapsed time;
the overlay is only on screen, not in the recording.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

from akkunet.config import load_config
from akkunet.negatives import NEGATIVE_SESSION_PATTERN

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", default=f"negatives_{time.strftime('%Y-%m-%d')}")
    parser.add_argument("--minutes", type=float, default=10.0, help="stop automatically after this long")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--config", default=REPO_ROOT / "config.yaml")
    args = parser.parse_args()

    if not NEGATIVE_SESSION_PATTERN.match(args.session):
        print("Session must be named negatives_<YYYY-MM-DD>[_note], e.g. negatives_2026-10-17_desk")
        return 1
    cfg = load_config(args.config)
    out_dir = cfg["paths"]["raw_negatives"] / args.session
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"rec_{time.strftime('%H%M%S')}.mp4"

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Could not open the camera. Allow camera access for your terminal "
              "(System Settings > Privacy & Security > Camera).")
        return 1
    ok, frame = cap.read()
    if not ok:
        print("Camera opened but returned no image.")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
    print(f"Recording to {out} (max {args.minutes:g} min). NO batteries in view! q / Esc stops.")

    start, frames = time.time(), 0
    try:
        while ok:
            writer.write(frame)
            frames += 1
            elapsed = time.time() - start
            shown = frame.copy()
            cv2.circle(shown, (24, 26), 9, (0, 0, 255), -1)
            cv2.putText(shown, f"REC {int(elapsed // 60):02d}:{int(elapsed % 60):02d}  no batteries in view!",
                        (40, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("AkkuNet recording", shown)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27) or elapsed >= args.minutes * 60:
                break
            ok, frame = cap.read()
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    print(f"Saved {frames} frames ({(time.time() - start) / 60:.1f} min): {out}")
    print("Next: python scripts/extract_negatives.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
