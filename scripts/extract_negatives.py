"""Turn the hard-negatives recordings into TRAIN-ONLY frames with EMPTY labels.

Usage:
    python scripts/extract_negatives.py                 # all data/raw_negatives/negatives_*/ sessions
    python scripts/extract_negatives.py --every-n 10    # sample more often than config (default 15)

Same filters as Stage 1 (every N-th frame, blur, near-duplicates). Rebuilds
data/negatives/ each time. Writes a contact sheet to
data/negatives/contact_sheet.jpg (git-ignored: it shows people). Check it:
no battery may be visible in any frame.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from akkunet.config import load_config
from akkunet.frames import summary_table
from akkunet.negatives import extract_negatives
from akkunet.sheets import grid, tile

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--every-n", type=int, help="override frame_extraction.every_n_frames")
    parser.add_argument("--sheet-tiles", type=int, default=80, help="frames on the contact sheet (evenly spaced)")
    parser.add_argument("--config", default=REPO_ROOT / "config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fe = dict(cfg["frame_extraction"])
    if args.every_n:
        fe["every_n_frames"] = args.every_n
    out_dir = cfg["paths"]["negatives"]
    try:
        stats, rows = extract_negatives(cfg["paths"]["raw_negatives"], out_dir, fe)
    except ValueError as e:
        print(e)
        return 1
    if not rows:
        print(f"No recordings in {cfg['paths']['raw_negatives']}/negatives_<date>/. "
              "Record first: python scripts/record_session.py")
        return 1
    print(summary_table(stats))

    picks = np.linspace(0, len(rows) - 1, min(args.sheet_tiles, len(rows))).astype(int)
    tiles = [tile(cv2.imread(str(out_dir / rows[i]["image"])), Path(rows[i]["image"]).stem[-22:], size=200)
             for i in picks]
    sheet = out_dir / "contact_sheet.jpg"
    cv2.imwrite(str(sheet), grid(tiles, cols=10), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"\n{len(rows)} negative frames (empty labels, train only) in {out_dir}")
    print(f"Contact sheet ({len(tiles)} evenly spaced frames): {sheet}  <- check: NO battery visible")
    return 0


if __name__ == "__main__":
    sys.exit(main())
