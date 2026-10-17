"""Part B2: show sample boxes of one source class, to check what it really contains.

Usage:
    python scripts/box_contact_sheet.py battery_and_vape battery          # 20 boxes
    python scripts/box_contact_sheet.py battery_and_vape battery --n 40 --seed 1

Picks at most one box per image (so one crowded image cannot fill the sheet),
from all splits, crops it with some context and arranges the crops in a grid.
Each caption is the split and the start of the source filename.

Output: reports/external/<name>__<class>__boxes.jpg
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from akkunet.config import load_config
from akkunet.external import find_split_images, label_path_for, read_boxes, read_class_names
from akkunet.sheets import crop_box, grid, tile

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name")
    parser.add_argument("source_class", help="class name exactly as in the export's data.yaml")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default=REPO_ROOT / "config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    export_dir = cfg["paths"]["external"] / args.name
    names = read_class_names(export_dir)
    if args.source_class not in names:
        print(f"'{args.source_class}' is not a class of {args.name}. Classes: {names}")
        return 1
    cls_id = names.index(args.source_class)

    candidates = []  # (split, image path, box)
    for rf_split, images in find_split_images(export_dir).items():
        for p in images:
            boxes = [b for b in read_boxes(label_path_for(p))[0] if b[0] == cls_id]
            if boxes:
                candidates.append((rf_split, p, boxes))
    if not candidates:
        print(f"No '{args.source_class}' boxes found.")
        return 1

    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(candidates), size=min(args.n, len(candidates)), replace=False)
    tiles = []
    for i in picks:
        rf_split, p, boxes = candidates[i]
        box = boxes[rng.integers(len(boxes))]
        tiles.append(tile(crop_box(cv2.imread(str(p)), *box[1:]), f"{rf_split} {p.stem[:26]}"))

    out = cfg["paths"]["reports"] / "external" / f"{args.name}__{args.source_class}__boxes.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"{len(candidates)} images contain '{args.source_class}'. Sheet of {len(tiles)} boxes: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
