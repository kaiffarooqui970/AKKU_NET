"""Part B2 step 5: face-shortcut check. How often is a vape shown next to a person?

If most vape images show someone vaping, the model may learn "thing near a
face" instead of "vape". This runs the COCO-pretrained YOLO person detector
over every image that has a vape box and reports:
  - how many images contain a person (confidence >= --conf)
  - how large the largest person is (share of the image area)
  - how many vape boxes have their centre inside a person box
Nothing is removed. A per-image CSV is written so a rule can be applied later.

Usage:
    python scripts/check_vape_person.py battery_and_vape vape

Output:
    reports/external/<name>_person_check.md / .csv / .jpg (contact sheet per size bucket)
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from akkunet.config import load_config
from akkunet.device import get_device
from akkunet.external import find_split_images, label_path_for, read_boxes, read_class_names
from akkunet.sheets import grid, tile

REPO_ROOT = Path(__file__).resolve().parents[1]
COCO_PERSON = 0
# Largest person's share of the image area -> bucket name.
BUCKETS = [(0.0, "no person"), (1e-9, "small (<10%)"), (0.10, "medium (10-40%)"), (0.40, "large (>=40%)")]


def bucket(area: float, has_person: bool) -> str:
    if not has_person:
        return BUCKETS[0][1]
    return [name for lo, name in BUCKETS[1:] if area >= lo][-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name")
    parser.add_argument("source_class", help="the vape class name exactly as in data.yaml")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--per-bucket", type=int, default=8, help="examples per bucket on the contact sheet")
    parser.add_argument("--config", default=REPO_ROOT / "config.yaml")
    args = parser.parse_args()

    from ultralytics import YOLO  # slow import, only needed here

    cfg = load_config(args.config)
    export_dir = cfg["paths"]["external"] / args.name
    cls_id = read_class_names(export_dir).index(args.source_class)
    images = [(s, p, [b for b in read_boxes(label_path_for(p))[0] if b[0] == cls_id])
              for s, ps in find_split_images(export_dir).items() for p in ps]
    images = [x for x in images if x[2]]
    print(f"{len(images)} images with '{args.source_class}' boxes. Running person detector ...", flush=True)

    model = YOLO("yolo11n.pt")
    device = get_device(cfg["device"])
    rows = []
    for rf_split, path, vapes in images:
        res = model.predict(str(path), classes=[COCO_PERSON], conf=args.conf, device=device, verbose=False)[0]
        persons = res.boxes.xyxyn.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
        areas = (persons[:, 2] - persons[:, 0]) * (persons[:, 3] - persons[:, 1]) if len(persons) else []
        inside = sum(any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in persons)
                     for _, cx, cy, _, _ in vapes)
        largest = float(max(areas)) if len(persons) else 0.0
        rows.append({"split": rf_split, "image": path.name, "persons": len(persons),
                     "largest_person_area": round(largest, 3), "vape_boxes": len(vapes),
                     "vapes_inside_person": inside, "bucket": bucket(largest, len(persons) > 0)})

    out_dir = cfg["paths"]["reports"] / "external"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.name}_person_check.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    counts = Counter(r["bucket"] for r in rows)
    vape_total = sum(r["vape_boxes"] for r in rows)
    vape_inside = sum(r["vapes_inside_person"] for r in rows)
    lines = [f"# Face-shortcut check: {args.name} / {args.source_class}", "",
             f"COCO person detector (yolo11n.pt), confidence >= {args.conf}. {n} images with vape boxes.", "",
             "| largest person in image | images | share |", "|---|---:|---:|"]
    for _, name in BUCKETS:
        lines.append(f"| {name} | {counts[name]} | {counts[name] / n:.0%} |")
    lines += ["", f"Vape boxes with their centre inside a person box: {vape_inside}/{vape_total} "
              f"({vape_inside / max(vape_total, 1):.0%}).", "",
              "Per split (images with any person / images):", ""]
    for s in sorted({r['split'] for r in rows}):
        sr = [r for r in rows if r["split"] == s]
        lines.append(f"- {s}: {sum(r['persons'] > 0 for r in sr)}/{len(sr)}")
    md = "\n".join(lines) + "\n"
    (out_dir / f"{args.name}_person_check.md").write_text(md)

    # Contact sheet: one row per bucket, person boxes blue, vape boxes green.
    rng = np.random.default_rng(0)
    by_path = {p.name: (p, v) for _, p, v in images}
    sheet_rows = []
    for _, name in BUCKETS:
        members = [r for r in rows if r["bucket"] == name]
        if not members:
            continue
        tiles = []
        for r in rng.choice(members, size=min(args.per_bucket, len(members)), replace=False):
            path, vapes = by_path[r["image"]]
            img = cv2.imread(str(path))
            H, W = img.shape[:2]
            res = model.predict(img, classes=[COCO_PERSON], conf=args.conf, device=device, verbose=False)[0]
            for x0, y0, x1, y1 in res.boxes.xyxy.cpu().numpy().astype(int):
                cv2.rectangle(img, (x0, y0), (x1, y1), (255, 120, 0), max(2, W // 200))
            for _, cx, cy, w, h in vapes:
                cv2.rectangle(img, (int((cx - w / 2) * W), int((cy - h / 2) * H)),
                              (int((cx + w / 2) * W), int((cy + h / 2) * H)), (0, 220, 0), max(2, W // 200))
            tiles.append(tile(img, f"{name} {r['largest_person_area']:.2f}"))
        tiles += [np.full_like(tiles[0], 30)] * (args.per_bucket - len(tiles))
        sheet_rows.append(grid(tiles, cols=args.per_bucket))
    cv2.imwrite(str(out_dir / f"{args.name}_person_check.jpg"), np.vstack(sheet_rows),
                [cv2.IMWRITE_JPEG_QUALITY, 88])

    print(md)
    print(f"Wrote {out_dir}/{args.name}_person_check.{{md,csv,jpg}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
