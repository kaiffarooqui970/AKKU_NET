"""Check the D-Fire fire/smoke dataset BEFORE using it (like Part B2's inspect_export).

Usage:
    python scripts/inspect_dfire.py                 # hashes cached in data/external/dfire/hashes.npz
    python scripts/inspect_dfire.py --rehash

Expects data/external/dfire/{train,test}/{images,labels}, splits/ (the official
lists) and data.yaml (names: [smoke, fire]). Reads only; copies no images.

Reports:
  - images vs. the official split lists (missing / extra files)
  - per split: images, empty, fire-only / smoke-only / both, boxes per class,
    image sizes, box sizes (how many are small: relevant for small flames)
  - near-duplicate leakage: test images within N dHash bits of a train image,
    its mirror or a 90° rotation; the same for 5-fold fold 1 (train1 vs valid1)
  - a contact sheet of the closest leak pairs, to judge whether they are real copies

Output: reports/external/dfire_inspection.md, dfire_leaks.csv, dfire__leak_pairs.jpg
"""
import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from akkunet.config import load_config
from akkunet.external import find_split_images, hash_variants, label_path_for, read_boxes, read_class_names
from akkunet.sheets import grid, tile

REPO_ROOT = Path(__file__).resolve().parents[1]
SMALL = 0.01        # box area below 1% of the image counts as "small"
TINY = 0.001        # below 0.1%: a few pixels at 640


def read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def prefix(name: str) -> str:
    return re.match(r"[A-Za-z_]+", name).group()


def load_hashes(images: list[Path], cache: Path, rehash: bool) -> np.ndarray:
    """(N, 4 variants, 4 x uint64) per image, cached by filename order."""
    names = np.array([p.name for p in images])
    if cache.exists() and not rehash:
        data = np.load(cache)
        if np.array_equal(data["names"], names):
            return data["hashes"]
    out = np.empty((len(images), 4, 4), dtype=np.uint64)
    for i, p in enumerate(images):
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"Could not read image {p}")
        out[i] = hash_variants(img)
        if i % 2000 == 0:
            print(f"  hashed {i}/{len(images)}", flush=True)
    np.savez(cache, names=names, hashes=out)
    return out


def nearest(query: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each query image (its plain hash): smallest distance to any reference variant, and which image."""
    best_d = np.full(len(query), 10_000, dtype=np.int64)
    best_i = np.zeros(len(query), dtype=np.int64)
    for start in range(0, len(query), 256):
        block = query[start:start + 256, 0][:, None, :]                       # (b, 1, 4)
        dist = np.min([np.bitwise_count(block ^ ref[None, :, k]).sum(-1) for k in range(4)], axis=0)
        best_d[start:start + 256] = dist.min(1)
        best_i[start:start + 256] = dist.argmin(1)
    return best_d, best_i


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="dfire")
    parser.add_argument("--threshold", type=int, default=None, help="dHash bits (default: bootstrap.near_copy_hamming)")
    parser.add_argument("--rehash", action="store_true")
    parser.add_argument("--config", default=REPO_ROOT / "config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = cfg["paths"]["external"] / args.name
    threshold = args.threshold if args.threshold is not None else cfg["bootstrap"]["near_copy_hamming"]
    names = read_class_names(root)
    splits = find_split_images(root)
    lines = [f"# Dataset inspection: {args.name}", "", f"Folder: `{root}`", "",
             f"Classes (ID order, verified on the contact sheets): {', '.join(f'`{n}`' for n in names)}", ""]

    # --- files vs. official lists
    lines += ["## Files vs. official split lists", ""]
    for split, images in splits.items():
        listed = set(read_list(root / "splits" / f"dfire_{split}.txt"))
        have = {p.name for p in images}
        no_label = sum(not label_path_for(p).exists() for p in images)
        lines.append(f"- {split}: {len(have)} images, {len(listed)} listed, missing {len(listed - have)}, "
                     f"not listed {len(have - listed)}, without label file {no_label}")

    # --- counts
    stats = {}
    for split, images in splits.items():
        s = Counter()
        for p in images:
            boxes, _ = read_boxes(label_path_for(p))
            present = {names[b[0]] for b in boxes}
            s["images"] += 1
            s[prefix(p.name)] += 1
            s["empty" if not present else "+".join(sorted(present))] += 1
            for b in boxes:
                area = b[3] * b[4]
                s[f"box_{names[b[0]]}"] += 1
                s[f"small_{names[b[0]]}"] += area < SMALL
                s[f"tiny_{names[b[0]]}"] += area < TINY
        stats[split] = s
    cols = list(stats)
    lines += ["", "## Split counts", "", "| | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    rows = [("images", "images"), ("empty (no fire, no smoke)", "empty"), ("fire only", "fire"),
            ("smoke only", "smoke"), ("fire + smoke", "fire+smoke")]
    rows += [(f"from {p}", p) for p in ("AoF", "WEB", "PublicDataset")]
    for n in names:
        rows += [(f"{n} boxes", f"box_{n}"), (f"{n} boxes < 1% of image", f"small_{n}"),
                 (f"{n} boxes < 0.1% of image", f"tiny_{n}")]
    for label, key in rows:
        lines.append(f"| {label} | " + " | ".join(str(stats[c][key]) for c in cols) + " |")

    sizes = Counter()
    for images in splits.values():
        for p in images[::50]:                                  # sample: reading all headers is slow
            h, w = cv2.imread(str(p)).shape[:2]
            sizes[f"{w}x{h}"] += 1
    lines += ["", "Image sizes (every 50th image): " + ", ".join(f"{k} ({v})" for k, v in sizes.most_common(6))]

    # --- leakage
    all_images = [p for images in splits.values() for p in images]
    hashes = load_hashes(all_images, root / "hashes.npz", args.rehash)
    index = {p.name: i for i, p in enumerate(all_images)}
    sel = lambda names_: hashes[[index[n] for n in names_]]

    train = [p.name for p in splits["train"]]
    test = [p.name for p in splits["test"]]
    d, j = nearest(sel(test), sel(train))
    leak_rows = sorted(({"test_image": t, "nearest_train_image": train[k], "dhash_bits": int(x)}
                        for t, k, x in zip(test, j, d) if x <= threshold), key=lambda r: r["dhash_bits"])
    lines += ["", "## Near-duplicate leakage", "",
              f"A test image counts as leaked if it is within {threshold} dHash bits (of 256) of a train image, "
              "its mirror or a 90° rotation (same rule as the bootstrap split).", "",
              "| | leaked | of | share | exact (0 bits) | ≤ 5 bits |", "|---|---:|---:|---:|---:|---:|",
              f"| test vs train | {len(leak_rows)} | {len(test)} | {len(leak_rows) / len(test):.1%} | "
              f"{int((d == 0).sum())} | {int((d <= 5).sum())} |"]
    by_prefix = Counter(prefix(r["test_image"]) for r in leak_rows)
    for p in ("AoF", "WEB", "PublicDataset"):
        n = sum(prefix(t) == p for t in test)
        lines.append(f"| … test {p} | {by_prefix[p]} | {n} | {by_prefix[p] / max(n, 1):.1%} | | |")

    folds = root / "splits" / "5-fold cross validation"
    f_train, f_val = read_list(folds / "dfire_train1.txt"), read_list(folds / "dfire_valid1.txt")
    fd, _ = nearest(sel(f_val), sel(f_train))
    lines.append(f"| 5-fold: valid1 vs train1 | {int((fd <= threshold).sum())} | {len(f_val)} | "
                 f"{(fd <= threshold).mean():.1%} | {int((fd == 0).sum())} | {int((fd <= 5).sum())} |")

    out_dir = cfg["paths"]["reports"] / "external"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "dfire_leaks.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["test_image", "nearest_train_image", "dhash_bits"])
        writer.writeheader()
        writer.writerows(leak_rows)

    # sheet: leak pairs spread over the distance range (closest, middle, at the threshold)
    if leak_rows:
        picks = [leak_rows[i] for i in np.unique(np.linspace(0, len(leak_rows) - 1, 10).astype(int))]
        tiles = []
        for r in picks:
            for split, n in (("test", r["test_image"]), ("train", r["nearest_train_image"])):
                tiles.append(tile(cv2.imread(str(root / split / "images" / n)),
                                  f"{split} {Path(n).stem} {r['dhash_bits']}b"))
        cv2.imwrite(str(out_dir / "dfire__leak_pairs.jpg"), grid(tiles, cols=4), [cv2.IMWRITE_JPEG_QUALITY, 85])
        lines += ["", f"Leak pairs (test | nearest train, 2 pairs per row, closest first): "
                  "`reports/external/dfire__leak_pairs.jpg`. Full list: `reports/external/dfire_leaks.csv`."]

    md = "\n".join(lines) + "\n"
    (out_dir / f"{args.name}_inspection.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
