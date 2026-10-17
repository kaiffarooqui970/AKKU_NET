"""fire_v1 step 1: re-split D-Fire by near-copy group (CPU only, no image copies).

Usage:
    python scripts/build_fire_dataset.py              # run name fire_v1
    python scripts/build_fire_dataset.py --run fire_v2 --rehash

Pools the authors' train and test images (their test split leaks), groups
near-copies (dHash <= fire.near_copy_hamming bits incl. mirror / 90° rotations,
hashes cached by scripts/inspect_dfire.py), assigns whole groups to train / val
(~fire.val_fraction of every class), then checks no group and no near-copy
crosses the split.

Output: data/fire/{lists/,<run>.yaml,manifest.csv}, reports/fire_dataset.md
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from akkunet.config import load_config
from akkunet.fire import build_fire_split, cached_hash_variants, collect_fire, write_fire_lists

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="fire_v1")
    parser.add_argument("--rehash", action="store_true")
    parser.add_argument("--config", default=REPO_ROOT / "config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fc = cfg["fire"]
    root = cfg["paths"]["external"] / fc["dataset"]
    images = [c.path for c in collect_fire(root, fc["classes"])]
    hashes = cached_hash_variants(images, root / "hashes.npz", args.rehash)
    cands, groups, assignment = build_fire_split(root, fc["classes"], hashes, fc["near_copy_hamming"],
                                                 fc["val_fraction"], fc["seed"])
    data_yaml, rows = write_fire_lists(cands, groups, assignment, fc["classes"], cfg["paths"]["fire"], args.run)

    # independent check: no val image within the threshold of any train image (or its variants)
    split = np.array([r["split"] for r in rows])
    tr, va = hashes[split == "train"], hashes[split == "val"]
    crossing = 0
    for start in range(0, len(va), 256):
        block = va[start:start + 256, 0][:, None, :]
        dist = np.min([np.bitwise_count(block ^ tr[None, :, k]).sum(-1) for k in range(4)], axis=0)
        crossing += int((dist.min(1) <= fc["near_copy_hamming"]).sum())
    group_splits = {}
    for r in rows:
        group_splits.setdefault(r["group"], set()).add(r["split"])
    straddling = sum(len(s) > 1 for s in group_splits.values())

    sizes = Counter(Counter(groups).values())
    big = sorted(Counter(groups).values(), reverse=True)[:5]
    lines = [f"# Fire dataset: {args.run} (D-Fire re-split by near-copy group)", "",
             "The authors' test split is ignored: 35% of it are near-copies of train "
             "(reports/dfire_dataset.md). All 21,527 images are pooled and near-copies "
             f"(dHash <= {fc['near_copy_hamming']} of 256 bits, incl. mirror and 90° rotations; linked "
             "transitively) form groups. Whole groups go to train or val, aiming at "
             f"{fc['val_fraction']:.0%} of every class. No images copied: lists point at "
             f"`data/external/{fc['dataset']}/`.", "",
             "| split | groups | images | smoke images | fire images | empty | smoke boxes | fire boxes |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in ("train", "val"):
        rs = [r for r in rows if r["split"] == s]
        lines.append(f"| {s} | {sum(1 for g, v in group_splits.items() if v == {s})} | {len(rs)} | "
                     f"{sum(r['smoke_boxes'] > 0 for r in rs)} | {sum(r['fire_boxes'] > 0 for r in rs)} | "
                     f"{sum(r['smoke_boxes'] + r['fire_boxes'] == 0 for r in rs)} | "
                     f"{sum(r['smoke_boxes'] for r in rs)} | {sum(r['fire_boxes'] for r in rs)} |")
    n = len(rows)
    lines += ["", f"- Groups in both splits: **{straddling}**.",
              f"- Val images within {fc['near_copy_hamming']} bits of any train image (checked independently): "
              f"**{crossing}**.",
              f"- Val share of images: {sum(r['split'] == 'val' for r in rows) / n:.1%}.",
              f"- Group sizes: {sizes[1]} single images; largest groups {big} images "
              "(fixed surveillance cameras: many frames of one view chain into one group).",
              "- Where the val images came from (authors' split): " + ", ".join(
                  f"{k} {v}" for k, v in Counter(r['original_split'] for r in rows if r['split'] == 'val').items()),
              "", "Known limitations (carry into every fire_v1 report):", "",
              "- Night scenes: the same bright glow is labelled `fire` in some images and `smoke` in others "
              "(e.g. AoF04543 on both contact sheets). Per-class night results are unreliable; the combined "
              "`fire_or_smoke` metric is the one that matters for the escalation.",
              "- Mostly outdoor footage; val is the same kind of data as train. Not tipping-floor performance.", ""]
    out = cfg["paths"]["reports"] / "fire_dataset.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"Wrote {data_yaml}, {out}")
    return 1 if straddling or crossing else 0


if __name__ == "__main__":
    sys.exit(main())
