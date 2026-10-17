"""Part B2: build data/bootstrap/ from all datasets in config bootstrap.datasets.

Usage:
    python scripts/build_bootstrap.py

Steps:
  1. per dataset: apply the class map and its drop_images rules
     (configs/class_maps/<name>.yaml; refuses while a class is unmapped/PENDING)
  2. pool everything that is kept and group near-copies: same Roboflow source
     name, or dHash within bootstrap.near_copy_hamming bits of the image, its
     mirror or a 90° rotation (greyscale copies match too)
  3. split whole groups into train / val (bootstrap_val), ~bootstrap.val_fraction
     of every class in val; the exports' own splits are ignored because they leak
  4. rebuild data/bootstrap/ from scratch

Output:
    data/bootstrap/{images,labels}/{train,val}/<dataset>__<file>
    data/bootstrap/manifest.csv              one row per image (source, original split, group)
    reports/external/bootstrap_dropped.csv   every dropped file and the rule that dropped it
    reports/bootstrap_dataset.md             counts per rule, class and split
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from akkunet.config import load_config
from akkunet.external import (BOOTSTRAP_MANIFEST_FIELDS, DROPPED_FIELDS, collect_dataset, group_near_copies,
                              hash_variants, load_class_map, split_groups, write_bootstrap, write_csv)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=REPO_ROOT / "config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths, classes, bs = cfg["paths"], cfg["classes"], cfg["bootstrap"]

    cands, dropped, images_in, notes = [], [], Counter(), {}
    try:
        for name in bs["datasets"]:
            cmap = load_class_map(paths["class_maps"] / f"{name}.yaml")
            kept, gone = collect_dataset(name, paths["external"] / name, cmap, classes)
            cands += kept
            notes[name] = cmap.get("notes") or []
            dropped += gone
            images_in[name] = len(kept) + len(gone)
            print(f"{name}: {len(kept)} kept, {len(gone)} dropped", flush=True)
    except (ValueError, FileNotFoundError) as e:
        print(f"Build refused: {e}")
        return 1

    print(f"Hashing {len(cands)} images ...", flush=True)
    hashes = np.stack([hash_variants(cv2.imread(str(c.path))) for c in cands])
    groups = group_near_copies(cands, hashes, bs["near_copy_hamming"])
    assignment = split_groups(cands, groups, classes, bs["val_fraction"], bs["seed"])

    rows = write_bootstrap(cands, groups, assignment, classes, paths["bootstrap"])
    write_csv(paths["bootstrap"] / "manifest.csv", rows, BOOTSTRAP_MANIFEST_FIELDS)
    write_csv(paths["reports"] / "external" / "bootstrap_dropped.csv", dropped, DROPPED_FIELDS)

    # Sanity check straight from the written manifest: no group in both splits.
    splits_per_group = defaultdict(set)
    for r in rows:
        splits_per_group[r["group"]].add(r["split"])
    straddling = sum(len(s) > 1 for s in splits_per_group.values())
    if straddling:
        print(f"ERROR: {straddling} groups are in both train and val")
        return 1

    md = report(cfg, cands, rows, dropped, images_in, groups, splits_per_group, notes)
    out = paths["reports"] / "bootstrap_dataset.md"
    out.write_text(md)
    print(md)
    print(f"Wrote {out}, {paths['bootstrap'] / 'manifest.csv'} and the dropped-file list")
    return 0


def report(cfg, cands, rows, dropped, images_in, groups, splits_per_group, notes) -> str:
    classes, bs = cfg["classes"], cfg["bootstrap"]
    lines = ["# Bootstrap dataset", "",
             "**BOOTSTRAP ONLY — public data, not tipping-floor data.**", "",
             "## Images removed per rule", "", "| dataset | images in export | rule | images removed |",
             "|---|---:|---|---:|"]
    by_rule = Counter((d["source"], d["rule"]) for d in dropped)
    for name in bs["datasets"]:
        rules = sorted(r for s, r in by_rule if s == name)
        for r in rules:
            lines.append(f"| {name} | {images_in[name]} | {r} | {by_rule[(name, r)]} |")
        kept = sum(1 for c in cands if c.source == name)
        lines.append(f"| {name} | | **kept** | **{kept}** |")
    lines += ["", "Full list: `reports/external/bootstrap_dropped.csv` (dataset, original split, file, rule).", ""]

    group_sizes = Counter(Counter(groups).values())
    n_groups = {s: sum(1 for v in splits_per_group.values() if v == {s}) for s in ("train", "val")}
    largest = max(Counter(groups).values())
    lines += ["## Split by source image (near-copy groups)", "",
              f"Near-copies = same Roboflow source name, or dHash ≤ {bs['near_copy_hamming']} bits to the image, "
              "its mirror or a 90° rotation (greyscale copies match too). Target: "
              f"{bs['val_fraction']:.0%} of every class in val.", "",
              "| split | groups | images |", "|---|---:|---:|"]
    for s in ("train", "val"):
        lines.append(f"| {s} | {n_groups[s]} | {sum(r['split'] == s for r in rows)} |")
    lines += ["", f"Groups in both train and val: **0** (checked from the written manifest). "
              f"Total {len(splits_per_group)} groups; {group_sizes[1]} single images; largest group {largest} images.", ""]

    lines += ["## Final counts", "", "| class | train images | train boxes | val images | val boxes | val share (boxes) |",
              "|---|---:|---:|---:|---:|---:|"]
    stats = {s: (Counter(), Counter()) for s in ("train", "val")}
    for c, r in zip(cands, rows):
        imgs, boxes = stats[r["split"]]
        present = Counter(classes[b[0]] for b in c.boxes)
        boxes.update(present)
        imgs.update(present.keys() or ["(background)"])
    for name in classes + ["(background)"]:
        tb, vb = stats["train"][1][name], stats["val"][1][name]
        share = f"{vb / (tb + vb):.0%}" if tb + vb else "–"
        note = " (no data: untrained)" if tb + vb == 0 and name != "(background)" else ""
        lines.append(f"| {name}{note} | {stats['train'][0][name]} | {tb} | {stats['val'][0][name]} | {vb} | {share} |")
    lines += ["", "Per source:", "", "| dataset | train | val |", "|---|---:|---:|"]
    for name in bs["datasets"]:
        lines.append(f"| {name} | " + " | ".join(
            str(sum(r["source"] == name and r["split"] == s for r in rows)) for s in ("train", "val")) + " |")
    lines += ["", "## Known label quality issues (from the class maps)", ""]
    lines += [f"- **{name}:** {n}" for name, ns in notes.items() for n in ns] or ["- none recorded"]
    lines += ["", "Licences and sources: `SOURCES.md`."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
