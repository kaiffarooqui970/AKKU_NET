"""Fire/smoke model data (fire_v1): D-Fire, re-split by near-copy group.

The authors' test split leaks (35% near-copies of train, 78% for the AoF
surveillance cameras; reports/dfire_dataset.md), so it is ignored. All images
are pooled, near-copies grouped (same rule as the bootstrap) and whole groups
assigned to train or val, aiming at val_fraction of every class.

No images are copied (disk is tight): Ultralytics reads image lists that point
at the original files; it finds labels by replacing /images/ with /labels/.

    data/fire/lists/<run>_{train,val}.txt   absolute image paths
    data/fire/<run>.yaml                    Ultralytics dataset
    data/fire/manifest.csv                  image, split, original split, group, boxes
"""
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from akkunet.external import (Candidate, find_split_images, group_near_copies, hash_variants, label_path_for,
                              read_boxes, read_class_names, split_groups, write_csv)

FIRE_MANIFEST_FIELDS = ["image", "split", "original_split", "group", "smoke_boxes", "fire_boxes"]
FIRE_BANNER = ("**FIRE_V1: D-Fire public images (mostly outdoor: wildfire, streets, surveillance cameras), "
               "NOT tipping-floor performance.**")


def cached_hash_variants(images: list[Path], cache: Path, rehash: bool = False, progress=print) -> np.ndarray:
    """hash_variants() for every image, cached in `cache` (.npz) keyed by the filename order."""
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
            progress(f"  hashed {i}/{len(images)}")
    np.savez(cache, names=names, hashes=out)
    return out


def collect_fire(root: Path, classes: list[str]) -> list[Candidate]:
    """Every image of the dataset with its boxes. Class IDs must match `classes` exactly."""
    names = read_class_names(root)
    if names != classes:
        raise ValueError(f"{root}/data.yaml has classes {names}, config fire.classes is {classes}")
    return [Candidate(root.name, p, split, read_boxes(label_path_for(p))[0], 0)
            for split, images in find_split_images(root).items() for p in images]


def build_fire_split(root: Path, classes: list[str], hashes: np.ndarray, threshold: int,
                     val_fraction: float, seed: int) -> tuple[list[Candidate], list[int], dict[int, str]]:
    cands = collect_fire(root, classes)
    if len(cands) != len(hashes):
        raise ValueError(f"{len(cands)} images but {len(hashes)} hashes; rerun with --rehash")
    groups = group_near_copies(cands, hashes, threshold)
    return cands, groups, split_groups(cands, groups, classes, val_fraction, seed)


def write_fire_lists(cands: list[Candidate], groups: list[int], assignment: dict[int, str],
                     classes: list[str], out_dir: Path, run: str) -> tuple[Path, list[dict]]:
    """Image lists + Ultralytics yaml + manifest rows. Refuses a group in both splits."""
    seen = {}
    for g in groups:
        if seen.setdefault(g, assignment[g]) != assignment[g]:
            raise ValueError(f"Group {g} would be in both splits")
    lists = out_dir / "lists"
    lists.mkdir(parents=True, exist_ok=True)
    rows, files = [], {}
    for split in ("train", "val"):
        paths = [str(c.path.resolve()) for c, g in zip(cands, groups) if assignment[g] == split]
        files[split] = lists / f"{run}_{split}.txt"
        files[split].write_text("\n".join(paths) + "\n")
    for c, g in zip(cands, groups):
        n = Counter(classes[b[0]] for b in c.boxes)
        rows.append({"image": str(c.path.resolve()), "split": assignment[g], "original_split": c.original_split,
                     "group": g, "smoke_boxes": n["smoke"], "fire_boxes": n["fire"]})
    write_csv(out_dir / "manifest.csv", rows, FIRE_MANIFEST_FIELDS)
    data = {"train": str(files["train"]), "val": str(files["val"]), "names": dict(enumerate(classes))}
    out = out_dir / f"{run}.yaml"
    out.write_text(f"# {run}: D-Fire re-split by near-copy group (authors' test split ignored: it leaks)\n"
                   + yaml.safe_dump(data, sort_keys=False))
    return out, rows
