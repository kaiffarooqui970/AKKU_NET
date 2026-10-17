"""Public datasets (Roboflow YOLOv8 exports) for the BOOTSTRAP model only.

A Roboflow export looks like:
    <export>/data.yaml              class names
    <export>/README.roboflow.txt    preprocessing + augmentation applied by Roboflow
    <export>/{train,valid,test}/images/*.jpg
    <export>/{train,valid,test}/labels/*.txt

Each dataset has a class map (configs/class_maps/<name>.yaml) that maps every
source class to one of our classes, or to null (= drop the box). A source class
that is missing from the map, or mapped to PENDING, stops the import: every
class needs a conscious decision. The map can also list `drop_images` rules
(e.g. X-ray scans); every dropped file is listed with its rule.

The exports' own splits are NOT used: they leak (mirrored/rotated copies of
one original sit in train and valid). All kept images are pooled, near-copies
are grouped, and whole groups are split into train / bootstrap_val.

Imported data goes to data/bootstrap/, never into data/dataset/ (own footage),
so public images can never end up in our val/test/shift_test splits.
"""
import csv
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from akkunet.frames import dhash

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Roboflow's split folder -> our split name. "valid" becomes bootstrap_val.
SPLIT_NAMES = {"train": "train", "valid": "val", "test": "test"}
PENDING = "PENDING"
BOOTSTRAP_MANIFEST_FIELDS = ["image", "split", "source", "original_split", "original_file", "group",
                             "boxes_kept", "boxes_dropped"]
DROPPED_FIELDS = ["source", "original_split", "original_file", "rule"]


# ---------------------------------------------------------------- reading an export

def read_class_names(export_dir: Path) -> list[str]:
    """Class names from data.yaml, in class-ID order (Roboflow writes a list; dicts are handled too)."""
    data = yaml.safe_load((Path(export_dir) / "data.yaml").read_text())
    names = data["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    return [str(n) for n in names]


def find_split_images(export_dir: Path) -> dict[str, list[Path]]:
    """Roboflow split folder name -> sorted image paths. Missing splits are left out."""
    out = {}
    for rf_split in SPLIT_NAMES:
        img_dir = Path(export_dir) / rf_split / "images"
        if img_dir.is_dir():
            out[rf_split] = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    return out


def label_path_for(image_path: Path) -> Path:
    return image_path.parent.parent / "labels" / (image_path.stem + ".txt")


def read_boxes(label_path: Path) -> tuple[list[tuple[int, float, float, float, float]], int]:
    """YOLO boxes (cls, cx, cy, w, h) from a label file, plus how many lines were polygons.

    Some Roboflow exports contain segmentation polygons (cls x1 y1 x2 y2 ...)
    instead of boxes; those are converted to their bounding box.
    """
    boxes, polygons = [], 0
    if not label_path.exists():
        return boxes, polygons
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls, values = int(float(parts[0])), [float(v) for v in parts[1:]]
        if len(values) == 4:
            boxes.append((cls, *values))
        elif len(values) >= 6 and len(values) % 2 == 0:
            polygons += 1
            xs, ys = values[0::2], values[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            boxes.append((cls, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
        else:
            raise ValueError(f"Unreadable label line in {label_path}: {line!r}")
    return boxes, polygons


def roboflow_notes(export_dir: Path) -> str:
    """The preprocessing/augmentation part of README.roboflow.txt (empty if the file is missing)."""
    readme = Path(export_dir) / "README.roboflow.txt"
    if not readme.exists():
        return ""
    text = readme.read_text(errors="replace")
    idx = text.lower().find("pre-processing")
    return text[text.rfind("\n", 0, idx) + 1:].strip() if idx >= 0 else text.strip()


def augmentation_base_name(image_path: Path) -> str:
    """Roboflow names files '<original>_jpg.rf.<hash>.jpg'. Augmented copies of
    one source image share '<original>_jpg' and differ only in the hash."""
    return re.sub(r"\.rf\.[0-9a-f]+$", "", image_path.stem)


# ---------------------------------------------------------------- inspection

@dataclass
class SplitStats:
    images: int = 0
    empty: int = 0
    boxes: Counter = field(default_factory=Counter)
    sizes: Counter = field(default_factory=Counter)
    polygons: int = 0
    augmented_groups: int = 0      # source images that appear more than once
    augmented_images: int = 0      # images in those groups


def inspect_export(export_dir: Path) -> tuple[list[str], dict[str, SplitStats], dict[str, int]]:
    """Count images, empty images, boxes per class, image sizes and augmented copies per split.

    Also counts near-duplicates across splits: valid/test images whose dHash is
    within 10 bits of a train image. A high number means Roboflow's split leaks
    and bootstrap_val scores will be optimistic.
    """
    names = read_class_names(export_dir)
    splits = find_split_images(export_dir)
    stats, hashes = {}, {}
    for rf_split, images in splits.items():
        s = SplitStats(images=len(images))
        groups = Counter(augmentation_base_name(p) for p in images)
        s.augmented_groups = sum(1 for n in groups.values() if n > 1)
        s.augmented_images = sum(n for n in groups.values() if n > 1)
        split_hashes = []
        for p in images:
            boxes, polys = read_boxes(label_path_for(p))
            s.polygons += polys
            s.empty += not boxes
            s.boxes.update(names[b[0]] for b in boxes)
            img = cv2.imread(str(p))
            if img is None:
                raise ValueError(f"Could not read image {p}")
            s.sizes[f"{img.shape[1]}x{img.shape[0]}"] += 1
            split_hashes.append(dhash(img))
        stats[rf_split] = s
        hashes[rf_split] = np.array(split_hashes, dtype=bool).reshape(-1, 256)

    leaks = {}
    if "train" in hashes and len(hashes["train"]):
        for rf_split in ("valid", "test"):
            if rf_split in hashes:
                leaks[rf_split] = sum(
                    int((hashes["train"] != h).sum(axis=1).min() <= 10) for h in hashes[rf_split]
                )
    return names, stats, leaks


def inspection_markdown(name: str, export_dir: Path) -> str:
    names, stats, leaks = inspect_export(export_dir)
    lines = [f"# Export inspection: {name}", "", f"Folder: `{export_dir}`", "",
             f"Classes in data.yaml (ID order): {', '.join(f'`{n}`' for n in names)}", "",
             "| split | images | empty (no boxes) | augmented copies | polygon labels | image sizes (top 3) |",
             "|---|---:|---:|---:|---:|---|"]
    for rf_split, s in stats.items():
        sizes = ", ".join(f"{k} ({v})" for k, v in s.sizes.most_common(3))
        aug = f"{s.augmented_images} imgs from {s.augmented_groups} sources" if s.augmented_groups else "0"
        lines.append(f"| {rf_split} | {s.images} | {s.empty} | {aug} | {s.polygons} | {sizes} |")

    lines += ["", "Boxes per class:", "", "| class | " + " | ".join(stats) + " |",
              "|---|" + "---:|" * len(stats)]
    for n in names:
        lines.append(f"| {n} | " + " | ".join(str(s.boxes[n]) for s in stats.values()) + " |")

    if leaks:
        lines += ["", "Near-duplicates of a train image (dHash <= 10 bits): "
                  + ", ".join(f"{k}: {v}/{stats[k].images}" for k, v in leaks.items())]
    all_sizes = Counter()
    for s in stats.values():
        all_sizes.update(s.sizes)
    if len(all_sizes) == 1 and next(iter(all_sizes)) in ("640x640", "416x416", "512x512"):
        lines += ["", f"**Every image is {next(iter(all_sizes))}: probably resized (check below "
                  "whether it was stretched, which distorts object shapes).**"]
    notes = roboflow_notes(export_dir)
    lines += ["", "Roboflow preprocessing / augmentation (from README.roboflow.txt):", "", "```",
              notes or "README.roboflow.txt not found", "```"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- class maps

def load_class_map(path: Path) -> dict:
    """Read configs/class_maps/<name>.yaml -> {"dataset": ..., "mapping": {source: target|None|PENDING}}."""
    cmap = yaml.safe_load(Path(path).read_text())
    if not isinstance(cmap.get("mapping"), dict):
        raise ValueError(f"{path}: needs a 'mapping:' section")
    return cmap


def check_class_map(mapping: dict, source_names: list[str], classes: list[str]) -> list[str]:
    """Problems that must be fixed before importing (empty list = OK)."""
    problems = []
    for n in source_names:
        if n not in mapping:
            problems.append(f"source class '{n}' is not in the map (map it, or set it to null to drop it)")
        elif mapping[n] == PENDING:
            problems.append(f"source class '{n}' is still PENDING (needs a decision)")
        elif mapping[n] is not None and mapping[n] not in classes:
            problems.append(f"source class '{n}' maps to unknown class '{mapping[n]}'")
    for n in mapping:
        if n not in source_names:
            problems.append(f"map lists '{n}', which is not in this export's data.yaml")
    return problems


def remap_boxes(boxes, source_names, mapping, classes):
    """Translate source class IDs to our IDs. Returns (kept boxes, number dropped)."""
    kept, dropped = [], 0
    for cls, *xywh in boxes:
        target = mapping[source_names[cls]]
        if target is None:
            dropped += 1
        else:
            kept.append((classes.index(target), *xywh))
    return kept, dropped


# ---------------------------------------------------------------- drop rules

ONLY_OUT_OF_SCOPE = "only_out_of_scope_classes"


def check_drop_rules(rules: list[dict], source_names: list[str]) -> list[str]:
    """Problems with a map's drop_images rules (empty list = OK)."""
    problems = []
    for r in rules:
        if not r.get("name"):
            problems.append(f"drop rule {r} has no name")
        if not {"filename", "filename_not", "has_class"} & set(r):
            problems.append(f"drop rule '{r.get('name')}' has no condition")
        if "has_class" in r and r["has_class"] not in source_names:
            problems.append(f"drop rule '{r.get('name')}' uses unknown class '{r['has_class']}'")
    return problems


def drop_rule(file_name: str, source_classes: set[str], rules: list[dict]) -> str | None:
    """Name of the first rule that drops this image, or None.

    A rule drops an image when ALL of its conditions hold:
      filename:      regex found in the source filename
      filename_not:  regex NOT found in the source filename
      has_class:     this source class has a box in the image
    """
    for r in rules:
        if "filename" in r and not re.search(r["filename"], file_name):
            continue
        if "filename_not" in r and re.search(r["filename_not"], file_name):
            continue
        if "has_class" in r and r["has_class"] not in source_classes:
            continue
        return r["name"]
    return None


# ---------------------------------------------------------------- collecting

@dataclass
class Candidate:
    """One image that survived the drop rules, with boxes already in our class IDs."""
    source: str
    path: Path
    original_split: str
    boxes: list
    boxes_dropped: int


def collect_dataset(name: str, export_dir: Path, cmap: dict, classes: list[str]
                    ) -> tuple[list[Candidate], list[dict]]:
    """Apply the class map and drop rules to one export.

    Returns (kept candidates, dropped rows). Images whose boxes are ALL mapped to
    null are dropped too (rule only_out_of_scope_classes), not kept as background:
    a car battery labelled as 'nothing' would teach the wrong thing. Images with
    no boxes in the export are kept as genuine background.
    """
    source_names = read_class_names(export_dir)
    mapping, rules = cmap["mapping"], cmap.get("drop_images") or []
    problems = check_class_map(mapping, source_names, classes) + check_drop_rules(rules, source_names)
    if problems:
        raise ValueError(f"Class map for {name} is not ready:\n  - " + "\n  - ".join(problems))

    kept, dropped = [], []
    for rf_split, images in find_split_images(export_dir).items():
        for p in images:
            boxes, _ = read_boxes(label_path_for(p))
            rule = drop_rule(p.name, {source_names[b[0]] for b in boxes}, rules)
            ours, n_dropped = remap_boxes(boxes, source_names, mapping, classes)
            if rule is None and boxes and not ours:
                rule = ONLY_OUT_OF_SCOPE
            if rule:
                dropped.append({"source": name, "original_split": rf_split, "original_file": p.name, "rule": rule})
            else:
                kept.append(Candidate(name, p, rf_split, ours, n_dropped))
    return kept, dropped


# ---------------------------------------------------------------- near-copy groups

def hash_variants(image: np.ndarray) -> np.ndarray:
    """dHash of the image, its mirror and both 90-degree rotations, packed to 4 x uint64 each.

    dHash works on greyscale, so greyscale copies match their colour original.
    """
    variants = [image, cv2.flip(image, 1), cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
                cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    return np.stack([np.packbits(dhash(v)).view(np.uint64) for v in variants])


def group_near_copies(cands: list[Candidate], hashes: np.ndarray, threshold: int = 10) -> list[int]:
    """Group id per candidate. Two images share a group if (transitively):
      - they come from the same dataset and have the same Roboflow source name
        (augmented copies, incl. crops and cutout that hashing can miss), or
      - one is within `threshold` dHash bits of the other, its mirror or a 90° rotation.
    """
    parent = list(range(len(cands)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    first = {}
    for i, c in enumerate(cands):
        key = (c.source, augmentation_base_name(c.path))
        if key in first:
            union(i, first[key])
        else:
            first[key] = i

    base = hashes[:, 0]                                   # (N, 4)
    for start in range(0, len(cands), 256):
        block = base[start:start + 256, None, :]          # (b, 1, 4)
        dist = np.min([np.bitwise_count(block ^ hashes[None, :, k]).sum(-1) for k in range(4)], axis=0)
        for a, b in zip(*np.nonzero(dist <= threshold)):
            if start + a != b:
                union(start + a, int(b))

    roots = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(len(cands))]


def split_groups(cands: list[Candidate], groups: list[int], classes: list[str],
                 val_fraction: float, seed: int = 0) -> dict[int, str]:
    """Assign whole groups to train or val, aiming at `val_fraction` of the images of every class.

    Groups are visited in random order; a group goes to val only if that keeps
    every class it contains (and the background images) at or below its val
    target. So per-class val shares end up close to, never far above, the target.
    """
    members = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)

    def kinds(i):  # what an image counts towards: its classes, or "background"
        return {classes[b[0]] for b in cands[i].boxes} or {"background"}

    total = Counter(k for i in range(len(cands)) for k in kinds(i))
    target = {k: val_fraction * n for k, n in total.items()}
    in_val = Counter()
    order = np.random.default_rng(seed).permutation(sorted(members))
    assignment = {}
    for g in order:
        need = Counter(k for i in members[g] for k in kinds(i))
        if all(in_val[k] + n <= target[k] for k, n in need.items()):
            assignment[int(g)] = "val"
            in_val.update(need)
        else:
            assignment[int(g)] = "train"
    return assignment


# ---------------------------------------------------------------- writing

def write_bootstrap(cands: list[Candidate], groups: list[int], assignment: dict[int, str],
                    classes: list[str], bootstrap_dir: Path) -> list[dict]:
    """(Re)build bootstrap_dir/{images,labels}/{train,val}/ and return manifest rows.

    The whole folder is rebuilt every time, so a changed rule never leaves stale
    images behind. Filenames are prefixed with the dataset name.
    """
    for sub in ("images", "labels", "lists"):
        shutil.rmtree(bootstrap_dir / sub, ignore_errors=True)
    rows, seen = [], set()
    for c, g in zip(cands, groups):
        split = assignment[g]
        out_name = f"{c.source}__{c.path.stem}"
        if out_name in seen:
            raise ValueError(f"Two images would both be called {out_name}")
        seen.add(out_name)
        img_out = bootstrap_dir / "images" / split / (out_name + c.path.suffix.lower())
        lbl_out = bootstrap_dir / "labels" / split / (out_name + ".txt")
        img_out.parent.mkdir(parents=True, exist_ok=True)
        lbl_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(c.path, img_out)
        lbl_out.write_text("".join(f"{k} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for k, x, y, w, h in c.boxes))
        rows.append({"image": str(img_out.relative_to(bootstrap_dir)), "split": split, "source": c.source,
                     "original_split": c.original_split, "original_file": c.path.name, "group": g,
                     "boxes_kept": len(c.boxes), "boxes_dropped": c.boxes_dropped})
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
