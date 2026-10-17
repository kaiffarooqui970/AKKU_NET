"""Public datasets (Roboflow YOLOv8 exports) for the BOOTSTRAP model only.

A Roboflow export looks like:
    <export>/data.yaml              class names
    <export>/README.roboflow.txt    preprocessing + augmentation applied by Roboflow
    <export>/{train,valid,test}/images/*.jpg
    <export>/{train,valid,test}/labels/*.txt

Each dataset has a class map (configs/class_maps/<name>.yaml) that maps every
source class to one of our classes, or to null (= drop the box). A source class
that is missing from the map, or mapped to PENDING, stops the import: every
class needs a conscious decision.

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
BOOTSTRAP_MANIFEST_FIELDS = ["image", "split", "source", "original_file", "boxes_kept", "boxes_dropped"]


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


# ---------------------------------------------------------------- import

@dataclass
class ImportStats:
    images_in: Counter = field(default_factory=Counter)
    imported: Counter = field(default_factory=Counter)
    background: Counter = field(default_factory=Counter)        # had no boxes in the export
    only_dropped: Counter = field(default_factory=Counter)      # every box was dropped -> skipped
    mixed: Counter = field(default_factory=Counter)             # some boxes dropped, image kept
    boxes: dict = field(default_factory=lambda: defaultdict(Counter))  # split -> our class -> n


def import_export(name: str, export_dir: Path, cmap: dict, classes: list[str],
                  bootstrap_dir: Path, max_train_images: int | None = None,
                  seed: int = 0) -> tuple[ImportStats, list[dict]]:
    """Copy one export into bootstrap_dir/{images,labels}/<split>/ with remapped labels.

    Rules:
      - images whose boxes were ALL dropped are skipped, not kept as background
        (a car battery labelled as 'nothing' would teach the wrong thing)
      - images that had no boxes in the export are kept as genuine background
      - filenames are prefixed with the dataset name so sources never collide
    `max_train_images` caps the train split (random sample with `seed`).
    """
    source_names = read_class_names(export_dir)
    mapping = cmap["mapping"]
    problems = check_class_map(mapping, source_names, classes)
    if problems:
        raise ValueError(f"Class map for {name} is not ready:\n  - " + "\n  - ".join(problems))

    stats, rows = ImportStats(), []
    for rf_split, images in find_split_images(export_dir).items():
        split = SPLIT_NAMES[rf_split]
        if split == "train" and max_train_images and len(images) > max_train_images:
            rng = np.random.default_rng(seed)
            images = sorted(rng.choice(images, size=max_train_images, replace=False))
        img_out = bootstrap_dir / "images" / split
        lbl_out = bootstrap_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for p in images:
            stats.images_in[split] += 1
            boxes, _ = read_boxes(label_path_for(p))
            kept, dropped = remap_boxes(boxes, source_names, mapping, classes)
            if boxes and not kept:
                stats.only_dropped[split] += 1
                continue
            if not boxes:
                stats.background[split] += 1
            if dropped and kept:
                stats.mixed[split] += 1

            out_name = f"{name}__{p.stem}"
            shutil.copy2(p, img_out / (out_name + p.suffix.lower()))
            (lbl_out / (out_name + ".txt")).write_text(
                "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in kept))
            stats.imported[split] += 1
            stats.boxes[split].update(classes[b[0]] for b in kept)
            rows.append({"image": f"images/{split}/{out_name}{p.suffix.lower()}", "split": split,
                         "source": name, "original_file": p.name,
                         "boxes_kept": len(kept), "boxes_dropped": dropped})
    return stats, rows


def remove_source(bootstrap_dir: Path, name: str) -> int:
    """Delete everything previously imported from one dataset (so re-imports are clean)."""
    removed = 0
    for sub in ("images", "labels"):
        for p in (bootstrap_dir / sub).glob(f"*/{name}__*"):
            p.unlink()
            removed += sub == "images"
    return removed


def update_manifest(bootstrap_dir: Path, name: str, rows: list[dict]) -> None:
    """Replace this dataset's rows in bootstrap_dir/manifest.csv, keeping other datasets' rows."""
    path = bootstrap_dir / "manifest.csv"
    existing = []
    if path.exists():
        with open(path, newline="") as f:
            existing = [r for r in csv.DictReader(f) if r["source"] != name]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BOOTSTRAP_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(existing + rows)


def write_data_yaml(bootstrap_dir: Path, classes: list[str]) -> Path:
    """Ultralytics dataset file. All 5 classes are listed even without data, so IDs never change."""
    data = {"path": str(bootstrap_dir), "train": "images/train", "val": "images/val",
            "names": dict(enumerate(classes))}
    if (bootstrap_dir / "images" / "test").is_dir():
        data["test"] = "images/test"
    out = bootstrap_dir / "bootstrap.yaml"
    out.write_text("# BOOTSTRAP ONLY - public data, not tipping-floor data\n" + yaml.safe_dump(data, sort_keys=False))
    return out
