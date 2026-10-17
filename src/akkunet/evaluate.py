"""Per-class detection metrics as a markdown table.

Classes without any validation boxes are listed too (as "no val data"), so a
class the model was never trained on can't silently disappear from a report.
"""
import csv
from pathlib import Path

import yaml

BOOTSTRAP_BANNER = "**BOOTSTRAP ONLY — not tipping-floor performance.**"


def write_run_dataset(bootstrap_dir: Path, run: str, classes: list[str],
                      train_sources: list[str], val_sources: list[str],
                      extra_train_images: list[str] | None = None) -> Path:
    """Write image lists + an Ultralytics yaml for one run, filtered by manifest source.

    Using lists (not folders) lets two runs share data/bootstrap/ but train on
    different datasets while validating on exactly the same bootstrap_val.
    `extra_train_images` (e.g. hard negatives) are appended to TRAIN only.
    """
    with open(bootstrap_dir / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    known = {r["source"] for r in rows}
    missing = sorted(set(train_sources + val_sources) - known)
    if missing:
        raise ValueError(f"Not imported yet: {missing}. Imported: {sorted(known)}")

    lists = bootstrap_dir / "lists"
    lists.mkdir(exist_ok=True)
    files = {}
    for split, sources in (("train", train_sources), ("val", val_sources)):
        paths = [str(bootstrap_dir / r["image"]) for r in rows if r["split"] == split and r["source"] in sources]
        if not paths:
            raise ValueError(f"No {split} images from {sources}")
        if split == "train":
            paths += extra_train_images or []
        files[split] = lists / f"{run}_{split}.txt"
        files[split].write_text("\n".join(paths) + "\n")

    data = {"train": str(files["train"]), "val": str(files["val"]), "names": dict(enumerate(classes))}
    out = bootstrap_dir / f"{run}.yaml"
    extra = f" + {len(extra_train_images)} extra train images" if extra_train_images else ""
    out.write_text(f"# BOOTSTRAP ONLY. train: {train_sources}{extra}  val: {val_sources}\n"
                   + yaml.safe_dump(data, sort_keys=False))
    return out


def per_class_table(metrics, classes: list[str], val_box_counts: dict[str, int]) -> str:
    """Markdown table from an Ultralytics DetMetrics object (model.val() result)."""
    box = metrics.box
    by_class = {int(c): i for i, c in enumerate(box.ap_class_index)}
    lines = ["| class | val boxes | precision | recall | mAP50 | mAP50-95 |",
             "|---|---:|---:|---:|---:|---:|"]
    for cid, name in enumerate(classes):
        n = val_box_counts.get(name, 0)
        if cid in by_class:
            i = by_class[cid]
            lines.append(f"| {name} | {n} | {box.p[i]:.3f} | {box.r[i]:.3f} | {box.ap50[i]:.3f} | {box.ap[i]:.3f} |")
        else:
            lines.append(f"| {name} | {n} | – | – | – | – (no val data; untrained) |")
    lines.append(f"| **all (classes with data)** | {sum(val_box_counts.values())} | {box.mp:.3f} | "
                 f"{box.mr:.3f} | {box.map50:.3f} | {box.map:.3f} |")
    return "\n".join(lines)


def count_boxes(list_file: Path, classes: list[str]) -> dict[str, int]:
    """Boxes per class over the images in an Ultralytics image list."""
    counts = dict.fromkeys(classes, 0)
    for img in Path(list_file).read_text().split():
        label = Path(img.replace("/images/", "/labels/")).with_suffix(".txt")
        if label.exists():
            for line in label.read_text().splitlines():
                if line.strip():
                    counts[classes[int(line.split()[0])]] += 1
    return counts
