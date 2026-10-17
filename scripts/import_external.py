"""Part B2: import a public dataset into data/bootstrap/ using its class map.

Usage:
    python scripts/import_external.py kashif_battery
    python scripts/import_external.py gvl610_battery --max-train-images 1500

Run scripts/inspect_export.py first. The import refuses to run while any
source class is unmapped or PENDING in configs/class_maps/<name>.yaml.
Re-running replaces this dataset's earlier import (other datasets are kept).

Output:
    data/bootstrap/{images,labels}/{train,val,test}/<name>__<file>
    data/bootstrap/manifest.csv     one row per imported image (source = dataset name)
    data/bootstrap/bootstrap.yaml   Ultralytics dataset file (all 5 classes)
    reports/external/<name>_import.md
"""
import argparse
import sys
from pathlib import Path

from akkunet.config import load_config
from akkunet.external import (check_class_map, import_export, load_class_map, read_class_names,
                              remove_source, update_manifest, write_data_yaml)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name")
    parser.add_argument("--max-train-images", type=int, default=None, help="random cap on the train split")
    parser.add_argument("--config", default=REPO_ROOT / "config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths, classes = cfg["paths"], cfg["classes"]
    export_dir = paths["external"] / args.name
    cmap_path = paths["class_maps"] / f"{args.name}.yaml"
    if not cmap_path.exists():
        print(f"No class map at {cmap_path}.")
        return 1

    bootstrap_dir = paths["bootstrap"]
    try:
        cmap = load_class_map(cmap_path)
        # Validate before deleting the previous import, so a bad map changes nothing.
        problems = check_class_map(cmap["mapping"], read_class_names(export_dir), classes)
        if problems:
            raise ValueError("Class map is not ready:\n  - " + "\n  - ".join(problems))
        removed = remove_source(bootstrap_dir, args.name)
        stats, rows = import_export(args.name, export_dir, cmap, classes, bootstrap_dir,
                                    max_train_images=args.max_train_images, seed=cfg["bootstrap"]["seed"])
    except ValueError as e:
        print(f"Import refused: {e}")
        return 1

    update_manifest(bootstrap_dir, args.name, rows)
    data_yaml = write_data_yaml(bootstrap_dir, classes)

    splits = list(stats.images_in)
    lines = [f"# Import: {args.name}", "", "BOOTSTRAP ONLY - public data, never used for val/test of own footage.", "",
             f"Replaced an earlier import of this dataset ({removed} images)." if removed else "First import.", "",
             "| split | images in export | imported | of which background (no boxes in export) "
             "| skipped (all boxes dropped) | kept with some boxes dropped |",
             "|---|---:|---:|---:|---:|---:|"]
    for s in splits:
        lines.append(f"| {s} | {stats.images_in[s]} | {stats.imported[s]} | {stats.background[s]} "
                     f"| {stats.only_dropped[s]} | {stats.mixed[s]} |")
    lines += ["", "Boxes per class after mapping:", "", "| class | " + " | ".join(splits) + " |",
              "|---|" + "---:|" * len(splits)]
    for c in classes:
        lines.append(f"| {c} | " + " | ".join(str(stats.boxes[s][c]) for s in splits) + " |")
    md = "\n".join(lines) + "\n"

    out = paths["reports"] / "external" / f"{args.name}_import.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)
    print(f"Wrote {out} and {data_yaml}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
