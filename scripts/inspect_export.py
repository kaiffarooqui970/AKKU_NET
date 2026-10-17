"""Part B2: check a downloaded public dataset BEFORE importing it.

Usage:
    python scripts/inspect_export.py battery_and_vape      # folder under data/external/

Reports per split: image count, empty images, Roboflow augmented copies,
image sizes (all 640x640 = resized, maybe stretched), boxes per class,
near-duplicates leaking from train into valid/test, and the preprocessing /
augmentation text from README.roboflow.txt.

Output: reports/external/<name>_inspection.md (also printed)
"""
import argparse
import sys
from pathlib import Path

from akkunet.config import load_config
from akkunet.external import inspection_markdown

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="dataset folder name under paths.external")
    parser.add_argument("--config", default=REPO_ROOT / "config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    export_dir = cfg["paths"]["external"] / args.name
    if not (export_dir / "data.yaml").exists():
        print(f"No data.yaml in {export_dir}. Download the YOLOv8 export and unzip it there.")
        return 1

    md = inspection_markdown(args.name, export_dir)
    out = cfg["paths"]["reports"] / "external" / f"{args.name}_inspection.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
