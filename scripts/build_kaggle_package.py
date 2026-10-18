"""Package bootstrap_v2 for training on Kaggle's free GPU (CPU only, local).

Usage:
    python scripts/build_kaggle_package.py                          # bootstrap data + v1 weights
    python scripts/build_kaggle_package.py --include-own-negatives  # + data/negatives (shows your face:
                                                                    #   upload as a PRIVATE Kaggle dataset only)

Writes (git-ignored):
    data/kaggle/akkunet_bootstrap_v2.zip   upload as a private Kaggle dataset
    data/kaggle/bootstrap_v2.ipynb         import into a Kaggle notebook (from kaggle/bootstrap_v2.py)

The public "no battery" images (COCO) are downloaded on Kaggle, not here.
"""
import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

from akkunet.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_SOURCE = REPO_ROOT / "kaggle" / "bootstrap_v2.py"


def py_to_ipynb(source: str) -> dict:
    """Split '# %%' / '# %% [markdown]' cells into a Jupyter notebook."""
    cells = []
    for chunk in re.split(r"^# %%", source, flags=re.M)[1:]:
        header, _, body = chunk.partition("\n")
        body = body.strip("\n")
        if "[markdown]" in header:
            text = "\n".join(re.sub(r"^# ?", "", line) for line in body.splitlines())
            cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": body.splitlines(keepends=True)})
    return {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                        "language": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--include-own-negatives", action="store_true")
    parser.add_argument("--weights", default=REPO_ROOT / "runs/bootstrap/bootstrap_v1/weights/best.pt")
    parser.add_argument("--config", default=REPO_ROOT / "config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bootstrap, negatives = cfg["paths"]["bootstrap"], cfg["paths"]["negatives"]
    out_dir = REPO_ROOT / "data" / "kaggle"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = Path(args.weights)
    if not weights.exists():
        print(f"{weights} not found")
        return 1

    files = [(weights, "weights/bootstrap_v1_best.pt")]
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        files += [(p, f"bootstrap/{sub}/{p.name}") for p in sorted((bootstrap / sub).iterdir()) if p.is_file()]
    if args.include_own_negatives:
        from akkunet.negatives import negative_train_images
        for img in map(Path, negative_train_images(negatives)):       # refuses non-train / non-empty labels
            label = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
            files += [(img, f"own_negatives/images/train/{img.name}"), (label, f"own_negatives/labels/train/{label.name}")]

    zip_path = out_dir / "akkunet_bootstrap_v2.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as z:   # JPEGs don't compress further
        for path, arc in files:
            z.write(path, arc)
        z.writestr("README.txt", "AkkuNet bootstrap_v2 upload. BOOTSTRAP ONLY: internet photos, not tipping-floor "
                   "performance. Sources and licences: SOURCES.md in the AkkuNet repo. Keep this dataset PRIVATE"
                   + (" (own_negatives/ shows a person)." if args.include_own_negatives else ".") + "\n")
    nb_path = out_dir / "bootstrap_v2.ipynb"
    nb_path.write_text(json.dumps(py_to_ipynb(NOTEBOOK_SOURCE.read_text()), indent=1))

    n_img = sum(1 for _, a in files if "/images/" in a)
    print(f"{zip_path}: {zip_path.stat().st_size / 1e6:.0f} MB, {n_img} images"
          + (" (incl. own negatives: keep the Kaggle dataset PRIVATE)" if args.include_own_negatives else ""))
    print(f"{nb_path}: {len(json.loads(nb_path.read_text())['cells'])} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
