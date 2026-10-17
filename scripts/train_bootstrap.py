"""Part B2 step 6-7: train the bootstrap model on imported public data.

Usage:
    python scripts/train_bootstrap.py bootstrap_v1 --dry-run        # print command, write lists
    python scripts/train_bootstrap.py bootstrap_v1 --benchmark      # time 10% of one epoch -> estimate
    python scripts/train_bootstrap.py bootstrap_v1                  # train + per-class report
    python scripts/train_bootstrap.py bootstrap_gvl610 --extra gvl610_battery

Train = config bootstrap.datasets (+ --extra). Val is ALWAYS the valid split of
bootstrap.datasets only, so runs with and without extras are compared on the
same images.

Output:
    runs/bootstrap/<run>/                 Ultralytics run (weights/best.pt, plots)
    reports/bootstrap_<run>_metrics.md    per-class P/R/mAP, all 5 classes
"""
import argparse
import sys
import time
from pathlib import Path

from akkunet.config import load_config
from akkunet.device import get_device
from akkunet.evaluate import BOOTSTRAP_BANNER, count_boxes, per_class_table, write_run_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_FRACTION = 0.1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="run name, e.g. bootstrap_v1")
    parser.add_argument("--extra", nargs="*", default=[], help="extra imported datasets for TRAIN only")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    parser.add_argument("--config", default=REPO_ROOT / "config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bs, classes = cfg["bootstrap"], cfg["classes"]
    try:
        data_yaml = write_run_dataset(cfg["paths"]["bootstrap"], args.run, classes,
                                      train_sources=bs["datasets"] + args.extra, val_sources=bs["datasets"])
    except (ValueError, FileNotFoundError) as e:
        print(f"Cannot build the dataset: {e}")
        return 1

    device = get_device(cfg["device"])
    train_args = dict(data=str(data_yaml), imgsz=bs["imgsz"], epochs=bs["epochs"], batch=bs["batch"],
                      patience=bs["patience"], seed=bs["seed"], deterministic=False, device=device,
                      project=str(REPO_ROOT / "runs" / "bootstrap"), name=args.run)
    n_train = len((data_yaml.parent / "lists" / f"{args.run}_train.txt").read_text().split())
    n_val = len((data_yaml.parent / "lists" / f"{args.run}_val.txt").read_text().split())
    print(f"{n_train} train images, {n_val} val images, device {device}")
    print("Equivalent command:\n  yolo detect train model=" + bs["model"] + " "
          + " ".join(f"{k}={v}" for k, v in train_args.items()))
    if args.dry_run:
        return 0

    from ultralytics import YOLO

    if args.benchmark:
        start = time.time()
        YOLO(bs["model"]).train(**{**train_args, "epochs": 1, "fraction": BENCHMARK_FRACTION, "val": False,
                                   "plots": False, "name": f"{args.run}_benchmark", "exist_ok": True})
        per_epoch = (time.time() - start) / BENCHMARK_FRACTION
        print(f"\n~{per_epoch / 60:.1f} min per epoch (train only, extrapolated from {BENCHMARK_FRACTION:.0%}) "
              f"-> up to ~{per_epoch * bs['epochs'] / 3600:.1f} h for {bs['epochs']} epochs, plus validation "
              f"each epoch. Early stopping (patience {bs['patience']}) can end it sooner.")
        return 0

    model = YOLO(bs["model"])
    model.train(**train_args)
    best = REPO_ROOT / "runs" / "bootstrap" / args.run / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(data=str(data_yaml), imgsz=bs["imgsz"], device=device,
                                  project=str(REPO_ROOT / "runs" / "bootstrap"), name=f"{args.run}_val")
    val_counts = count_boxes(data_yaml.parent / "lists" / f"{args.run}_val.txt", classes)
    md = (f"# Bootstrap metrics: {args.run}\n\n{BOOTSTRAP_BANNER}\n\n"
          f"Public data only. Train: {bs['datasets'] + args.extra} ({n_train} images). "
          f"Val: bootstrap_val = valid splits of {bs['datasets']} ({n_val} images). "
          f"Model {bs['model']}, imgsz {bs['imgsz']}, device {device}. Weights: `{best.relative_to(REPO_ROOT)}`\n\n"
          f"{per_class_table(metrics, classes, val_counts)}\n\n{BOOTSTRAP_BANNER}\n")
    out = cfg["paths"]["reports"] / f"bootstrap_{args.run}_metrics.md"
    out.write_text(md)
    print(md)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
