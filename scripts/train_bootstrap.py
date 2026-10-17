"""Part B2 step 6-7: train the bootstrap model on imported public data.

Usage:
    python scripts/train_bootstrap.py bootstrap_v1 --dry-run        # print command, write lists
    python scripts/train_bootstrap.py bootstrap_v1 --benchmark      # time 10% of one epoch -> estimate
    python scripts/train_bootstrap.py bootstrap_v1                  # train + per-class report
    python scripts/train_bootstrap.py bootstrap_v1 --resume         # continue from weights/last.pt
    python scripts/train_bootstrap.py bootstrap_v1 --report-only    # per-class report from best.pt
    python scripts/train_bootstrap.py bootstrap_v2 --init runs/bootstrap/bootstrap_v1/weights/best.pt --negatives
                                       # fine-tune, with own hard negatives (empty labels) added to TRAIN only

Run scripts/build_bootstrap.py first. Train / val = the group split it made
(val = bootstrap_val).

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
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--resume", action="store_true", help="continue an interrupted run from last.pt")
    group.add_argument("--report-only", action="store_true", help="skip training, evaluate best.pt")
    parser.add_argument("--init", help="start from these weights instead of bootstrap.model (fine-tuning)")
    parser.add_argument("--negatives", action="store_true",
                        help="add data/negatives/ frames (empty labels) to the TRAIN list only")
    parser.add_argument("--epochs", type=int, help="override bootstrap.epochs (e.g. shorter fine-tune)")
    parser.add_argument("--config", default=REPO_ROOT / "config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bs, classes = dict(cfg["bootstrap"]), cfg["classes"]
    if args.init:
        bs["model"] = args.init
    if args.epochs:
        bs["epochs"] = args.epochs
    try:
        extra = []
        if args.negatives:
            from akkunet.negatives import negative_train_images
            extra = negative_train_images(cfg["paths"]["negatives"])
            print(f"Adding {len(extra)} hard-negative frames (empty labels) to TRAIN only.")
        data_yaml = write_run_dataset(cfg["paths"]["bootstrap"], args.run, classes,
                                      train_sources=bs["datasets"], val_sources=bs["datasets"],
                                      extra_train_images=extra)
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
        # Time only the training loop and one validation pass (callbacks), not
        # start-up, dataset scanning or plotting, then scale to a full run.
        marks = {}
        model = YOLO(bs["model"])
        model.add_callback("on_train_epoch_start", lambda t: marks.__setitem__("train0", time.time()))
        model.add_callback("on_train_epoch_end", lambda t: marks.__setitem__("train1", time.time()))
        model.add_callback("on_val_start", lambda v: marks.__setitem__("val0", time.time()))
        model.add_callback("on_val_end", lambda v: marks.__setitem__("val1", time.time()))
        model.train(**{**train_args, "epochs": 1, "fraction": BENCHMARK_FRACTION, "plots": False,
                       "name": f"{args.run}_benchmark", "exist_ok": True})
        train_epoch = (marks["train1"] - marks["train0"]) / BENCHMARK_FRACTION
        val_pass = marks["val1"] - marks["val0"]
        per_epoch = train_epoch + val_pass
        print(f"\nTraining loop ~{train_epoch / 60:.1f} min per full epoch (scaled from {BENCHMARK_FRACTION:.0%}), "
              f"validation ~{val_pass:.0f} s per epoch -> ~{per_epoch / 60:.1f} min per epoch.\n"
              f"{bs['epochs']} epochs: up to ~{per_epoch * bs['epochs'] / 3600:.1f} h. Early stopping "
              f"(patience {bs['patience']}) usually ends it sooner; e.g. 40 epochs = "
              f"~{per_epoch * 40 / 3600:.1f} h.")
        return 0

    run_dir = REPO_ROOT / "runs" / "bootstrap" / args.run
    if args.resume:
        # Ultralytics restores every setting (epochs, optimizer state, ...) from last.pt.
        YOLO(str(run_dir / "weights" / "last.pt")).train(resume=True)
    elif not args.report_only:
        # The folder may already exist because the shell log (train.log) is written
        # into it. Reuse it only while it holds no weights, so a finished run is
        # never overwritten (and Ultralytics doesn't silently switch to <run>2).
        if (run_dir / "weights").exists():
            print(f"{run_dir} already has weights. Use --resume, --report-only or a new run name.")
            return 1
        YOLO(bs["model"]).train(**train_args, exist_ok=True)
    best = run_dir / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(data=str(data_yaml), imgsz=bs["imgsz"], device=device,
                                  project=str(REPO_ROOT / "runs" / "bootstrap"), name=f"{args.run}_val", exist_ok=True)
    val_counts = count_boxes(data_yaml.parent / "lists" / f"{args.run}_val.txt", classes)
    md = (f"# Bootstrap metrics: {args.run}\n\n{BOOTSTRAP_BANNER}\n\n"
          f"Public data only: {bs['datasets']}. Train {n_train} images, bootstrap_val {n_val} images "
          f"(split by near-copy group, see reports/bootstrap_dataset.md; val is still the same kind of "
          f"internet photos as train). "
          + (f"Plus {len(extra)} own hard-negative frames (empty labels) in train only. " if extra else "")
          + f"Model {bs['model']}, imgsz {bs['imgsz']}, device {device}. Weights: `{best.relative_to(REPO_ROOT)}`\n\n"
          f"{per_class_table(metrics, classes, val_counts)}\n\n{BOOTSTRAP_BANNER}\n")
    out = cfg["paths"]["reports"] / f"bootstrap_{args.run}_metrics.md"
    out.write_text(md)
    print(md)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
