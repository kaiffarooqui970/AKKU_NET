"""fire_v1 step 2: train the separate fire/smoke model on the D-Fire group split.

Usage:
    python scripts/train_fire.py fire_v1 --dry-run        # print command and data sizes (no GPU)
    python scripts/train_fire.py fire_v1 --benchmark      # time 10% of one epoch -> estimate
    python scripts/train_fire.py fire_v1                  # train + report
    python scripts/train_fire.py fire_v1 --resume         # continue from weights/last.pt
    python scripts/train_fire.py fire_v1 --report-only    # report from best.pt

Run scripts/build_fire_dataset.py first. GPU order: one job at a time, after
the battery runs (bootstrap_v1 -> bootstrap_v2 -> fire_v1).

Output:
    runs/fire/<run>/                 Ultralytics run (weights/best.pt, plots)
    reports/fire_<run>_metrics.md    per-class P/R/mAP (smoke, fire) and combined fire_or_smoke
"""
import argparse
import sys
import time
from pathlib import Path

from akkunet.config import load_config
from akkunet.device import get_device
from akkunet.evaluate import count_boxes, per_class_table
from akkunet.fire import FIRE_BANNER

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_FRACTION = 0.1
LIMITATIONS = (
    "Known limitations:\n\n"
    "- **Night glow ambiguity:** the same bright glow is labelled `fire` in some D-Fire images and `smoke` in "
    "others, so per-class numbers mix up at night. The escalation triggers on either class; "
    "`fire_or_smoke` is the number that matters for it.\n"
    "- **Mostly outdoor data** (wildfire, streets, surveillance cameras). Dust/haze vs smoke and indoor hall "
    "footage are untested (docs/robustness_checks.md).\n"
    "- **Val is the same kind of data as train.** Near-copies are split apart, but the same fixed camera in "
    "very different light can still sit on both sides.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="run name, e.g. fire_v1")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--resume", action="store_true", help="continue an interrupted run from last.pt")
    group.add_argument("--report-only", action="store_true", help="skip training, evaluate best.pt")
    parser.add_argument("--epochs", type=int, help="override fire.epochs")
    parser.add_argument("--config", default=REPO_ROOT / "config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fc, classes = dict(cfg["fire"]), cfg["fire"]["classes"]
    if args.epochs:
        fc["epochs"] = args.epochs
    data_yaml = cfg["paths"]["fire"] / f"{args.run}.yaml"
    if not data_yaml.exists():
        print(f"{data_yaml} not found. Run: python scripts/build_fire_dataset.py --run {args.run}")
        return 1
    lists = data_yaml.parent / "lists"
    n_train = len((lists / f"{args.run}_train.txt").read_text().split())
    n_val = len((lists / f"{args.run}_val.txt").read_text().split())

    device = get_device(cfg["device"])
    project = REPO_ROOT / "runs" / "fire"
    train_args = dict(data=str(data_yaml), imgsz=fc["imgsz"], epochs=fc["epochs"], batch=fc["batch"],
                      patience=fc["patience"], seed=fc["seed"], deterministic=False, device=device,
                      project=str(project), name=args.run)
    print(f"{n_train} train images, {n_val} val images, device {device}")
    print("Equivalent command:\n  yolo detect train model=" + fc["model"] + " "
          + " ".join(f"{k}={v}" for k, v in train_args.items()))
    if args.dry_run:
        return 0

    from ultralytics import YOLO

    if args.benchmark:
        marks = {}
        model = YOLO(fc["model"])
        model.add_callback("on_train_epoch_start", lambda t: marks.__setitem__("train0", time.time()))
        model.add_callback("on_train_epoch_end", lambda t: marks.__setitem__("train1", time.time()))
        model.add_callback("on_val_start", lambda v: marks.__setitem__("val0", time.time()))
        model.add_callback("on_val_end", lambda v: marks.__setitem__("val1", time.time()))
        model.train(**{**train_args, "epochs": 1, "fraction": BENCHMARK_FRACTION, "plots": False,
                       "name": f"{args.run}_benchmark", "exist_ok": True})
        per_epoch = (marks["train1"] - marks["train0"]) / BENCHMARK_FRACTION + marks["val1"] - marks["val0"]
        print(f"\n~{per_epoch / 60:.1f} min per epoch; {fc['epochs']} epochs: up to "
              f"~{per_epoch * fc['epochs'] / 3600:.1f} h (early stopping, patience {fc['patience']}, may end sooner).")
        return 0

    run_dir = project / args.run
    if args.resume:
        YOLO(str(run_dir / "weights" / "last.pt")).train(resume=True)
    elif not args.report_only:
        if (run_dir / "weights").exists():
            print(f"{run_dir} already has weights. Use --resume, --report-only or a new run name.")
            return 1
        YOLO(fc["model"]).train(**train_args, exist_ok=True)

    best = run_dir / "weights" / "best.pt"
    val_args = dict(data=str(data_yaml), imgsz=fc["imgsz"], device=device, project=str(project), exist_ok=True)
    metrics = YOLO(str(best)).val(**val_args, name=f"{args.run}_val")
    # Combined: both classes merged into one ("is there fire OR smoke here?"), as the escalation uses it.
    combined = YOLO(str(best)).val(**val_args, name=f"{args.run}_val_combined", single_cls=True).box
    counts = count_boxes(lists / f"{args.run}_val.txt", classes)
    md = (f"# Fire/smoke metrics: {args.run}\n\n{FIRE_BANNER}\n\n"
          f"D-Fire, group split (reports/fire_dataset.md): train {n_train} images, val {n_val} images. "
          f"Model {fc['model']}, imgsz {fc['imgsz']}, device {device}. Weights: `{best.relative_to(REPO_ROOT)}`\n\n"
          f"{per_class_table(metrics, classes, counts)}\n"
          f"| **fire_or_smoke (classes merged)** | {sum(counts.values())} | {combined.mp:.3f} | {combined.mr:.3f} | "
          f"{combined.map50:.3f} | {combined.map:.3f} |\n\n{LIMITATIONS}\n{FIRE_BANNER}\n")
    out = cfg["paths"]["reports"] / f"fire_{args.run}_metrics.md"
    out.write_text(md)
    print(md)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
