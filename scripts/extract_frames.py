"""Stage 1: extract frames from data/raw_videos/<session>/*.mp4 into a
session-split dataset.

Usage:
    python scripts/extract_frames.py                    # uses config.yaml
    python scripts/extract_frames.py --config other.yaml
    python scripts/extract_frames.py --overwrite        # redo extraction

Output:
    <dataset>/images/{train,val,test,shift_test}/<session>__<video>__f<frame>.jpg
    <dataset>/manifest.csv          one row per kept frame
    <reports>/frame_extraction_summary.md
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

from akkuwatch.config import load_config
from akkuwatch.frames import assign_splits, extract_session, find_sessions, summary_table

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=REPO_ROOT / "config.yaml")
    parser.add_argument("--overwrite", action="store_true", help="delete previously extracted images first")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fe = cfg["frame_extraction"]
    raw_dir, dataset_dir = cfg["paths"]["raw_videos"], cfg["paths"]["dataset"]
    images_root, labels_root = dataset_dir / "images", dataset_dir / "labels"
    smoke = cfg.get("smoke_test", False)

    if smoke:
        print("*** SMOKE TEST DATA - synthetic, NOT FOR EVALUATION ***\n")

    sessions = find_sessions(raw_dir, fe["video_extensions"])
    if not sessions:
        print(f"No videos found. Put them in {raw_dir}/<session_name>/")
        return 1

    # Splits are checked BEFORE any file is written, so a config mistake
    # never leaves a half-finished dataset behind.
    try:
        assignment, warnings = assign_splits(list(sessions), cfg["splits"])
    except ValueError as e:
        print(f"Split config error: {e}")
        return 1
    for w in warnings:
        print(f"WARNING: {w}")

    # Protect existing work: re-extracting can rename frames, which would
    # orphan any labels already made for them.
    if labels_root.exists() and any(labels_root.rglob("*.txt")):
        print(f"Refusing to run: labels already exist in {labels_root}. Move them away first.")
        return 1
    if images_root.exists() and any(images_root.rglob("*.jpg")):
        if not args.overwrite:
            print(f"Frames already exist in {images_root}. Use --overwrite to redo extraction.")
            return 1
        shutil.rmtree(images_root)

    all_stats, manifest = [], []
    for session, videos in sessions.items():
        split = assignment[session]
        print(f"[{split:>10}] {session}: {len(videos)} video(s) ...", flush=True)
        stats, rows = extract_session(session, split, videos, images_root, fe)
        all_stats.append(stats)
        manifest += rows

    dataset_dir.mkdir(parents=True, exist_ok=True)
    with open(dataset_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "split", "session", "video", "frame_index", "blur_score"])
        writer.writeheader()
        writer.writerows(manifest)

    table = summary_table(all_stats)
    print("\n" + table)

    title = "# Frame extraction summary"
    if smoke:
        title += "\n\n**SMOKE TEST — synthetic data, NOT FOR EVALUATION.**"
    settings = (f"Settings: every {fe['every_n_frames']} frames, blur threshold {fe['blur_threshold']}, "
                f"dHash {fe['hash_size']}x{fe['hash_size']}, hamming <= {fe['hamming_threshold']} = duplicate.")
    reports_dir = cfg["paths"]["reports"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "frame_extraction_summary.md"
    report_path.write_text(f"{title}\n\n{settings}\n\n{table}\n")
    print(f"\nWrote {len(manifest)} frames, manifest and {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
