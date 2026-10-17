"""Hard negatives: own recordings with NO battery in view (faces, hands, lookalikes).

Frames are extracted like Stage 1 (every N-th frame, blur filter, duplicate
filter) and get EMPTY label files: "nothing to detect here". They are
TRAIN ONLY by construction: they live in their own folder (data/negatives/),
are never part of data/dataset/ or data/bootstrap/ and its splits, and the
training script can only add them to the train list. The recordings show
people, so everything stays in git-ignored data/.
"""
import csv
import re
from pathlib import Path

from akkunet.frames import MANIFEST_FIELDS, extract_session

NEGATIVES_SOURCE = "own_negatives"
NEGATIVE_SESSION_PATTERN = re.compile(r"^negatives_\d{4}-\d{2}-\d{2}(_[a-z0-9_]+)?$")
SPLIT = "train"   # the only split negatives can ever be in


def find_negative_sessions(raw_dir: Path, video_extensions: list[str]) -> dict[str, list[Path]]:
    """negatives_<date>[_note] folder -> its videos. Other folder names are an error."""
    if not Path(raw_dir).is_dir():
        return {}
    exts = {e.lower() for e in video_extensions}
    sessions = {}
    for folder in sorted(p for p in Path(raw_dir).iterdir() if p.is_dir()):
        if not NEGATIVE_SESSION_PATTERN.match(folder.name):
            raise ValueError(f"Negatives folder '{folder.name}' must be named negatives_<YYYY-MM-DD>[_note]")
        videos = sorted(f for f in folder.iterdir() if f.suffix.lower() in exts)
        if videos:
            sessions[folder.name] = videos
    return sessions


def extract_negatives(raw_dir: Path, out_dir: Path, fe_cfg: dict):
    """(Re)build out_dir/{images,labels}/train/ + manifest.csv from all negative sessions.

    Returns the per-session stats. Every kept frame gets an empty label file.
    """
    sessions = find_negative_sessions(raw_dir, fe_cfg["video_extensions"])
    for sub in ("images", "labels"):
        for old in (out_dir / sub / SPLIT).glob("*"):
            old.unlink()
    all_stats, rows = [], []
    for session, videos in sessions.items():
        stats, session_rows = extract_session(session, SPLIT, videos, out_dir / "images", fe_cfg)
        all_stats.append(stats)
        rows += session_rows
    labels = out_dir / "labels" / SPLIT
    labels.mkdir(parents=True, exist_ok=True)
    for r in rows:
        r["source"] = NEGATIVES_SOURCE
        (labels / (Path(r["image"]).stem + ".txt")).write_text("")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return all_stats, rows


def negative_train_images(out_dir: Path) -> list[str]:
    """Absolute image paths for a training list. Refuses anything outside train."""
    manifest = Path(out_dir) / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"No negatives yet ({manifest}). Run scripts/extract_negatives.py first.")
    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    bad = [r["image"] for r in rows if r["split"] != SPLIT or r["source"] != NEGATIVES_SOURCE]
    if bad:
        raise ValueError(f"Negatives manifest has rows outside train/{NEGATIVES_SOURCE}: {bad[:3]}")
    for r in rows:
        label = Path(out_dir) / r["image"].replace("images/", "labels/", 1)
        if Path(label).with_suffix(".txt").read_text().strip():
            raise ValueError(f"Negative frame has a non-empty label: {label}")
    return [str(Path(out_dir) / r["image"]) for r in rows]
