"""Stage 1: turn recording-session videos into a clean, leak-free set of frames.

Pipeline per session:
  1. sample every N-th frame from each video
  2. drop blurry frames   (variance of the Laplacian below a threshold)
  3. drop near-duplicates (difference hash close to a frame we already kept)
  4. save the rest as JPEGs into images/<split>/

Splits are assigned per SESSION. Frames from one video are nearly identical,
so putting some in train and some in test would let the model "recognise"
the test images instead of generalising (data leakage, inflated scores).
"""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

SPLITS = ("train", "val", "test", "shift_test")

# Origin of every frame in manifest.csv. Frames from my own recordings are
# "own"; public datasets (added to train only) use their dataset name.
OWN_SOURCE = "own"
MANIFEST_FIELDS = ["image", "split", "source", "session", "video", "frame_index", "blur_score"]


# ---------------------------------------------------------------- image checks

def laplacian_variance(image: np.ndarray, long_side: int = 640) -> float:
    """Sharpness score. The Laplacian responds to edges; a blurry image has few
    strong edges, so the variance of the Laplacian is low.

    The image is first resized so its long side is `long_side` px, because the
    same scene at 4K has a different score than at 720p.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape
    scale = long_side / max(h, w)
    gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def dhash(image: np.ndarray, hash_size: int = 16) -> np.ndarray:
    """Difference hash (a perceptual hash).

    Shrink to (hash_size+1) x hash_size grey pixels, then for each pixel record
    whether it is brighter than its right neighbour. Result: hash_size^2 bits
    that stay almost the same under small changes (compression, slight camera
    shake) but differ when the scene really changes.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    """Number of bits that differ between two hashes."""
    return int(np.count_nonzero(a != b))


# ---------------------------------------------------------------- sessions & splits

def find_sessions(raw_dir: Path, video_extensions: list[str]) -> dict[str, list[Path]]:
    """Map session folder name -> sorted list of videos inside it."""
    exts = {e.lower() for e in video_extensions}
    sessions = {}
    for folder in sorted(p for p in Path(raw_dir).iterdir() if p.is_dir()):
        videos = sorted(f for f in folder.iterdir() if f.suffix.lower() in exts)
        if videos:
            sessions[folder.name] = videos
    return sessions


def assign_splits(session_names: list[str], split_cfg: dict) -> tuple[dict[str, str], list[str]]:
    """Decide which split each session belongs to.

    Sessions listed in the config go to val / test / shift_test; everything
    else goes to train. Returns (session -> split, list of warnings).
    Raises ValueError for mistakes that would silently corrupt the evaluation.
    """
    listed = {
        "val": split_cfg.get("val_sessions") or [],
        "test": split_cfg.get("test_sessions") or [],
        "shift_test": split_cfg.get("shift_test_sessions") or [],
    }
    known = set(session_names)

    # A typo in the config would otherwise send that session to train.
    for split, names in listed.items():
        missing = [n for n in names if n not in known]
        if missing:
            raise ValueError(f"{split}_sessions lists unknown session(s): {missing}. Found: {sorted(known)}")

    # The same session in two splits = leakage.
    seen = {}
    for split, names in listed.items():
        for n in names:
            if n in seen:
                raise ValueError(f"Session '{n}' is in both {seen[n]} and {split}.")
            seen[n] = split

    if not listed["shift_test"]:
        raise ValueError(
            "No shift_test session configured. Reserve at least one whole session "
            "with a different background/lighting under splits.shift_test_sessions."
        )

    assignment = {n: seen.get(n, "train") for n in session_names}
    if "train" not in assignment.values():
        raise ValueError("Every session is assigned to val/test/shift_test; nothing left to train on.")

    warnings = [f"No {s} sessions configured." for s in ("val", "test") if not listed[s]]
    return assignment, warnings


# ---------------------------------------------------------------- extraction

@dataclass
class SessionStats:
    session: str
    split: str
    videos: int = 0
    sampled: int = 0
    blurry: int = 0
    duplicates: int = 0
    kept: int = 0
    blur_scores: list[float] = field(default_factory=list)

    @property
    def median_blur(self) -> float:
        return float(np.median(self.blur_scores)) if self.blur_scores else float("nan")


def extract_session(
    session: str, split: str, videos: list[Path], images_root: Path, fe_cfg: dict
) -> tuple[SessionStats, list[dict]]:
    """Run sample -> blur filter -> duplicate filter -> save for one session.

    Duplicates are checked against all frames kept so far in this session
    (across all its videos), not only the previous one, because the camera
    often returns to the same view. Sessions never need comparing with each
    other since they always end up in a split of their own.
    """
    out_dir = images_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = SessionStats(session=session, split=split, videos=len(videos))
    kept_hashes: list[np.ndarray] = []
    manifest_rows = []

    for video in videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video}")

        frame_idx = -1
        while True:
            frame_idx += 1
            # grab() advances without decoding to a numpy array; only frames we
            # actually sample are fully retrieved (much faster).
            if not cap.grab():
                break
            if frame_idx % fe_cfg["every_n_frames"] != 0:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            stats.sampled += 1

            score = laplacian_variance(frame, fe_cfg["blur_measure_long_side"])
            stats.blur_scores.append(score)
            if score < fe_cfg["blur_threshold"]:
                stats.blurry += 1
                continue

            h = dhash(frame, fe_cfg["hash_size"])
            if any(hamming(h, k) <= fe_cfg["hamming_threshold"] for k in kept_hashes):
                stats.duplicates += 1
                continue
            kept_hashes.append(h)

            # Session name in the filename makes every frame traceable to its
            # recording and keeps names unique once all splits are combined.
            name = f"{session}__{video.stem}__f{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, fe_cfg["jpeg_quality"]])
            stats.kept += 1
            manifest_rows.append({
                "image": f"images/{split}/{name}",
                "split": split,
                "source": OWN_SOURCE,
                "session": session,
                "video": video.name,
                "frame_index": frame_idx,
                "blur_score": round(score, 1),
            })
        cap.release()

    return stats, manifest_rows


# ---------------------------------------------------------------- summary

def summary_table(all_stats: list[SessionStats]) -> str:
    """Markdown table: one row per session, then totals per split."""
    lines = [
        "| session | split | videos | sampled | blurry removed | duplicates removed | kept | median blur score |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in sorted(all_stats, key=lambda s: (SPLITS.index(s.split), s.session)):
        lines.append(f"| {s.session} | {s.split} | {s.videos} | {s.sampled} | {s.blurry} "
                     f"| {s.duplicates} | {s.kept} | {s.median_blur:.0f} |")

    lines += ["", "| split | sessions | frames kept |", "|---|---:|---:|"]
    for split in SPLITS:
        members = [s for s in all_stats if s.split == split]
        lines.append(f"| {split} | {len(members)} | {sum(s.kept for s in members)} |")
    return "\n".join(lines)
