"""Create a tiny SYNTHETIC dataset to check the pipeline runs end to end.

Everything goes into data/smoke_test/, fully separate from real data, with its
own config_smoke.yaml (marked smoke_test: true so all outputs carry a warning).
The videos contain random patterns, not batteries: NOT FOR EVALUATION.

Usage:
    python scripts/make_smoke_data.py
    python scripts/extract_frames.py --config data/smoke_test/config_smoke.yaml
"""
import shutil
from pathlib import Path

import yaml

from akkuwatch.synthetic import make_synthetic_video

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_DIR = REPO_ROOT / "data" / "smoke_test"

SESSIONS = {
    "SMOKE_session_01": "train",
    "SMOKE_session_02": "train",
    "SMOKE_session_03": "val",
    "SMOKE_session_04": "test",
    "SMOKE_session_05_shift": "shift_test",
}


def main() -> None:
    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)
    raw = SMOKE_DIR / "raw_videos"

    for i, session in enumerate(SESSIONS):
        make_synthetic_video(raw / session / "clip.mp4", seed=i)

    # Same settings as the real config, but pointing at the smoke folders.
    cfg = yaml.safe_load((REPO_ROOT / "config.yaml").read_text())
    cfg["smoke_test"] = True
    cfg["paths"] = {"raw_videos": "raw_videos", "dataset": "dataset", "reports": "reports"}
    cfg["frame_extraction"]["every_n_frames"] = 1  # tiny videos: keep every frame
    cfg["splits"] = {
        f"{split}_sessions": [s for s, sp in SESSIONS.items() if sp == split]
        for split in ("val", "test", "shift_test")
    }
    (SMOKE_DIR / "config_smoke.yaml").write_text(
        "# SMOKE TEST CONFIG - synthetic data, NOT FOR EVALUATION\n" + yaml.safe_dump(cfg, sort_keys=False)
    )
    (SMOKE_DIR / "README_NOT_FOR_EVALUATION.txt").write_text(
        "Synthetic random-pattern videos used only to check that the code runs.\n"
        "They contain no batteries. Never train, evaluate or report metrics on them.\n"
    )
    print(f"Wrote {len(SESSIONS)} synthetic sessions to {SMOKE_DIR}")


if __name__ == "__main__":
    main()
