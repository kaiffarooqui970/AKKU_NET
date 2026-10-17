# AkkuNet — battery detection demo for waste facilities

A computer-vision demo that spots lithium-battery items (vapes, power banks,
battery packs, tool batteries, loose cells) in household waste on the tipping
floor, **before** they reach the bunker or compactor and start fires.

> Status: Stage 0 (setup) and Stage 1 (frame extraction) are done. Later stages
> (labelling, training, robustness, live demo, dashboard) are not built yet.
> Thermal detection is planned for phase 2 (needs real thermal hardware).

## Licence note: Ultralytics YOLO is AGPL-3.0

This demo uses Ultralytics YOLO, which is licensed under **AGPL-3.0**. That is
fine for this demo and for research. A **commercial product** (e.g. software sold
to or run as a service for a recycling facility) would need either an
**Ultralytics Enterprise licence** or a switch to a permissively licensed model.
Nothing has been changed for this yet; it is documented so the decision is made
consciously before any commercial pilot.

## Setup

```bash
# Python 3.11. Keep the venv OUTSIDE ~/Desktop (iCloud can evict files and hang imports).
python3.11 -m venv ~/.venvs/akkunet
source ~/.venvs/akkunet/bin/activate
pip install -r requirements.txt
pip install --no-deps -r requirements-nodeps.txt   # grad-cam, see below
pip install -e .          # makes `import akkunet` work from scripts/ and tests/
pytest                    # all tests use synthetic data only
python scripts/check_opencv_gui.py                 # should open a small test window
```

**Why two requirements files:** `opencv-python` (with GUI, needed for live
demo windows) and `opencv-python-headless` (no GUI) both install into the same
`cv2/` folder and break each other. grad-cam asks for the headless one, so it is
installed with `--no-deps`. `pip check` will then say *"grad-cam requires
opencv-python-headless"*. That is expected and harmless.

All settings live in `config.yaml`. Paths in it are relative to the config file.

## Folder layout

```
config.yaml            every setting (paths, classes, thresholds, splits)
data/raw_videos/       your recordings, one subfolder per session (git-ignored)
data/dataset/          extracted frames + manifest.csv (git-ignored)
data/smoke_test/       synthetic smoke-test data, NOT FOR EVALUATION (git-ignored)
src/akkunet/           the Python package (config, device, frames, ...)
scripts/               command-line entry points
notebooks/             Colab training notebook (Stage 3)
tests/                 pytest
reports/               generated markdown reports
docs/                  guides (labelling guide in Stage 2)
```

## Run order so far

### 1. Record
Film the mini tipping floor from above. Put each session's videos in its own folder:

```
data/raw_videos/session_01_tarp_daylight/IMG_0001.MOV
data/raw_videos/session_02_tarp_lamp/...
data/raw_videos/session_05_concrete_evening/...   <- different background/light
```

### 2. Assign splits in `config.yaml`
Splits are made **per session, never per frame**. Frames from one video are
almost identical, so splitting by frame would put near-copies of test images
into training and inflate scores. Unlisted sessions go to train:

```yaml
splits:
  val_sessions: [session_03_tarp_cloudy]
  test_sessions: [session_04_box_daylight]
  shift_test_sessions: [session_05_concrete_evening]   # required, at least one
```

The script stops with an error if a session name is misspelled, a session is in
two splits, no shift-test session is set, or nothing is left for training.

### 3. Extract frames
```bash
python scripts/extract_frames.py              # add --overwrite to redo
```
For each session: keep every `every_n_frames`-th frame → drop blurry ones
(Laplacian variance < `blur_threshold`) → drop near-duplicates (dHash within
`hamming_threshold` bits of a frame already kept in that session) → save to
`data/dataset/images/<split>/`. Each kept frame gets a row in `manifest.csv`
with `source=own` (public datasets added later get their dataset name). It prints a per-session table and writes it to
`reports/frame_extraction_summary.md`. The script refuses to run if labels
already exist, because re-extracting could orphan them.

**Tuning:** check the "median blur score" column and look at a few removed
frames. If good frames get dropped, lower `blur_threshold`. If too many
similar frames survive, raise `hamming_threshold`.

### Smoke test (pipeline check only)
```bash
python scripts/make_smoke_data.py
python scripts/extract_frames.py --config data/smoke_test/config_smoke.yaml
```
Synthetic random-pattern videos with no batteries. They only show that the
code runs. **Never report metrics on them.**
