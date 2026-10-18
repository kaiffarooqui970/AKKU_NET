# AkkuNet — battery detection demo for waste facilities

A computer-vision demo that spots lithium-battery items (vapes, power banks,
battery packs, tool batteries, loose cells) in household waste on the tipping
floor, **before** they reach the bunker or compactor and start fires.

> Status: Stage 0 (setup) and Stage 1 (frame extraction) are done. Part B2:
> bootstrap model trained on public data; Part B3: live webcam demo. Later stages (labelling own footage, robustness, live demo,
> dashboard) are not built yet.
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
data/raw_photos/       own photos for TRAINING ONLY: train_only_<YYYY-MM-DD>[_note]/ (git-ignored)
data/dataset/          extracted frames + manifest.csv (git-ignored)
data/external/         public dataset exports as downloaded (git-ignored), see SOURCES.md
data/bootstrap/        public data imported for the bootstrap model only (git-ignored)
configs/class_maps/    per public dataset: source class -> our class / null (drop)
data/smoke_test/       synthetic smoke-test data, NOT FOR EVALUATION (git-ignored)
src/akkunet/           the Python package (config, device, frames, ...)
scripts/               command-line entry points
notebooks/             Colab training notebook (Stage 3)
tests/                 pytest
reports/               generated markdown reports
docs/                  guides: labelling_guide.md, robustness_checks.md, fire_escalation.md
data/escalation/       escalation event log, snapshots, hard-negative candidates (git-ignored)
.env                   local secrets (Telegram); never committed, template .env.example
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

**Own photos (train only):** to give power_bank / tool_battery some training
data, put JPEG photos in `data/raw_photos/train_only_<YYYY-MM-DD>[_note]/`
(e.g. `train_only_2026-10-20_powerbanks`). They get the same blur and duplicate
filters and always land in train. Badly named folders, or listing one under
val/test/shift_test, stop the script. Don't reuse these objects/backgrounds in
test recordings.

### Bootstrap model on public data (Part B2)
A first model for the webcam demo, trained only on public Roboflow data. Its
scores say nothing about the tipping floor. Download each dataset as a YOLOv8
export into `data/external/<name>/` (names in `SOURCES.md`), then:

```bash
python scripts/inspect_export.py battery_and_vape           # splits, sizes, augmentation, leaks
python scripts/box_contact_sheet.py battery_and_vape battery  # what does a class really contain?
python scripts/check_vape_person.py battery_and_vape "vape - v1 2024-08-06 1-47am"  # face-shortcut check
python scripts/build_bootstrap.py                           # drop rules, group near-copies, 85/15 split
python scripts/train_bootstrap.py bootstrap_v1 --dry-run    # print the command
python scripts/train_bootstrap.py bootstrap_v1 --benchmark  # estimate run time
python scripts/train_bootstrap.py bootstrap_v1              # train + per-class report
```

`build_bootstrap.py` refuses while any class is unmapped or `PENDING`. It drops
images by the `drop_images` rules in each class map and lists every dropped
file. The exports' own splits leak copies, so it ignores them: all kept images
are pooled, near-copies (same source name, or mirrored, rotated or greyscale
duplicates) are grouped, and whole groups are split about 85% train / 15%
bootstrap_val per class. See `reports/bootstrap_dataset.md` and
`reports/finding_vape_cloud_labels.md`.

Training runs for hours. Start it in the background so the Mac doesn't sleep:

```bash
mkdir -p runs/bootstrap/bootstrap_v1
nohup caffeinate -i python scripts/train_bootstrap.py bootstrap_v1 > runs/bootstrap/bootstrap_v1/train.log 2>&1 &
python scripts/train_bootstrap.py bootstrap_v1 --resume     # if it stopped: continue from weights/last.pt
```

### Live demo (Part B3)
```bash
python -m akkunet.app                                # webcam 0, weights from config demo.weights
python -m akkunet.app --source clip.mp4 --record out/annotated.mp4
python -m akkunet.app --weights yolo11n.pt --alert-classes person   # check with a stock COCO model
```
Each box shows `#track class confidence`. A track raises **one** alert (red box,
red banner "Possible battery detected", console line) after it has been seen
in `demo.alert_min_frames` frames with confidence ≥ `demo.alert_conf`. Press q
or Esc to quit.

**Sound:** `--audio beep` (default: macOS system sound via `afplay`), `--audio voice`
(German via `say`, voice "Anna": "Achtung: möglicher Akku erkannt"; set
`demo.audio.say_class_name: true` for e.g. "Achtung: E-Zigarette erkannt") or
`--audio off`. Sounds play in the background and at most once per
`demo.audio.cooldown_seconds` (5 s). Without afplay/say (not macOS) it warns
once and rings the terminal bell. On macOS the terminal needs camera permission (System Settings ›
Privacy & Security › Camera).

### Fire/smoke escalation (human-in-the-loop)
**The software never calls 112 or any number.** In the demo window, `s` toggles
SIMULATED smoke. After `escalation.trigger_min_frames` frames, the primary
contact gets an alert with snapshot and buttons "Confirm fire" / "False alarm".
Confirm → they receive 112, the address and a text to read out, and the site
alarm sounds. No answer within 90 s → second contact + site alarm. False alarm
→ snapshot saved as a hard-negative candidate. Everything is logged to
`data/escalation/events.sqlite`. The default channel is `console` (demo, keys `c`/`f`
answer as the contact). `--escalation telegram` needs a local `.env` (see
`.env.example`). Design and the requirements for a real site (certified fire
alarm system, DIN 14675): `docs/fire_escalation.md`.

### bootstrap_v2 on Kaggle (free GPU)
Same recipe as bootstrap_v1, plus 3,000 public COCO "no battery" images
(licence-filtered, empty labels, train only). A held-out 500-image false-alarm
test compares v1 and v2.
```bash
python scripts/build_kaggle_package.py      # -> data/kaggle/akkunet_bootstrap_v2.zip + bootstrap_v2.ipynb
```
Upload the zip as a **private** Kaggle dataset. Import the notebook, set GPU and
Internet on, add the dataset, then Run all. Download `bootstrap_v2_results.zip`
and unzip it into `runs/bootstrap/`. `--include-own-negatives` adds your own
negative frames, which show a person: private datasets only.

### Fire/smoke model fire_v1 (prepared, not trained)
A separate model (classes `smoke`, `fire`) on D-Fire (CC0, `SOURCES.md`). The
authors' test split leaks, so it is re-split by near-copy group
(`reports/fire_dataset.md`). No images are copied: lists point at
`data/external/dfire/`.
```bash
python scripts/inspect_dfire.py              # counts, leakage, leak-pair sheet
python scripts/build_fire_dataset.py         # group split -> data/fire/fire_v1.yaml
python scripts/train_fire.py fire_v1 --dry-run
python scripts/train_fire.py fire_v1         # GPU, one job at a time (after the battery runs)
```
The report gives per-class metrics plus the combined `fire_or_smoke` metric.

### Smoke test (pipeline check only)
```bash
python scripts/make_smoke_data.py
python scripts/extract_frames.py --config data/smoke_test/config_smoke.yaml
```
Synthetic random-pattern videos with no batteries. They only show that the
code runs. **Never report metrics on them.**
