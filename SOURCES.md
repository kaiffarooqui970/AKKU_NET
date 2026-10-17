# Data sources

Every image that is not our own footage is listed here. Public data is used for
the **bootstrap model only** (Part B2): it never enters the val, test or
shift_test splits of our own recordings, which live in a separate folder
(`data/dataset/` vs `data/bootstrap/`).

## Licence caveat (applies to all Roboflow Universe datasets below)

Licences on Roboflow Universe are **declared by the uploader**. Several of these
datasets visibly contain scraped product photos and adverts, so whether the
uploader had the right to license those images cannot be verified. "CC BY 4.0"
below therefore means **"CC BY 4.0 as declared by the uploader"**. Before any
commercial use, re-check or replace these images.

CC BY 4.0 requires attribution: credit the dataset name, its author/workspace
and the licence wherever the model or its outputs are shown.

## Used for the bootstrap model

| folder (`data/external/…`) | dataset | author / workspace | licence | images (project page) | our classes | notes |
|---|---|---|---|---:|---|---|
| `battery_and_vape` | Battery and Vape | david (Roboflow Universe); downloaded as a fork, `kaifs-workspace-jfhy5/battery-and-vape-cp2zi` v1 | CC BY 4.0 as declared by the uploader | 9,432 (6,461 kept) | vape; battery → loose_cell | see label-quality notes below |
| `kashif_battery` | Battery Detection | Kashif (Roboflow Universe); fork `kaifs-workspace-jfhy5/battery-detection-7dnqh-yk0pu` v1 | CC BY 4.0 as declared by the uploader | 7,337 (3,400 kept) | Laptop/Smartphone Battery → battery_pack; Dry Cell → loose_cell; Automotive/Bike Battery dropped | X-ray scans dropped |

### Label-quality notes and what was removed

Counts per rule are in `reports/bootstrap_dataset.md`, and every dropped file is
in `reports/external/bootstrap_dropped.csv`.

- **Battery and Vape, `battery` → `loose_cell`: known label noise.** A 20-box
  sample was mostly loose cells (AA/AAA, C/D, 18650, 9V), but 3–4 of 20 (~15–20%)
  were flat smartphone battery packs, which should be `battery_pack`. This was
  accepted for the bootstrap model (decision 2026-10-17). Expect some
  battery_pack ↔ loose_cell confusion. One sampled box was clip art, and some
  boxes cover several cells at once.
- **Battery and Vape, `vape`: about half the boxes mark the vapour cloud, not the
  device.** Only the `<n>_jpg` and `IMG_<n>` filename groups (device boxes) are
  kept. 2,158 cloud images and 813 unreviewed mixed images were dropped. See
  `reports/finding_vape_cloud_labels.md`.
- **Kashif:** 548 X-ray baggage scans (`xray_*`, all Laptop Battery) were dropped.
  3,389 images containing only car or bike batteries were dropped too (not kept
  as background).
- **Both:** augmentation was baked in by the original uploaders (mirrors, 90°
  rotations, greyscale, crops, cutout), even though our export lists none. All
  images are 640×640, and many are letterboxed. The exports' own train/valid
  splits leaked near-copies, so the data was re-split by source image.

## Not downloaded (for now)

- `gvl610_battery`: battery (gvl610, Roboflow Universe), CC BY 4.0 as declared,
  8,109 images. Planned white-sheet background experiment, skipped for now
  (decision 2026-10-17). Its class map stays in `configs/class_maps/`.

Image counts are the project totals on each page; the downloaded version can
differ. Preprocessing, augmentation, leakage and other findings for each download are
recorded in `reports/external/<folder>_inspection.md`.

## Pretrained weights

- `yolo11n.pt` (Ultralytics, trained on COCO), AGPL-3.0, see README licence note.

## Checked and rejected (B1)

TACO (2 battery boxes in 1,500 images), ZeroWaste (CC BY-NC 4.0), XBAT+ (X-ray
classification), RecyBat24 (conflicting licence: CC BY 4.0 vs CC BY-NC-ND 4.0),
retastaz "battery" (no versions), Waste-battery-only (boxes mark the label, not
the battery).
