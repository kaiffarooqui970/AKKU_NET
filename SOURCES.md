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
| `battery_and_vape` | Battery and Vape | david (Roboflow Universe) | CC BY 4.0 as declared by the uploader | 9,432 | vape; battery → loose_cell **pending contact-sheet review** | vape product shots, in hand, some people vaping |
| `kashif_battery` | Battery Detection | Kashif (Roboflow Universe) | CC BY 4.0 as declared by the uploader | 7,337 | Laptop/Smartphone Battery → battery_pack; Dry Cell → loose_cell; car/bike batteries dropped | product/internet photos and adverts |

## Optional experiment (Part B2 step 6)

| folder | dataset | author / workspace | licence | images | use |
|---|---|---|---|---:|---|
| `gvl610_battery` | battery | gvl610 (Roboflow Universe) | CC BY 4.0 as declared by the uploader | 8,109 | ≤1,500 train images, to test whether the model learns its white sheet background |

Image counts are the project totals on each page; the downloaded version can
differ. The exact export version, URL, preprocessing and augmentation of each
download are recorded in `reports/external/<folder>_inspection.md`.

## Pretrained weights

- `yolo11n.pt` (Ultralytics, trained on COCO), AGPL-3.0, see README licence note.

## Checked and rejected (B1)

TACO (2 battery boxes in 1,500 images), ZeroWaste (CC BY-NC 4.0), XBAT+ (X-ray
classification), RecyBat24 (conflicting licence: CC BY 4.0 vs CC BY-NC-ND 4.0),
retastaz "battery" (no versions), Waste-battery-only (boxes mark the label, not
the battery).
