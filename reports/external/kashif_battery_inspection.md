# Export inspection: kashif_battery

Folder: `/Users/kaif/akkunet/data/external/kashif_battery`

Classes in data.yaml (ID order): `Automotive Battery`, `Bike Battery`, `Dry Cell`, `Laptop Battery`, `Smartphone Battery`

| split | images | empty (no boxes) | augmented copies | polygon labels | image sizes (top 3) |
|---|---:|---:|---:|---:|---|
| train | 5725 | 21 | 2281 imgs from 473 sources | 4109 | 640x640 (5725) |
| valid | 925 | 3 | 39 imgs from 16 sources | 863 | 640x640 (925) |
| test | 687 | 9 | 74 imgs from 33 sources | 647 | 640x640 (687) |

Boxes per class:

| class | train | valid | test |
|---|---:|---:|---:|
| Automotive Battery | 1895 | 153 | 72 |
| Bike Battery | 928 | 203 | 360 |
| Dry Cell | 2136 | 418 | 143 |
| Laptop Battery | 1818 | 326 | 168 |
| Smartphone Battery | 583 | 156 | 75 |

Near-duplicates of a train image (dHash <= 10 bits): valid: 234/925, test: 219/687

**Every image is 640x640: probably resized (check below whether it was stretched, which distorts object shapes).**

Roboflow preprocessing / augmentation (from README.roboflow.txt):

```
The following pre-processing was applied to each image:

No image augmentation techniques were applied.
```

## Manual follow-up checks (2026-10-17)

- Fork in our own Roboflow workspace (version 1); as with Battery and Vape,
  augmentation (mirror, 90° rotation, greyscale, cutout squares) and resizing were
  baked in upstream. 154/300 sampled images have letterbox bars.
- **X-ray images:** 548 of 2,180 Laptop Battery images (25%) are X-ray baggage
  scans (`xray_*` filenames). No other class has X-ray images.
- Same-name groups of 78-123 images (`frame1105_jpg`, ...) are video frames from
  different sources sharing a name, not augmentations.
- Leakage from train (incl. mirrors/90° rotations): 255/925 valid (28%),
  230/687 test (33%).
- Most labels are polygons (4,109 train), converted to bounding boxes on import.
