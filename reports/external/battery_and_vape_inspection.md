# Export inspection: battery_and_vape

Folder: `/Users/kaif/akkunet/data/external/battery_and_vape`

Classes in data.yaml (ID order): `battery - v1 2024-06-26 4-12am`, `vape - v1 2024-08-06 1-47am`

| split | images | empty (no boxes) | augmented copies | polygon labels | image sizes (top 3) |
|---|---:|---:|---:|---:|---|
| train | 7054 | 0 | 5889 imgs from 1673 sources | 735 | 640x640 (7054) |
| valid | 1609 | 0 | 630 imgs from 260 sources | 156 | 640x640 (1609) |
| test | 769 | 0 | 219 imgs from 99 sources | 81 | 640x640 (769) |

Boxes per class:

| class | train | valid | test |
|---|---:|---:|---:|
| battery - v1 2024-06-26 4-12am | 8732 | 1629 | 726 |
| vape - v1 2024-08-06 1-47am | 4255 | 1195 | 604 |

Near-duplicates of a train image (dHash <= 10 bits): valid: 440/1609, test: 248/769

**Every image is 640x640: probably resized (check below whether it was stretched, which distorts object shapes).**

Roboflow preprocessing / augmentation (from README.roboflow.txt):

```
The following pre-processing was applied to each image:

No image augmentation techniques were applied.
```

## Manual follow-up checks (2026-10-17)

- This is a fork in our own Roboflow workspace (version 1). The README lists no
  preprocessing or augmentation **for this version**, but the images were already
  augmented and resized upstream: pairs with the same source name are mirrored,
  rotated 90°, greyscaled or cropped copies; every image is 640x640 and about a
  third (96/300 sampled) have letterbox bars, so the rest are probably
  stretched or cropped.
- Leakage from train: 603/1609 valid (37%) and 333/769 test (43%) are
  near-duplicates of a train image when mirrors and 90° rotations are counted;
  1148/1609 valid images share a source filename with a train image (filenames
  can collide, so this is an upper bound). bootstrap_val scores will be optimistic.
- 735 train labels are polygons (converted to their bounding box on import).
