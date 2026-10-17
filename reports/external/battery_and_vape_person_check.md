# Face-shortcut check: battery_and_vape / vape - v1 2024-08-06 1-47am

COCO person detector (yolo11n.pt), confidence >= 0.4. 4930 images with vape boxes.

| largest person in image | images | share |
|---|---:|---:|
| no person | 2100 | 43% |
| small (<10%) | 40 | 1% |
| medium (10-40%) | 794 | 16% |
| large (>=40%) | 1996 | 40% |

Vape boxes with their centre inside a person box: 2586/6054 (43%).

Per split (images with any person / images):

- test: 285/488
- train: 1998/3460
- valid: 547/982

## Manual follow-up (2026-10-17)

**Many "vape" boxes mark the vapour cloud, not the device.** About 20 of 40
randomly sampled vape boxes were clouds. The filename prefix separates them in
the samples checked (10 of 10 per group):

| filename group | vape images | box content (sample of 10) | no person | small | medium | large |
|---|---:|---|---:|---:|---:|---:|
| `Image_<n>_jpg…` | 1,976 | clouds 10/10 | 759 | 8 | 273 | 936 |
| `Download-<n>-_mp4…` | 182 | clouds 10/10 | 87 | 2 | 32 | 61 |
| `<n>_jpg…` | 1,639 | devices 10/10 | 651 | 14 | 320 | 654 |
| `IMG_<n>…` | 320 | devices 10/10 | 188 | 1 | 52 | 79 |
| other | 813 | mixed, ~3/10 clouds | 415 | 15 | 117 | 266 |

The COCO "person" class also fires on a hand alone, so "contains a person" is
not the same as "shows a face". Without the two cloud groups: 2,772 images, of
which 1,364 / 3,387 vape boxes have their centre inside a person box.
