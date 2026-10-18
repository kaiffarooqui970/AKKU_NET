# Bootstrap metrics: bootstrap_v1

**BOOTSTRAP ONLY — not tipping-floor performance.**

Public data only: ['battery_and_vape', 'kashif_battery']. Train 8385 images, bootstrap_val 1476 images (split by near-copy group, see reports/bootstrap_dataset.md; val is still the same kind of internet photos as train). Model yolo11n.pt, imgsz 640, device mps. Weights: `runs/bootstrap/bootstrap_v1/weights/best.pt`

| class | val boxes | precision | recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| vape | 327 | 0.863 | 0.908 | 0.925 | 0.575 |
| power_bank | 0 | – | – | – | – (no val data; untrained) |
| battery_pack | 371 | 0.830 | 0.943 | 0.926 | 0.873 |
| tool_battery | 0 | – | – | – | – (no val data; untrained) |
| loose_cell | 1895 | 0.917 | 0.887 | 0.938 | 0.768 |
| **all (classes with data)** | 2593 | 0.870 | 0.913 | 0.930 | 0.738 |

**BOOTSTRAP ONLY — not tipping-floor performance.**
