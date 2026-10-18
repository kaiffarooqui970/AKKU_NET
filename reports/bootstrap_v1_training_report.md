# Training report: bootstrap_v1

**BOOTSTRAP ONLY — internet photos, not tipping-floor performance.**

- **Model:** YOLO11n (COCO-pretrained), imgsz 640, batch 16, on MPS.
- **Run:** 60 of 60 epochs in 8.4 h; early stopping never triggered, and the
  best epoch was 60.
- **Data:** public data only (battery_and_vape, kashif_battery), group split;
  see `reports/bootstrap_dataset.md`.
- **Split:** train 8,385 images, bootstrap_val 1,476 images. Val is the same
  kind of internet photos as train.
- **Weights:** `runs/bootstrap/bootstrap_v1/weights/best.pt` (local, not committed).

## Per-class metrics (bootstrap_val, best.pt)

| class | val boxes | precision | recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| vape | 327 | 0.863 | 0.908 | 0.925 | 0.575 |
| power_bank | 0 | – | – | – | – (no val data; untrained) |
| battery_pack | 371 | 0.830 | 0.943 | 0.926 | 0.873 |
| tool_battery | 0 | – | – | – | – (no val data; untrained) |
| loose_cell | 1,895 | 0.917 | 0.887 | 0.938 | 0.768 |
| **all (classes with data)** | 2,593 | 0.870 | 0.913 | 0.930 | 0.738 |

**power_bank and tool_battery are untrained.** The model can't detect them.

## Training curves

`reports/bootstrap_v1/training_curves.png`

| epoch | precision | recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 1 | 0.520 | 0.684 | 0.599 | 0.379 |
| 4 (first webcam test) | 0.714 | 0.670 | 0.718 | 0.479 |
| 30 | 0.843 | 0.870 | 0.900 | 0.705 |
| 60 | 0.869 | 0.914 | 0.929 | 0.739 |

- The train and val losses fall together, and the val loss levels off without
  rising, so there's no sign of overfitting.
- The visible step at epoch 50 is Ultralytics turning off mosaic augmentation
  for the last 10 epochs (`close_mosaic`).
- In the last 10 epochs mAP50 gained only about 0.01. More epochs on the same
  data would add little.

## Confusion matrix (bootstrap_val, conf 0.25)

`reports/bootstrap_v1/confusion_matrix.png` (counts) and
`confusion_matrix_normalized.png`. Columns are the true class, rows the
predicted class.

- **vape:** 315 of 327 correct (96%).
- **loose_cell:** 1,746 of 1,895 correct (92%); 57 missed (3%).
- **battery_pack:** only 203 of 371 are predicted as battery_pack; 157 (42%)
  come out as loose_cell.
  - This is the known label noise: in battery_and_vape, phone batteries were
    labelled "battery", which we mapped to loose_cell.
  - For the alert it doesn't matter, because every class raises it. The class
    shown is unreliable for packs.
- **Background (false boxes):** 1,349 vape, 867 battery_pack and 2,836
  loose_cell boxes on things that aren't batteries, at the low 0.25
  confidence threshold.
  - The demo alerts only at confidence ≥ 0.5 over 5 tracked frames, so far
    fewer of these reach an alert.
  - The pattern still matches the webcam tests: the model readily calls small
    objects loose_cell or vape.

## What the numbers don't show: webcam tests

- **Epoch 4 (`reports/finding_webcam_face_false_positives.md`):** 16 alerts,
  all loose_cell: 7 on the face or head, 7 on small hand-held objects.
- **Near-final weights (user test, 2026-10-18, not recorded):**
  - A Tic Tac box was detected as a vape.
  - A wardrobe in the background was detected.
  - The user saw no clear improvement over epoch 4.
- **Final weights, recorded test (2026-10-18, 55 s, screen recording kept
  local because it shows the user):** 4 alerts, all false:

  | time | object | alerted as |
  |---|---|---|
  | 16.7 s | Tic Tac | vape |
  | 25.3 s | smartphone (lock screen) | vape |
  | 31.5 s | white wardrobe in the background | battery_pack |
  | 51.1 s | Tic Tac held in front of the face | vape |

  - The wardrobe was also boxed below the alert threshold at 0 s, 29 s and 36 s.
  - **Faces alone: 0 alerts,** compared with 7 at epoch 4. That's an improvement.
  - No real battery was shown, so recall wasn't tested.
- **Reason:** the training data has almost no "this is not a battery"
  examples from a real room, so an internet-photo mAP of 0.93 doesn't carry
  over. The next step, bootstrap_v2, adds the 308 own hard-negative frames
  (train only).

## Demo command (final weights)

```bash
cd ~/akkunet && source ~/.venvs/akkunet/bin/activate
python -m akkunet.app --weights runs/bootstrap/bootstrap_v1/weights/best.pt --source 0 \
  --record <scratchpad>/webcam_v1_final.mp4     # recording shows your face: keep it local, never commit
```

Add `--device cpu` while another training run uses the GPU.

## 5-minute webcam test plan

Hold each item still for about 10 seconds, about 50 cm from the camera, then
move it slowly. Use the same room and lighting each time, so v1 and v2 can be
compared.

| # | item | expected | write down |
|---:|---|---|---|
| 1 | AA/AAA cells (2–3) | loose_cell alert | alert yes/no, class, confidence |
| 2 | phone battery (or a phone with its back off) | alert (battery_pack or loose_cell) | same |
| 3 | vape | vape alert | same |
| 4 | lighter | **no alert** | alerts, class |
| 5 | marker | **no alert** | alerts, class |
| 6 | lipstick | **no alert** | alerts, class |
| 7 | USB stick | **no alert** | alerts, class |
| 8 | glue stick | **no alert** | alerts, class |
| 9 | Tic Tac box | **no alert** (v1: vape) | alerts, class |
| 10 | empty hand, then face close up | **no alert** | alerts, class |
| 11 | slow pan over the room, including the wardrobe | **no alert** (v1: wardrobe detected) | alerts, class |

**Score:**
- batteries found: items 1–3, out of 3;
- items with false alerts: items 4–11, out of 8;
- total false alerts.

Run the same plan for v1 and then v2.

**BOOTSTRAP ONLY — internet photos, not tipping-floor performance.**
