# Finding: faces and small hand-held objects detected as `loose_cell`

Documented shortcut finding, Part B3, 2026-10-17. This complements
`reports/finding_vape_cloud_labels.md`.

## Setup

- Model: bootstrap_v1 after **epoch 4 of 60** (frozen copy of `last.pt`).
  This was an early, unfinished model.
- Live webcam demo (`python -m akkunet.app`) on CPU, about 18 minutes,
  9,730 frames at ~9 fps, indoor room, one person in front of the camera.
- Alert rule: one alert per track after 5 frames with confidence ≥ 0.5.

## Result

16 alerts, all `loose_cell`. Reviewed frame by frame at each alert:

| what triggered the alert | alerts |
|---|---:|
| the person's **head/face** (mostly low in the frame or partly hidden) | 7 |
| a **Tic Tac box** (white pills in a clear case) | 3 |
| a small coloured hand-held object (lighter or similar) | 3 |
| a **phone** | 1 |
| unclear (box no longer visible in the reviewed frame) | 2 |

So at least 14 of 16 alerts were false. Their confidences (0.50–0.83) were
often above the alert threshold, so the 5-frame rule alone did not stop them.

## Interpretation

- **Face shortcut.** The public training images often show cells and vapes
  held close to a face or hand, and they contain no faces *without* a battery.
  The model can therefore treat "face/head" as evidence for a battery. This is
  the risk the B2 person check flagged for vapes; here it showed up as
  `loose_cell`.
- **Lookalikes.** Small hand-held objects of cell-like size and shape (pill
  boxes, lighters, phones) are not separated from cells. The training data has
  no hard negatives for them.
- **A validation score doesn't reveal this.** bootstrap_val contains the same
  kind of internet photos as train, so it has no faces-without-batteries and no
  lookalikes to fail on.

## Caveats

- Epoch 4 only. The finished model must be re-tested before conclusions are
  drawn. The recording is kept locally and not committed (it shows a person's face).
- One person, one room, one session. Counts are from a single review of the
  alert frames.

## Planned follow-up

Hard negatives: a recorded session of faces, hands and lookalikes with no
batteries in view, added as empty-label TRAIN-ONLY images, then fine-tuning
(bootstrap_v2) and a before/after comparison of alerts on the same
categories. See `docs/robustness_checks.md`.
