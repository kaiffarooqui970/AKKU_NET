# Robustness checks (planned)

Tests to run on the trained models in the robustness stage. Each check gets its
own small set of images that is **never** used for training.

| check | what it tests | why |
|---|---|---|
| Face/hand without a vape | False positives on faces, mouths, hands holding pens, lipstick, phones, USB sticks | Public vape images often show the device next to a face or in a hand, so the model may have learned "object near mouth/hand" (see `reports/finding_vape_cloud_labels.md`) |
| Steam, smoke, white bags | False positives on white cloud-like regions | Half the public "vape" labels marked vapour clouds; those images were removed, but the risk is worth measuring |
| Lookalikes in hand | False positives on Tic Tac boxes, lighters, phones, markers, USB sticks, glue sticks | Epoch-4 webcam test: 7 of 16 alerts were on a face/head, 7 on small hand-held objects (`reports/finding_webcam_face_false_positives.md`) |
| Background / lighting shift | Recall drop on `shift_test` sessions | Model must not depend on one tarp or lamp |
| **Fire model: dust/haze is not smoke** (required) | False `fire_or_smoke` alerts on unloading dust, fog, steam, exhaust, sunlight through dusty air | D-Fire has many faint haze-like smoke boxes from distant cameras (`reports/dfire_dataset.md`); dust from unloading is the tipping floor's everyday "smoke lookalike" |
| **Fire model: indoor hall footage** (required) | Recall and false alarms on our own hall recordings (smouldering in waste, small flames, hall lighting, vehicles' lights) | D-Fire is mostly outdoor (wildfire, streets, surveillance cameras); its scores say nothing about a hall |

## Documented findings so far

- `reports/finding_vape_cloud_labels.md`: about half of the public "vape"
  labels marked the vapour cloud, not the device (label quality / shortcut).
- `reports/finding_webcam_face_false_positives.md`: faces and small hand-held
  objects detected as `loose_cell` by the epoch-4 model (shortcut).
- `reports/dfire_dataset.md`: D-Fire's official test split leaks (35%); night
  glows are labelled `fire` in some images and `smoke` in others (known
  limitation: report the combined `fire_or_smoke` metric).
