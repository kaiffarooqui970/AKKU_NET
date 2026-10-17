# Robustness checks (planned)

Tests to run on the trained models in the robustness stage. Each check gets its
own small set of images that is **never** used for training.

| check | what it tests | why |
|---|---|---|
| Face/hand without a vape | False positives on faces, mouths, hands holding pens, lipstick, phones, USB sticks | Public vape images often show the device next to a face or in a hand, so the model may have learned "object near mouth/hand" (see `reports/finding_vape_cloud_labels.md`) |
| Steam, smoke, white bags | False positives on white cloud-like regions | Half the public "vape" labels marked vapour clouds; those images were removed, but the risk is worth measuring |
| Background / lighting shift | Recall drop on `shift_test` sessions | Model must not depend on one tarp or lamp |
