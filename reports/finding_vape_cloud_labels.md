# Finding: "vape" labels that mark the vapour cloud, not the device

Documented label-quality and shortcut finding, Part B2, 2026-10-17.

## What we found

In the public Roboflow dataset **Battery and Vape** (workspace "david", CC BY 4.0
as declared by the uploader), **about half of all "vape" boxes are drawn around
the exhaled vapour cloud, not around the e-cigarette.** In a random sample of 40
vape boxes, about 20 contained only smoke.

A model trained on these labels would learn **"white cloud = vape"**. It would
fire on steam, smoke, fog or a white plastic bag, and it would miss a vape lying
in waste, which is the only case that matters on a tipping floor.

The two failure modes are different:
- **Wrong target:** the box marks a cloud instead of the object.
- **Shortcut risk:** even device boxes often sit next to a face or hand (see the
  person check below). A model can learn "near a mouth" instead of "vape".

## How it was found

1. Face-shortcut check: the COCO person detector run over all 4,930 vape images
   (`reports/external/battery_and_vape_person_check.md`). The contact sheet
   (`reports/external/battery_and_vape_person_check.jpg`, green = vape label, blue
   = detected person) showed green boxes on smoke clouds.
2. 40 random vape boxes (`reports/external/finding_vape_40_random_boxes.jpg`):
   about 20 of them were clouds.
3. Split by filename pattern, 10 random boxes per group
   (`reports/external/finding_vape_boxes_by_filename_group.jpg`, one row per group):

| filename group | vape images | content of sampled boxes |
|---|---:|---|
| `Image_<n>_jpg…` | 1,976 | clouds 10/10 |
| `Download-<n>-_mp4…` | 182 | clouds 10/10 (frames from social-media videos) |
| `<n>_jpg…` | 1,639 | devices 10/10 |
| `IMG_<n>…` | 320 | devices 10/10 |
| other names | 813 | mixed, ~3/10 clouds |

So the dataset appears to merge at least two sources with different labelling
conventions: one labels "vaping" (the cloud) and one labels "vape" (the device).
The class name doesn't tell you which one you get.

## What we did

- Dropped every vape image outside `<n>_jpg` and `IMG_<n>`: 2,158 cloud images
  plus 813 mixed images (kept out until reviewed). 1,959 vape images remain.
- Every dropped file is listed in `reports/external/bootstrap_dropped.csv`
  (rules `vape_cloud_image_n`, `vape_cloud_download_mp4`, `vape_mixed_unreviewed`).
- Face/hand images were **not** removed. A "face or hand without a vape" set is
  planned as a false-positive test (`docs/robustness_checks.md`).

## Caveats

- 10 boxes per group is a small sample. The filename rule is a heuristic, not a
  per-box review. Some device boxes may have been dropped, and some cloud boxes
  may remain.
- The COCO "person" class also fires on a hand alone, so "contains a person"
  overstates "next to a face".

## Why it matters beyond this project

The page says "CC BY 4.0, 9,432 images, classes battery and vape", which looks
ready to use. Only looking at the boxes shows that half the vape labels teach a
different concept. Public detection datasets need a box-level check before use,
and a model's validation score on the same data would not reveal the problem,
because the validation split carries the same mislabelling.
