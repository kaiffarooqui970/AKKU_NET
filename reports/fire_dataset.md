# Fire dataset: fire_v1 (D-Fire re-split by near-copy group)

The authors' test split is ignored: 35% of it are near-copies of train (reports/dfire_dataset.md). All 21,527 images are pooled and near-copies (dHash <= 10 of 256 bits, incl. mirror and 90° rotations; linked transitively) form groups. Whole groups go to train or val, aiming at 15% of every class. No images copied: lists point at `data/external/dfire/`.

| split | groups | images | smoke images | fire images | empty | smoke boxes | fire boxes |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 12336 | 18301 | 8947 | 4949 | 8363 | 10083 | 12275 |
| val | 2382 | 3226 | 1578 | 873 | 1475 | 1782 | 2417 |

- Groups in both splits: **0**.
- Val images within 10 bits of any train image (checked independently): **0**.
- Val share of images: 15.0%.
- Group sizes: 13538 single images; largest groups [566, 263, 245, 174, 130] images (fixed surveillance cameras: many frames of one view chain into one group).
- Where the val images came from (authors' split): train 2588, test 638

Known limitations (carry into every fire_v1 report):

- Night scenes: the same bright glow is labelled `fire` in some images and `smoke` in others (e.g. AoF04543 on both contact sheets). Per-class night results are unreliable; the combined `fire_or_smoke` metric is the one that matters for the escalation.
- Mostly outdoor footage; val is the same kind of data as train. Not tipping-floor performance.
