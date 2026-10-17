# D-Fire inspection (fire/smoke model, separate from the battery model)

Inspection only. Nothing has been trained on D-Fire. Numbers come from
`python scripts/inspect_dfire.py` (`reports/external/dfire_inspection.md`);
sheets come from `python scripts/box_contact_sheet.py dfire fire|smoke --n 40`.

## Class IDs

There's no data.yaml in the download. The label counts match the D-Fire README
(14,692 / 11,865 boxes, 1,164 fire-only images), and the sheets confirm:
**0 = smoke, 1 = fire**. Written to `data/external/dfire/data.yaml`.

## Split counts (official train/test, unchanged)

| | train | test |
|---|---:|---:|
| images | 17,221 | 4,306 |
| empty (neither) | 7,833 | 2,005 |
| fire boxes | 11,814 | 2,878 |
| smoke boxes | 9,550 | 2,315 |
| fire boxes < 1% / < 0.1% of image | 7,601 / 1,878 | 1,860 / 490 |

- All image files match the authors' lists, and every image has a label file.
- Images are at their original sizes (mostly 1280x720 and 640x360), with no
  Roboflow resizing or augmented copies.
- About 2/3 of the fire boxes are small (< 1% of the image area). That's useful
  for small flames, but most of them are distant fires, not a flame close up.

## Leakage: the official test split leaks

1,508 of 4,306 test images (35.0%) are near-copies (≤ 10 dHash bits, including
mirror and 90° rotations) of a train image. Per source:

| source | leaked test images |
|---|---:|
| AoF | 1,294 / 1,661 (77.9%) |
| WEB | 170 / 2,364 (7.2%) |
| PublicDataset | 44 / 281 (15.7%) |

The 5-fold split leaks the same way (fold 1: 34.7%).

On `reports/external/dfire__leak_pairs.jpg` the pairs are real copies: the
same fixed surveillance camera and view at another time, or the same web video
frame. These aren't boundary effects: all AoF and WEB leak pairs are more than
500 file numbers apart. **D-Fire test scores would be optimistic, especially
for AoF.** If we ever train on it, we would re-split by near-copy group, as we
did for the bootstrap.

## Contact sheets: do the smoke boxes have the vape-cloud problem?

- `reports/external/dfire__smoke__boxes.jpg` (40 random smoke boxes, at most one per image)
- `reports/external/dfire__fire__boxes.jpg` (40 random fire boxes)

**Smoke: not the vape problem, but its own problems.**

The vape problem was a box on the wrong *object*: the cloud instead of the
device. D-Fire's smoke class *is* the cloud, so that part is correct. My count
of the 40 smoke boxes:

| what the box shows | count |
|---|---:|
| clearly visible smoke (plumes, black smoke over burning cars or buildings, wildfire smoke) | ~22 |
| faint, almost featureless grey haze from distant surveillance cameras (e.g. AoF08278, AoF05200, AoF06271, AoF08295) | ~9 |
| night scenes where the box covers a bright glow, the same kind of white or purple blob the fire class contains (e.g. AoF04543, AoF04407, AoF07825) | ~9 |

- **The night glows are the closest thing to vape-cloud-style label noise.**
  AoF04543 is on *both* sheets, so the same bright blob carries both labels.
  At night the model can't learn a clean fire/smoke difference from this data.
- **The faint haze boxes** are probably correct early smoke, but at crop scale
  they look like sky, fog or dust. On a tipping floor, dust from unloading is
  exactly the false positive to worry about.
- **There are clouds next to smoke** (PublicDataset00610, WEB06320): sky clouds
  as a lookalike are in the data, which is good for learning the difference.

**Fire:** about 22 of 40 boxes are clearly visible flames. About 13 are
overexposed white blobs from night cameras, about 4 are heavily pixelated tiny
boxes, and 1 is nearly black (WEB10030). No box is on a non-fire object.

## Fit for AkkuNet (tipping floor)

D-Fire is mostly **outdoor**: wildfire, distant hills, streets and burning
vehicles. Indoor hall scenes, close-up smouldering in waste, lighter flames and
dust clouds are rare or absent. Any fire/smoke model trained on it must be
tested on our own hall footage before anyone relies on it, and stays an
**information aid** in the human-in-the-loop escalation
(`docs/fire_escalation.md`), never a certified detector.

## Not done (as agreed)

No fire/smoke training while bootstrap_v1 is running. No re-split yet.
