# Dataset inspection: dfire

Folder: `/Users/kaif/akkunet/data/external/dfire`

Classes (ID order, verified on the contact sheets): `smoke`, `fire`

## Files vs. official split lists

- train: 17221 images, 17221 listed, missing 0, not listed 0, without label file 0
- test: 4306 images, 4306 listed, missing 0, not listed 0, without label file 0

## Split counts

| | train | test |
|---|---:|---:|
| images | 17221 | 4306 |
| empty (no fire, no smoke) | 7833 | 2005 |
| fire only | 944 | 220 |
| smoke only | 4681 | 1186 |
| fire + smoke | 3763 | 895 |
| from AoF | 6723 | 1661 |
| from WEB | 9443 | 2364 |
| from PublicDataset | 1055 | 281 |
| smoke boxes | 9550 | 2315 |
| smoke boxes < 1% of image | 1896 | 499 |
| smoke boxes < 0.1% of image | 185 | 60 |
| fire boxes | 11814 | 2878 |
| fire boxes < 1% of image | 7601 | 1860 |
| fire boxes < 0.1% of image | 1878 | 490 |

Image sizes (every 50th image): 1280x720 (181), 640x360 (81), 480x360 (18), 640x352 (18), 540x360 (5), 1920x1080 (4)

## Near-duplicate leakage

A test image counts as leaked if it is within 10 dHash bits (of 256) of a train image, its mirror or a 90° rotation (same rule as the bootstrap split).

| | leaked | of | share | exact (0 bits) | ≤ 5 bits |
|---|---:|---:|---:|---:|---:|
| test vs train | 1508 | 4306 | 35.0% | 137 | 1041 |
| … test AoF | 1294 | 1661 | 77.9% | | |
| … test WEB | 170 | 2364 | 7.2% | | |
| … test PublicDataset | 44 | 281 | 15.7% | | |
| 5-fold: valid1 vs train1 | 1196 | 3445 | 34.7% | 84 | 840 |

Leak pairs (test | nearest train, 2 pairs per row, closest first): `reports/external/dfire__leak_pairs.jpg`. Full list: `reports/external/dfire_leaks.csv`.
