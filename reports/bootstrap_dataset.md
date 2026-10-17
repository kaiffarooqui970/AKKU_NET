# Bootstrap dataset

**BOOTSTRAP ONLY — public data, not tipping-floor data.**

## Images removed per rule

| dataset | images in export | rule | images removed |
|---|---:|---|---:|
| battery_and_vape | 9432 | vape_cloud_download_mp4 | 182 |
| battery_and_vape | 9432 | vape_cloud_image_n | 1976 |
| battery_and_vape | 9432 | vape_mixed_unreviewed | 813 |
| battery_and_vape | | **kept** | **6461** |
| kashif_battery | 7337 | only_out_of_scope_classes | 3389 |
| kashif_battery | 7337 | xray_scan | 548 |
| kashif_battery | | **kept** | **3400** |

Full list: `reports/external/bootstrap_dropped.csv` (dataset, original split, file, rule).

## Split by source image (near-copy groups)

Near-copies = same Roboflow source name, or dHash ≤ 10 bits to the image, its mirror or a 90° rotation (greyscale copies match too). Target: 15% of every class in val.

| split | groups | images |
|---|---:|---:|
| train | 3824 | 8385 |
| val | 675 | 1476 |

Groups in both train and val: **0** (checked from the written manifest). Total 4499 groups; 2357 single images; largest group 71 images.

## Final counts

| class | train images | train boxes | val images | val boxes | val share (boxes) |
|---|---:|---:|---:|---:|---:|
| vape | 1666 | 2023 | 293 | 327 | 14% |
| power_bank (no data: untrained) | 0 | 0 | 0 | 0 | – |
| battery_pack | 1970 | 2205 | 347 | 371 | 14% |
| tool_battery (no data: untrained) | 0 | 0 | 0 | 0 | – |
| loose_cell | 4724 | 11889 | 833 | 1895 | 14% |
| (background) | 29 | 0 | 4 | 0 | – |

Per source:

| dataset | train | val |
|---|---:|---:|
| battery_and_vape | 5493 | 968 |
| kashif_battery | 2892 | 508 |

## Known label quality issues (from the class maps)

- **battery_and_vape:** Known label noise: battery -> loose_cell, but ~15-20% of sampled battery boxes (3-4 of 20) are flat smartphone battery packs (should be battery_pack). Accepted for the bootstrap model; expect some battery_pack/loose_cell confusion.
- **battery_and_vape:** About half of the original vape boxes marked the vapour cloud, not the device; only the device filename groups are kept (reports/finding_vape_cloud_labels.md).
- **kashif_battery:** 548 X-ray baggage scans (all Laptop Battery) dropped.

Licences and sources: `SOURCES.md`.
