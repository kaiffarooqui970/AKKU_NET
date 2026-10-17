# Labelling guide

One box per item, drawn tight around the **whole visible object** (not just its
label or logo). Partly hidden items are labelled if a person could still tell
what they are. Class IDs follow `config.yaml` and never change.

| ID | class | what counts |
|---:|---|---|
| 0 | vape | disposable and refillable e-cigarettes / vape pens / pods |
| 1 | power_bank | portable phone/laptop chargers with a built-in battery |
| 2 | battery_pack | removable laptop and smartphone batteries, other flat or boxed multi-cell packs |
| 3 | tool_battery | slide-on / clip-on batteries for cordless power tools and garden tools |
| 4 | loose_cell | single cells: cylindrical (AA, AAA, 14500, 18650, 21700, C, D), 9V block, coin/button cells |

## Deliberate decisions

### Alkaline cells count as `loose_cell` (decided 2026-10-17)

AA/AAA alkaline cells are **not lithium**, but they are labelled `loose_cell`
anyway:

- A camera cannot tell an alkaline AA from a lithium 14500 cell, or an alkaline
  AAA from a lithium AAA. Asking the model to separate them would teach it to
  read printed labels, which are often missing, dirty or turned away.
- Any cell in a compactor is a risk, and the alert means "possible battery,
  check it", not "confirmed lithium".

Lithium coin cells (CR2032 etc.) are lithium and also count as `loose_cell`.
Alkaline/silver-oxide button cells look the same and are labelled the same way.
9V blocks are labelled `loose_cell` too (same reasoning; one item, one box).

Consequence: `loose_cell` precision should be read as "found a cell", not
"found a lithium cell".

### Out of scope

Car, motorbike and other lead-acid starter batteries are **not labelled**. In
public datasets they are dropped (images that contained only those are left out,
not kept as empty background).

### Own photos for training only

Photos of our own power banks and tool batteries go into
`data/raw_photos/train_only_<YYYY-MM-DD>[_note]/` (e.g.
`train_only_2026-10-20_powerbanks`). They are **always put in train** by
`scripts/extract_frames.py`; listing such a folder under val/test/shift_test is
an error. Use objects and backgrounds for these photos that you will **not**
reuse when filming test sessions, so test clips stay unseen.
