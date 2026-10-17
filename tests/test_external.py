"""Tests for Part B2 public-dataset import (synthetic Roboflow-style exports only)."""
import cv2
import numpy as np
import pytest
import yaml

from akkunet.evaluate import count_boxes, write_run_dataset
from akkunet.external import (BOOTSTRAP_MANIFEST_FIELDS, ONLY_OUT_OF_SCOPE, Candidate, augmentation_base_name,
                              check_class_map, collect_dataset, group_near_copies, hash_variants, inspect_export,
                              read_boxes, split_groups, write_bootstrap, write_csv)

CLASSES = ["vape", "power_bank", "battery_pack", "tool_battery", "loose_cell"]
SOURCE_NAMES = ["Car Battery", "Dry Cell", "Laptop Battery"]
MAPPING = {"Car Battery": None, "Dry Cell": "loose_cell", "Laptop Battery": "battery_pack"}


def make_export(root, images):
    """images: {rf_split: {stem: [label lines]}} -> a fake Roboflow YOLOv8 export."""
    (root / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": SOURCE_NAMES, "nc": len(SOURCE_NAMES)}))
    (root / "README.roboflow.txt").write_text("Intro\nThe following pre-processing was applied:\n* Resize to 640x640 (Stretch)\n")
    rng = np.random.default_rng(0)
    for split, files in images.items():
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        for stem, lines in files.items():
            img = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
            cv2.imwrite(str(root / split / "images" / f"{stem}.jpg"), cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST))
            (root / split / "labels" / f"{stem}.txt").write_text("\n".join(lines))
    return root


@pytest.fixture
def export(tmp_path):
    return make_export(tmp_path / "kashif", {
        "train": {
            "a_jpg.rf.aaa": ["1 0.5 0.5 0.2 0.2"],                        # dry cell
            "a_jpg.rf.bbb": ["1 0.5 0.5 0.2 0.2"],                        # augmented copy of a
            "car_jpg.rf.ccc": ["0 0.5 0.5 0.8 0.8"],                      # only a dropped class
            "mixed_jpg.rf.ddd": ["0 0.3 0.3 0.2 0.2", "2 0.7 0.7 0.2 0.2"],
            "empty_jpg.rf.eee": [],                                       # genuine background
        },
        "valid": {"v_jpg.rf.fff": ["2 0.5 0.5 0.4 0.3"]},
    })


def test_polygon_labels_become_boxes(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("1 0.2 0.2 0.6 0.2 0.6 0.8 0.2 0.8\n")
    boxes, polygons = read_boxes(p)
    assert polygons == 1
    assert boxes[0][0] == 1 and np.allclose(boxes[0][1:], (0.4, 0.5, 0.4, 0.6))


def test_augmentation_base_name():
    from pathlib import Path
    assert augmentation_base_name(Path("IMG_1_jpg.rf.0a1b2c.jpg")) == "IMG_1_jpg"


def test_class_map_must_cover_every_class_and_reject_pending():
    assert check_class_map(MAPPING, SOURCE_NAMES, CLASSES) == []
    assert any("not in the map" in p for p in check_class_map({"Dry Cell": "loose_cell"}, SOURCE_NAMES, CLASSES))
    assert any("PENDING" in p for p in check_class_map({**MAPPING, "Dry Cell": "PENDING"}, SOURCE_NAMES, CLASSES))
    assert any("unknown class" in p for p in check_class_map({**MAPPING, "Dry Cell": "cell"}, SOURCE_NAMES, CLASSES))
    assert any("not in this export" in p for p in check_class_map({**MAPPING, "Bike": None}, SOURCE_NAMES, CLASSES))


def test_inspect_counts(export):
    names, stats, leaks = inspect_export(export)
    assert names == SOURCE_NAMES
    assert stats["train"].images == 5 and stats["train"].empty == 1
    assert stats["train"].augmented_groups == 1 and stats["train"].augmented_images == 2
    assert stats["train"].boxes["Car Battery"] == 2
    assert stats["train"].sizes == {"64x64": 5}
    assert "valid" in leaks


def collect(export, mapping=MAPPING, rules=None):
    return collect_dataset("kashif", export, {"mapping": mapping, "drop_images": rules or []}, CLASSES)


def test_collect_remaps_and_drops_images_with_only_out_of_scope_classes(export):
    kept, dropped = collect(export)
    names = {c.path.stem: c for c in kept}
    assert [d["original_file"] for d in dropped] == ["car_jpg.rf.ccc.jpg"]
    assert dropped[0]["rule"] == ONLY_OUT_OF_SCOPE
    assert names["mixed_jpg.rf.ddd"].boxes[0][0] == 2 and names["mixed_jpg.rf.ddd"].boxes_dropped == 1
    assert names["empty_jpg.rf.eee"].boxes == []          # genuine background kept
    assert names["v_jpg.rf.fff"].original_split == "valid"


def test_drop_rules_need_all_conditions(export):
    rules = [{"name": "no_dry_cell_a", "has_class": "Dry Cell", "filename": "^a_"},
             {"name": "laptop_not_v", "has_class": "Laptop Battery", "filename_not": "^v_"}]
    _, dropped = collect(export, rules=rules)
    by_rule = {d["original_file"]: d["rule"] for d in dropped}
    assert by_rule["a_jpg.rf.aaa.jpg"] == by_rule["a_jpg.rf.bbb.jpg"] == "no_dry_cell_a"
    assert by_rule["mixed_jpg.rf.ddd.jpg"] == "laptop_not_v"
    assert "v_jpg.rf.fff.jpg" not in by_rule


def test_drop_rule_with_unknown_class_is_refused(export):
    with pytest.raises(ValueError, match="unknown class"):
        collect(export, rules=[{"name": "x", "has_class": "Nope"}])
    with pytest.raises(ValueError, match="PENDING"):
        collect(export, mapping={**MAPPING, "Dry Cell": "PENDING"})


def test_mirrored_rotated_and_grey_copies_are_grouped(tmp_path):
    def img(seed):
        b = np.random.default_rng(seed).integers(0, 256, (12, 12, 3), dtype=np.uint8)
        return cv2.resize(b, (96, 96), interpolation=cv2.INTER_NEAREST)
    a, other = img(1), img(2)
    grey = cv2.cvtColor(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    images = [a, cv2.flip(a, 1), cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE), grey, other]
    cands = [Candidate("d", tmp_path / f"f{i}.jpg", "train", [], 0) for i in range(len(images))]
    groups = group_near_copies(cands, np.stack([hash_variants(im) for im in images]))
    assert len(set(groups[:4])) == 1 and groups[4] != groups[0]


def test_same_source_name_is_grouped_even_if_images_differ(export):
    kept, _ = collect(export)
    hashes = np.stack([hash_variants(cv2.imread(str(c.path))) for c in kept])
    groups = group_near_copies(kept, hashes, threshold=0)
    g = {c.path.stem: grp for c, grp in zip(kept, groups)}
    assert g["a_jpg.rf.aaa"] == g["a_jpg.rf.bbb"]
    assert g["a_jpg.rf.aaa"] != g["mixed_jpg.rf.ddd"]


def test_split_keeps_groups_whole_and_near_target(tmp_path):
    # 100 groups of 1-3 images, each image one loose_cell box.
    cands, groups = [], []
    for g in range(100):
        for _ in range(1 + g % 3):
            cands.append(Candidate("d", tmp_path / f"{len(cands)}.jpg", "train", [(4, .5, .5, .1, .1)], 0))
            groups.append(g)
    assignment = split_groups(cands, groups, CLASSES, val_fraction=0.15, seed=0)
    val_images = sum(assignment[g] == "val" for g in groups)
    assert 0.10 * len(cands) <= val_images <= 0.15 * len(cands)
    assert set(assignment.values()) == {"train", "val"}


def test_write_bootstrap_and_run_lists(export, tmp_path):
    kept, _ = collect(export)
    groups = list(range(len(kept)))
    assignment = {g: ("val" if kept[g].original_split == "valid" else "train") for g in groups}
    out = tmp_path / "bootstrap"
    (out / "images" / "train").mkdir(parents=True)
    (out / "images" / "train" / "stale.jpg").write_bytes(b"x")       # rebuilt from scratch
    rows = write_bootstrap(kept, groups, assignment, CLASSES, out)
    write_csv(out / "manifest.csv", rows, BOOTSTRAP_MANIFEST_FIELDS)

    assert not (out / "images" / "train" / "stale.jpg").exists()
    assert (out / "labels" / "val" / "kashif__v_jpg.rf.fff.txt").read_text().startswith("2 ")
    assert (out / "labels" / "train" / "kashif__empty_jpg.rf.eee.txt").read_text() == ""
    data_yaml = write_run_dataset(out, "run", CLASSES, ["kashif"], ["kashif"])
    assert yaml.safe_load(data_yaml.read_text())["names"] == dict(enumerate(CLASSES))
    assert count_boxes(out / "lists" / "run_val.txt", CLASSES)["battery_pack"] == 1
    with pytest.raises(ValueError, match="Not imported"):
        write_run_dataset(out, "x", CLASSES, ["nope"], ["kashif"])
