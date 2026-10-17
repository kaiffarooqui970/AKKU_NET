"""Tests for Part B2 public-dataset import (synthetic Roboflow-style exports only)."""
import csv

import cv2
import numpy as np
import pytest
import yaml

from akkunet.evaluate import count_boxes, write_run_dataset
from akkunet.external import (augmentation_base_name, check_class_map, import_export, inspect_export,
                              read_boxes, remove_source, update_manifest)

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


def test_import_remaps_and_skips_images_with_only_dropped_classes(export, tmp_path):
    out = tmp_path / "bootstrap"
    stats, rows = import_export("kashif", export, {"mapping": MAPPING}, CLASSES, out)

    assert stats.only_dropped["train"] == 1          # car-only image skipped
    assert stats.background["train"] == 1            # empty image kept
    assert stats.mixed["train"] == 1
    assert stats.imported == {"train": 4, "val": 1}
    assert not (out / "images" / "train" / "kashif__car_jpg.rf.ccc.jpg").exists()
    assert (out / "labels" / "train" / "kashif__mixed_jpg.rf.ddd.txt").read_text().split()[0] == "2"
    assert (out / "labels" / "val" / "kashif__v_jpg.rf.fff.txt").read_text().startswith("2 ")
    assert (out / "labels" / "train" / "kashif__empty_jpg.rf.eee.txt").read_text() == ""
    assert stats.boxes["train"]["loose_cell"] == 2 and stats.boxes["train"]["battery_pack"] == 1
    assert {r["source"] for r in rows} == {"kashif"}


def test_import_refuses_pending_map(export, tmp_path):
    with pytest.raises(ValueError, match="PENDING"):
        import_export("kashif", export, {"mapping": {**MAPPING, "Dry Cell": "PENDING"}}, CLASSES, tmp_path / "b")


def test_train_cap_and_reimport_replace(export, tmp_path):
    out = tmp_path / "bootstrap"
    stats, rows = import_export("kashif", export, {"mapping": MAPPING}, CLASSES, out, max_train_images=2)
    assert stats.images_in["train"] == 2
    update_manifest(out, "kashif", rows)
    update_manifest(out, "other", [{**rows[0], "source": "other"}])
    assert remove_source(out, "kashif") == len(rows)
    update_manifest(out, "kashif", [])
    with open(out / "manifest.csv") as f:
        assert {r["source"] for r in csv.DictReader(f)} == {"other"}


def test_run_dataset_keeps_val_fixed_when_extra_train_data_is_added(export, tmp_path):
    out = tmp_path / "bootstrap"
    _, rows = import_export("kashif", export, {"mapping": MAPPING}, CLASSES, out)
    update_manifest(out, "kashif", rows)
    extra = make_export(tmp_path / "gvl", {"train": {"w_jpg.rf.111": ["1 0.5 0.5 0.1 0.1"]},
                                           "valid": {"wv_jpg.rf.222": ["1 0.5 0.5 0.1 0.1"]}})
    _, rows = import_export("gvl", extra, {"mapping": MAPPING}, CLASSES, out)
    update_manifest(out, "gvl", rows)

    base = write_run_dataset(out, "base", CLASSES, ["kashif"], ["kashif"])
    plus = write_run_dataset(out, "plus", CLASSES, ["kashif", "gvl"], ["kashif"])
    lists = out / "lists"
    assert (lists / "base_val.txt").read_text() == (lists / "plus_val.txt").read_text()
    assert len((lists / "plus_train.txt").read_text().split()) == len((lists / "base_train.txt").read_text().split()) + 1
    assert yaml.safe_load(base.read_text())["names"] == dict(enumerate(CLASSES))
    assert count_boxes(lists / "base_val.txt", CLASSES)["battery_pack"] == 1
    with pytest.raises(ValueError, match="Not imported"):
        write_run_dataset(out, "x", CLASSES, ["nope"], ["kashif"])
