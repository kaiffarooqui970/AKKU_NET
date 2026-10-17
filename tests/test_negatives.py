"""Tests for hard negatives: empty labels, train only (synthetic videos only)."""
import csv

import pytest

from akkunet.evaluate import write_run_dataset
from akkunet.negatives import NEGATIVES_SOURCE, extract_negatives, negative_train_images
from akkunet.synthetic import make_synthetic_video

FE = {"every_n_frames": 1, "jpeg_quality": 95, "blur_threshold": 100.0, "blur_measure_long_side": 640,
      "hash_size": 16, "hamming_threshold": 10, "video_extensions": [".mp4"]}
CLASSES = ["vape", "power_bank", "battery_pack", "tool_battery", "loose_cell"]


@pytest.fixture
def negatives(tmp_path):
    make_synthetic_video(tmp_path / "raw" / "negatives_2026-10-17_desk" / "rec.mp4", n_scenes=3,
                         frames_per_scene=2, n_blurry_scenes=1, seed=4)
    stats, rows = extract_negatives(tmp_path / "raw", tmp_path / "neg", FE)
    return tmp_path, rows


def test_frames_get_empty_labels_and_are_train_only(negatives):
    tmp_path, rows = negatives
    assert len(rows) == 3                                          # blur + duplicates removed
    assert {r["split"] for r in rows} == {"train"} and {r["source"] for r in rows} == {NEGATIVES_SOURCE}
    labels = list((tmp_path / "neg" / "labels" / "train").glob("*.txt"))
    assert len(labels) == 3 and all(p.read_text() == "" for p in labels)
    assert not (tmp_path / "neg" / "images" / "val").exists()
    assert len(negative_train_images(tmp_path / "neg")) == 3


def test_rebuild_replaces_old_frames(negatives):
    tmp_path, _ = negatives
    _, rows = extract_negatives(tmp_path / "raw", tmp_path / "neg", FE)
    assert len(list((tmp_path / "neg" / "images" / "train").glob("*.jpg"))) == len(rows) == 3


def test_badly_named_session_folder_is_refused(tmp_path):
    (tmp_path / "raw" / "my_faces").mkdir(parents=True)
    with pytest.raises(ValueError, match="negatives_"):
        extract_negatives(tmp_path / "raw", tmp_path / "neg", FE)


def test_negatives_with_a_label_or_outside_train_are_refused(negatives):
    tmp_path, rows = negatives
    label = next((tmp_path / "neg" / "labels" / "train").glob("*.txt"))
    label.write_text("4 0.5 0.5 0.1 0.1\n")
    with pytest.raises(ValueError, match="non-empty label"):
        negative_train_images(tmp_path / "neg")
    label.write_text("")
    manifest = tmp_path / "neg" / "manifest.csv"
    text = manifest.read_text().replace("images/train", "images/val", 1).replace(",train,", ",val,", 1)
    manifest.write_text(text)
    with pytest.raises(ValueError, match="outside train"):
        negative_train_images(tmp_path / "neg")


def test_negatives_only_extend_the_train_list(negatives):
    tmp_path, _ = negatives
    boot = tmp_path / "boot"
    boot.mkdir()
    with open(boot / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "split", "source"])
        w.writeheader()
        w.writerows([{"image": "images/train/a.jpg", "split": "train", "source": "d"},
                     {"image": "images/val/b.jpg", "split": "val", "source": "d"}])
    neg = negative_train_images(tmp_path / "neg")
    write_run_dataset(boot, "v2", CLASSES, ["d"], ["d"], extra_train_images=neg)
    train = (boot / "lists" / "v2_train.txt").read_text().split()
    val = (boot / "lists" / "v2_val.txt").read_text().split()
    assert len(train) == 1 + len(neg) and val == [str(boot / "images/val/b.jpg")]
