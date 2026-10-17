"""Tests for the fire_v1 dataset build (synthetic D-Fire-style folder, no real images)."""
import cv2
import numpy as np
import pytest
import yaml

from akkunet.fire import build_fire_split, cached_hash_variants, collect_fire, write_fire_lists

CLASSES = ["smoke", "fire"]


@pytest.fixture
def dfire(tmp_path):
    """D-Fire layout: {train,test}/{images,labels}, data.yaml written by us. The test image
    't_copy' is a mirrored copy of train image 'a' (the authors' split leaks like that)."""
    root = tmp_path / "dfire"
    (root / "data.yaml").parent.mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": CLASSES}))
    rng = np.random.default_rng(0)
    imgs = {name: cv2.resize(rng.integers(0, 256, (12, 12, 3), dtype=np.uint8), (96, 96),
                             interpolation=cv2.INTER_NEAREST) for name in "abcdefgh"}
    files = {"train": {n: imgs[n] for n in "abcdef"},
             "test": {"t_copy": cv2.flip(imgs["a"], 1), "g": imgs["g"], "h": imgs["h"]}}
    labels = {"a": "1 0.5 0.5 0.2 0.2", "t_copy": "1 0.5 0.5 0.2 0.2", "b": "0 0.5 0.5 0.6 0.6", "g": "0 0.4 0.4 0.3 0.3"}
    for split, items in files.items():
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for stem, img in items.items():
            cv2.imwrite(str(root / split / "images" / f"{stem}.png"), img)
            (root / split / "labels" / f"{stem}.txt").write_text(labels.get(stem, ""))
    return root


def test_fire_split_keeps_copies_together_and_copies_no_images(dfire, tmp_path):
    images = [c.path for c in collect_fire(dfire, CLASSES)]
    hashes = cached_hash_variants(images, tmp_path / "h.npz", progress=lambda *a: None)
    assert np.array_equal(cached_hash_variants(images, tmp_path / "h.npz"), hashes)    # cache reused

    for seed in range(5):
        cands, groups, assignment = build_fire_split(dfire, CLASSES, hashes, 10, 0.5, seed)
        group_of = {c.path.stem: g for c, g in zip(cands, groups)}
        assert group_of["a"] == group_of["t_copy"]                     # mirrored copy across the old split
        out = tmp_path / f"fire{seed}"
        data_yaml, rows = write_fire_lists(cands, groups, assignment, CLASSES, out, "fire_v1")
        listed = (out / "lists" / "fire_v1_train.txt").read_text().split() + \
                 (out / "lists" / "fire_v1_val.txt").read_text().split()
        assert sorted(listed) == sorted(str(p.resolve()) for p in images)   # originals, each once
        assert not list(out.rglob("*.png"))                                  # nothing copied
        split = {r["image"].rsplit("/", 1)[1]: r["split"] for r in rows}
        assert split["a.png"] == split["t_copy.png"]
        assert yaml.safe_load(data_yaml.read_text())["names"] == {0: "smoke", 1: "fire"}


def test_class_order_must_match_config(dfire):
    with pytest.raises(ValueError, match="classes"):
        collect_fire(dfire, ["fire", "smoke"])
