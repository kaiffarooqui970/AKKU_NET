"""Tests for the Part B3 demo logic (no model, no camera)."""
import numpy as np
import pytest

from akkunet.app import BANNER_TEXT, AlertTracker, Detection, draw, parse_source, resolve_classes


def det(track_id, conf=0.9, cls=0):
    return Detection((10, 10, 50, 50), cls, conf, track_id)


def test_alert_needs_min_frames_and_fires_once_per_track():
    t = AlertTracker(conf_threshold=0.5, min_frames=3)
    fired = [t.update([det(1)], i) for i in range(6)]
    assert [len(f) for f in fired] == [0, 0, 1, 0, 0, 0]
    assert fired[2][0].track_id == 1 and fired[2][0].frame_index == 2


def test_low_confidence_frames_do_not_count():
    t = AlertTracker(conf_threshold=0.5, min_frames=3)
    for i, conf in enumerate([0.9, 0.3, 0.3, 0.9]):
        assert t.update([det(1, conf)], i) == []
    assert len(t.update([det(1, 0.9)], 4)) == 1   # 3rd qualifying frame; gaps allowed


def test_untracked_boxes_and_other_classes_never_alert():
    t = AlertTracker(conf_threshold=0.5, min_frames=1, alert_classes={4})
    assert t.update([det(None), det(2, cls=0)], 0) == []
    assert [a.track_id for a in t.update([det(3, cls=4)], 1)] == [3]


def test_each_track_alerts_separately():
    t = AlertTracker(conf_threshold=0.5, min_frames=2)
    t.update([det(1), det(2)], 0)
    assert sorted(a.track_id for a in t.update([det(1), det(2)], 1)) == [1, 2]


def test_draw_shows_banner_only_when_active():
    frame = np.zeros((240, 320, 3), np.uint8)
    with_banner = draw(frame, [det(1)], {0: "vape"}, {1}, f"{BANNER_TEXT}: vape #1", "status")
    without = draw(frame, [det(1)], {0: "vape"}, set(), None, "status")
    assert with_banner.shape == frame.shape and frame.sum() == 0      # input untouched
    assert with_banner[5, 300, 2] > 200 and with_banner[5, 300, :2].max() < 50   # red banner strip
    assert without[5, 300].sum() == 0


def test_parse_source_and_resolve_classes():
    assert parse_source("0") == 0 and parse_source(1) == 1 and parse_source("clip.mp4") == "clip.mp4"
    names = {0: "vape", 4: "loose_cell"}
    assert resolve_classes(names, []) is None
    assert resolve_classes(names, ["loose_cell"]) == {4}
    with pytest.raises(ValueError, match="Unknown"):
        resolve_classes(names, ["battery"])
