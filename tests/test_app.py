"""Tests for the Part B3 demo logic (no model, no camera)."""
import numpy as np
import pytest

from akkunet.app import BANNER_TEXT, AlertTracker, AudioAlerter, Detection, draw, parse_source, resolve_classes


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


# ---------------------------------------------------------------- audio (no real sound)

class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class FakeProcess:
    def poll(self):
        return 0


def make_audio(mode="beep", which=lambda name: f"/usr/bin/{name}", **kw):
    clock, spawned = FakeClock(), []
    audio = AudioAlerter(mode=mode, cooldown=5.0, clock=clock, which=which,
                         spawn=lambda cmd: spawned.append(cmd) or FakeProcess(), **kw)
    return audio, clock, spawned


def test_cooldown_allows_one_sound_per_window():
    audio, clock, spawned = make_audio()
    assert audio.alert("loose_cell") is True
    assert audio.alert("vape") is False            # same moment: several tracks alert -> one sound
    clock.t += 4.9
    assert audio.alert("vape") is False            # still inside the cooldown
    clock.t += 0.1
    assert audio.alert("vape") is True             # exactly 5 s later
    assert len(spawned) == 2


def test_suppressed_alert_does_not_extend_cooldown():
    audio, clock, spawned = make_audio()
    audio.alert("x")
    clock.t += 3
    audio.alert("x")                               # suppressed
    clock.t += 2                                   # 5 s after the FIRST sound
    assert audio.alert("x") is True


def test_off_mode_never_plays():
    audio, _, spawned = make_audio(mode="off")
    assert audio.alert("vape") is False and spawned == []


def test_commands_for_beep_and_voice():
    beep, _, spawned = make_audio(beep_sound="/x/Sosumi.aiff")
    beep.alert("vape")
    assert spawned == [["afplay", "/x/Sosumi.aiff"]]

    voice, _, spawned = make_audio(mode="voice")
    voice.alert("vape")
    assert spawned == [["say", "-v", "Anna", "Achtung: möglicher Akku erkannt"]]

    named, _, spawned = make_audio(mode="voice", say_class_name=True, spoken_class_names={"vape": "E-Zigarette"})
    named.alert("vape")
    assert spawned[0][-1] == "Achtung: E-Zigarette erkannt"


def test_missing_program_falls_back_to_bell_and_warns_once(capsys):
    audio, clock, spawned = make_audio(which=lambda name: None)
    assert audio.alert("vape") is True
    clock.t += 10
    audio.alert("vape")
    out = capsys.readouterr()
    assert spawned == [] and out.out == "\a\a"
    assert out.err.count("WARNING") == 1


def test_from_config_and_cli_override():
    cfg = {"mode": "voice", "cooldown_seconds": 2, "voice": "Eddy"}
    assert AudioAlerter.from_config(cfg).voice == "Eddy"
    assert AudioAlerter.from_config(cfg, "off").mode == "off"
    assert AudioAlerter.from_config(None).mode == "beep"
    with pytest.raises(ValueError, match="audio mode"):
        AudioAlerter(mode="loud")
