"""Tests for the fire/smoke escalation (Telegram API mocked, no sound, no network)."""
import json
import re
from pathlib import Path

import pytest

from akkunet.config import load_config
from akkunet.escalation import (CONFIRMED, DISMISSED, ESCALATED, OPEN, EscalationManager, EventLog, Facility,
                                SiteAlarm, SmokeTrigger, build_escalation)
from akkunet.notify import CONFIRM, DISMISS, ConsoleNotifier, Contact, Notifier, Response, TelegramNotifier, load_env

REPO_ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


class FakeProcess:
    def __init__(self):
        self.running = True

    def poll(self):
        return None if self.running else 0

    def terminate(self):
        self.running = False


@pytest.fixture
def setup(tmp_path):
    clock, spawned = Clock(), []
    notifier = ConsoleNotifier(printer=lambda *a: None)
    alarm = SiteAlarm(spawn=lambda cmd: spawned.append(cmd) or FakeProcess())
    contacts = {"primary": Contact("primary", "Schichtleitung", "111"),
                "secondary": Contact("secondary", "Betriebsleitung", "222")}
    log = EventLog(":memory:")
    manager = EscalationManager(notifier, log, contacts, Facility("Demo-Anlage", "Musterstraße 1, Musterstadt"),
                                response_timeout=90, site_alarm=alarm, hard_negative_dir=tmp_path / "hn",
                                clock=clock)
    snap = tmp_path / "snap.jpg"
    snap.write_bytes(b"jpeg")
    return manager, notifier, clock, spawned, log, snap


def kinds(log, inc_id):
    return [e["kind"] for e in log.events(inc_id)]


def test_trigger_alerts_primary_only(setup):
    manager, notifier, clock, spawned, log, snap = setup
    inc = manager.trigger("Bunker 1", "Rauch", snap)
    assert [(k, a) for k, a, _ in notifier.sent] == [("alert", "111")]
    assert "Bunker 1" in notifier.sent[0][2]
    assert kinds(log, inc.id) == ["detected", "notified"] and spawned == []


def test_no_answer_escalates_to_secondary_and_sounds_alarm_once(setup):
    manager, notifier, clock, spawned, log, snap = setup
    inc = manager.trigger("Bunker 1", "Rauch", snap)
    clock.t += 89
    manager.tick()
    assert inc.status == OPEN and len(notifier.sent) == 1
    clock.t += 1
    manager.tick()
    manager.tick()
    assert inc.status == ESCALATED
    assert [a for k, a, _ in notifier.sent if k == "alert"] == ["111", "222"]
    assert len(spawned) == 1                                       # alarm started once
    assert log.incident(inc.id)["status"] == ESCALATED


def test_confirm_sends_call_script_and_alarm_never_calls(setup):
    manager, notifier, clock, spawned, log, snap = setup
    inc = manager.trigger("Bunker 1", "Rauch", snap)
    clock.t += 30
    notifier.inject(inc.id, CONFIRM, "111")
    manager.tick()
    kind, address, text = notifier.sent[-1]
    assert (kind, address) == ("text", "111")
    assert "112" in text and "tel:112" in text and "Musterstraße 1" in text and "Bunker 1" in text
    assert "does NOT call" in text
    assert len(spawned) == 1
    row = log.incident(inc.id)
    assert row["status"] == CONFIRMED and row["resolved_by"] == "Schichtleitung" and row["response_seconds"] == 30
    assert "call_script_sent" in kinds(log, inc.id)


def test_first_decision_wins_and_late_answers_are_logged(setup):
    manager, notifier, clock, spawned, log, snap = setup
    inc = manager.trigger("Z", "Rauch", snap)
    notifier.inject(inc.id, CONFIRM, "111")
    notifier.inject(inc.id, DISMISS, "222")
    manager.tick()
    assert inc.status == CONFIRMED
    assert kinds(log, inc.id)[-1] == "late_response"


def test_false_alarm_saves_hard_negative_and_informs_alerted_contacts(setup, tmp_path):
    manager, notifier, clock, spawned, log, snap = setup
    inc = manager.trigger("Z", "Rauch", snap)
    clock.t += 95
    manager.tick()                                               # escalated: both alerted, alarm on
    notifier.inject(inc.id, DISMISS, "222")
    manager.tick()
    assert inc.status == DISMISSED
    assert (tmp_path / "hn" / f"incident{inc.id:05d}_snap.jpg").read_bytes() == b"jpeg"
    assert not manager.alarm.active                              # alarm stopped
    assert {a for k, a, t in notifier.sent if k == "text" and "false alarm" in t} == {"111", "222"}
    assert log.incident(inc.id)["resolved_by"] == "Betriebsleitung"


def test_answers_from_unknown_people_or_incidents_are_ignored(setup):
    manager, notifier, clock, spawned, log, snap = setup
    inc = manager.trigger("Z", "Rauch", snap)
    notifier.inject(inc.id, CONFIRM, "999")                      # not a configured contact
    notifier.inject(12345, CONFIRM, "111")                       # no such incident
    manager.tick()
    assert inc.status == OPEN
    assert kinds(log, inc.id)[-1] == "ignored_response"


def test_failed_message_is_logged_and_escalation_continues(setup):
    manager, notifier, clock, spawned, log, snap = setup
    notifier.send_alert = lambda *a: (_ for _ in ()).throw(RuntimeError("network down"))
    inc = manager.trigger("Z", "Rauch", snap)
    assert "notify_failed" in kinds(log, inc.id)
    clock.t += 90
    manager.tick()
    assert inc.status == ESCALATED and len(spawned) == 1         # still escalates and sounds the alarm


def test_smoke_trigger_needs_consecutive_frames_and_fires_once():
    t = SmokeTrigger(min_frames=3)
    assert [t.update(s) for s in [True, True, False, True, True, True, True]] == [False] * 5 + [True, False]
    t.reset()
    assert [t.update(True) for _ in range(3)] == [False, False, True]


# ---------------------------------------------------------------- Telegram (mocked HTTP)

class FakeTelegram:
    def __init__(self):
        self.calls = []

    def __call__(self, url, data, files=None, timeout=30):
        method = url.rsplit("/", 1)[1]
        self.calls.append((method, data, files))
        return {"message_id": 77} if method.startswith("send") else True


def test_telegram_alert_has_confirm_and_false_alarm_buttons(tmp_path):
    http = FakeTelegram()
    tg = TelegramNotifier("TOKEN", http=http, start_polling=False)
    snap = tmp_path / "s.jpg"
    snap.write_bytes(b"jpeg")
    tg.send_alert(Contact("primary", "Schichtleitung", "111"), 5, "Rauch in Bunker 1", snap)
    method, data, files = http.calls[0]
    assert method == "sendPhoto" and data["chat_id"] == "111" and "photo" in files
    buttons = json.loads(data["reply_markup"])["inline_keyboard"][0]
    assert [b["callback_data"] for b in buttons] == ["confirm:5", "dismiss:5"]
    tg.remove_buttons(5)
    assert http.calls[-1][0] == "editMessageReplyMarkup" and http.calls[-1][1]["message_id"] == 77


def test_telegram_callbacks_become_responses():
    http = FakeTelegram()
    tg = TelegramNotifier("TOKEN", http=http, start_polling=False)
    tg.handle_updates([
        {"update_id": 10, "callback_query": {"id": "q1", "from": {"id": 111}, "data": "confirm:5",
                                             "message": {"chat": {"id": 111}}}},
        {"update_id": 11, "message": {"text": "hi"}},                       # not a button press
        {"update_id": 12, "callback_query": {"id": "q2", "from": {"id": 222}, "data": "garbage"}},
    ])
    responses = tg.responses()
    assert [(r.incident_id, r.action, r.address) for r in responses] == [(5, CONFIRM, "111")]
    assert tg.offset == 13
    assert [c[0] for c in http.calls] == ["answerCallbackQuery", "answerCallbackQuery"]
    assert tg.responses() == []


def test_telegram_without_secrets_is_refused(tmp_path):
    cfg = load_config(REPO_ROOT / "config.yaml")
    cfg["paths"]["env_file"] = tmp_path / ".env"           # empty: no token, no chat IDs
    with pytest.raises(ValueError, match="TELEGRAM_CHAT_ID_PRIMARY"):
        build_escalation(cfg, "telegram", log_path=":memory:")
    with pytest.raises(ValueError, match="TOKEN"):
        TelegramNotifier("", start_polling=False)


def test_env_file_parsing_and_telegram_build(tmp_path, monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("TELEGRAM_"):
            monkeypatch.delenv(k)
    env = tmp_path / ".env"
    env.write_text("# comment\nTELEGRAM_BOT_TOKEN='abc:123'\nTELEGRAM_CHAT_ID_PRIMARY=111\n")
    assert load_env(env) == {"TELEGRAM_BOT_TOKEN": "abc:123", "TELEGRAM_CHAT_ID_PRIMARY": "111"}
    cfg = load_config(REPO_ROOT / "config.yaml")
    cfg["paths"]["env_file"] = env
    manager, _, _ = build_escalation(cfg, "telegram", notifier=ConsoleNotifier(printer=lambda *a: None),
                                     log_path=":memory:")
    assert list(manager.contacts) == ["primary"] and manager.contacts["primary"].address == "111"


def test_demo_default_is_console_channel():
    cfg = load_config(REPO_ROOT / "config.yaml")
    assert cfg["escalation"]["channel"] == "console"
    _, _, notifier = build_escalation(cfg, log_path=":memory:")
    assert isinstance(notifier, ConsoleNotifier)


# ---------------------------------------------------------------- HARD RULE guard

def test_no_code_can_place_a_phone_call():
    """The software must never dial 112 or any number. No channel has a call method,
    and no telephony API or dialling command appears anywhere in the package."""
    assert not [m for m in dir(Notifier) if re.search(r"call|dial|phone", m, re.I) and not m.startswith("_")]
    forbidden = re.compile(r"twilio|vonage|nexmo|plivo|sinch|calls\.create|\bATD|sip:|dial\(|place_call|"
                           r"\[\s*['\"]open['\"]|xdg-open|FaceTime", re.I)   # ["open", "tel:112"] would start a call
    for path in (REPO_ROOT / "src" / "akkunet").rglob("*.py"):
        hits = forbidden.findall(path.read_text())
        assert not hits, f"{path.name} contains {hits}"
