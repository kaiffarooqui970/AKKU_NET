"""Fire/smoke escalation: human-in-the-loop, never an automatic emergency call.

HARD RULE: this software NEVER dials 112 or any other number, in any mode. It
notifies a responsible person, and if they confirm the fire it hands them the
number and a ready-to-read text. They make the call from their own phone. A real
installation must connect to the facility's certified fire alarm system
(DIN 14675, connection to the Leitstelle) instead; see docs/fire_escalation.md.

Flow of one incident:
  smoke seen in >= trigger_min_frames frames  -> incident opened, snapshot saved,
                                                 alert to the PRIMARY contact
  no answer within response_timeout_seconds   -> same alert to the SECONDARY
                                                 contact + site alarm
  "Confirm fire" (either contact)             -> call script (112, address, zone,
                                                 what was seen) to the confirmer
                                                 + site alarm
  "False alarm"  (either contact)             -> logged, snapshot copied to the
                                                 hard-negative candidates folder
The first decision wins; later answers are logged and ignored. Answers from
addresses that are not configured contacts are ignored.
"""
import shutil
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from akkunet.notify import CONFIRM, DISMISS, Contact, Notifier

OPEN, ESCALATED, CONFIRMED, DISMISSED = "open", "escalated", "confirmed", "dismissed"


# ---------------------------------------------------------------- event log (SQLite)

class EventLog:
    """incidents: one row per incident; events: everything that happened, with timestamps."""

    def __init__(self, path: Path | str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                zone TEXT, detected TEXT, detected_at REAL, snapshot TEXT,
                status TEXT, resolved_at REAL, resolved_by TEXT, response_seconds REAL);
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id INTEGER, at REAL, kind TEXT, contact TEXT, detail TEXT);
        """)

    def open_incident(self, zone, detected, at, snapshot) -> int:
        cur = self.db.execute("INSERT INTO incidents (zone, detected, detected_at, snapshot, status) "
                              "VALUES (?, ?, ?, ?, ?)", (zone, detected, at, str(snapshot), OPEN))
        self.db.commit()
        return cur.lastrowid

    def event(self, incident_id, at, kind, contact="", detail=""):
        self.db.execute("INSERT INTO events (incident_id, at, kind, contact, detail) VALUES (?, ?, ?, ?, ?)",
                        (incident_id, at, kind, contact, detail))
        self.db.commit()

    def set_status(self, incident_id, status):
        self.db.execute("UPDATE incidents SET status = ? WHERE id = ?", (status, incident_id))
        self.db.commit()

    def resolve(self, incident_id, status, at, by):
        self.db.execute("UPDATE incidents SET status = ?, resolved_at = ?, resolved_by = ?, "
                        "response_seconds = ? - detected_at WHERE id = ?", (status, at, by, at, incident_id))
        self.db.commit()

    def incident(self, incident_id) -> dict:
        cur = self.db.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,))
        return dict(zip([c[0] for c in cur.description], cur.fetchone()))

    def events(self, incident_id) -> list[dict]:
        cur = self.db.execute("SELECT * FROM events WHERE incident_id = ? ORDER BY id", (incident_id,))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


# ---------------------------------------------------------------- local site alarm (demo)

@dataclass
class SiteAlarm:
    """Loud local alarm for the demo: plays a sound file repeatedly in the background.

    `spawn` is injectable so tests stay silent. A real site would trigger the
    facility's own alarm system here (relay / BMA input), not a laptop speaker.
    """
    sound: str = "/System/Library/Sounds/Funk.aiff"
    repeats: int = 10
    spawn: callable = None
    process: object = None

    def __post_init__(self):
        if self.spawn is None:
            self.spawn = lambda cmd: subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @property
    def active(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self) -> bool:
        """Start the alarm unless it's already sounding. Returns True if started now."""
        if self.active:
            return False
        if shutil.which("afplay") is None:
            print("\a" * 3 + "SITE ALARM (afplay not available)", flush=True)
            return True
        loop = f"for i in $(seq {self.repeats}); do afplay -v 1 '{self.sound}'; done"
        self.process = self.spawn(["/bin/sh", "-c", loop])
        return True

    def stop(self):
        if self.active:
            self.process.terminate()


# ---------------------------------------------------------------- trigger

@dataclass
class SmokeTrigger:
    """Fires once when smoke has been seen in `min_frames` consecutive frames.

    Consecutive (unlike the battery alert) because a smoke detector should not
    add up scattered single-frame flickers over minutes. Re-arms after `reset()`.
    """
    min_frames: int
    count: int = 0
    fired: bool = False

    def update(self, smoke_seen: bool) -> bool:
        self.count = self.count + 1 if smoke_seen else 0
        if self.count >= self.min_frames and not self.fired:
            self.fired = True
            return True
        return False

    def reset(self):
        self.count, self.fired = 0, False


# ---------------------------------------------------------------- escalation

@dataclass
class Facility:
    name: str
    address: str
    emergency_number: str = "112"


@dataclass
class Incident:
    id: int
    zone: str
    detected: str
    detected_at: float
    snapshot: Path | None
    status: str = OPEN
    notified: list = field(default_factory=list)     # contact roles alerted so far


class EscalationManager:
    """Runs incidents through the flow in the module docstring. Call `tick()` every frame."""

    def __init__(self, notifier: Notifier, log: EventLog, contacts: dict[str, Contact], facility: Facility,
                 response_timeout: float, site_alarm: SiteAlarm, hard_negative_dir: Path,
                 clock=time.time, alert_template: str = None, call_template: str = None):
        if "primary" not in contacts:
            raise ValueError("escalation needs at least a 'primary' contact")
        self.notifier, self.log, self.contacts = notifier, log, contacts
        self.facility, self.timeout, self.alarm = facility, response_timeout, site_alarm
        self.hard_negative_dir = Path(hard_negative_dir)
        self.clock = clock
        self.alert_template = alert_template or DEFAULT_ALERT
        self.call_template = call_template or DEFAULT_CALL_SCRIPT
        self.incidents: dict[int, Incident] = {}

    # -- opening
    def trigger(self, zone: str, detected: str, snapshot: Path | None) -> Incident:
        now = self.clock()
        inc_id = self.log.open_incident(zone, detected, now, snapshot)
        inc = Incident(inc_id, zone, detected, now, snapshot)
        self.incidents[inc_id] = inc
        self.log.event(inc_id, now, "detected", detail=detected)
        self._alert(inc, "primary")
        return inc

    def _alert(self, inc: Incident, role: str):
        contact = self.contacts[role]
        text = self.alert_template.format(zone=inc.zone, detected=inc.detected, time=_fmt(inc.detected_at),
                                          facility=self.facility.name, incident=inc.id,
                                          escalated=" (no answer from first contact)" if role != "primary" else "")
        try:
            self.notifier.send_alert(contact, inc.id, text, inc.snapshot)
            self.log.event(inc.id, self.clock(), "notified", contact.name, role)
        except Exception as e:  # a failed message must never stop the escalation
            self.log.event(inc.id, self.clock(), "notify_failed", contact.name, str(e))
        inc.notified.append(role)

    # -- every frame
    def tick(self) -> None:
        now = self.clock()
        for r in self.notifier.responses():
            self.handle(r)
        for inc in self.incidents.values():
            if inc.status == OPEN and now - inc.detected_at >= self.timeout:
                inc.status = ESCALATED
                self.log.set_status(inc.id, ESCALATED)
                self.log.event(inc.id, now, "timeout", detail=f"no answer after {self.timeout:.0f} s")
                if "secondary" in self.contacts:
                    self._alert(inc, "secondary")
                self._sound_alarm(inc, "no answer")

    def handle(self, r) -> None:
        inc = self.incidents.get(r.incident_id)
        who = next((c for c in self.contacts.values() if str(c.address) == str(r.address)), None)
        if inc is None or who is None:
            self.log.event(r.incident_id, r.received_at, "ignored_response", str(r.address),
                           "unknown incident" if inc is None else "not a configured contact")
            return
        if inc.status in (CONFIRMED, DISMISSED):
            self.log.event(inc.id, r.received_at, "late_response", who.name, r.action)
            return
        now = self.clock()
        if hasattr(self.notifier, "remove_buttons"):
            self.notifier.remove_buttons(inc.id)
        if r.action == CONFIRM:
            inc.status = CONFIRMED
            self.log.resolve(inc.id, CONFIRMED, now, who.name)
            self.log.event(inc.id, now, "confirmed", who.name)
            self.notifier.send_text(who, self.call_script(inc, who))
            self.log.event(inc.id, now, "call_script_sent", who.name)
            self._sound_alarm(inc, "confirmed")
        elif r.action == DISMISS:
            inc.status = DISMISSED
            self.log.resolve(inc.id, DISMISSED, now, who.name)
            self.log.event(inc.id, now, "dismissed", who.name)
            saved = self._save_hard_negative(inc)
            self.alarm.stop()
            for c in self.contacts.values():   # tell everyone who was alerted that it's over
                if c.role in inc.notified:
                    self.notifier.send_text(c, f"Incident {inc.id}: false alarm ({who.name}). No action needed.")
            self.log.event(inc.id, now, "hard_negative_saved", detail=str(saved))

    def call_script(self, inc: Incident, who: Contact) -> str:
        f = self.facility
        return self.call_template.format(number=f.emergency_number, facility=f.name, address=f.address,
                                         zone=inc.zone, detected=inc.detected, time=_fmt(inc.detected_at),
                                         name=who.name)

    def _sound_alarm(self, inc, reason):
        if self.alarm.start():
            self.log.event(inc.id, self.clock(), "site_alarm", detail=reason)

    def _save_hard_negative(self, inc) -> Path | None:
        if not inc.snapshot or not Path(inc.snapshot).exists():
            return None
        self.hard_negative_dir.mkdir(parents=True, exist_ok=True)
        dest = self.hard_negative_dir / f"incident{inc.id:05d}_{Path(inc.snapshot).name}"
        shutil.copy2(inc.snapshot, dest)
        return dest


def _fmt(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")


DEFAULT_ALERT = ("⚠️ {facility}: possible SMOKE/FIRE{escalated}\n"
                 "Zone: {zone}\nDetected: {detected}\nTime: {time}\nIncident: {incident}\n"
                 "Please check the camera image / the zone and answer.")

DEFAULT_CALL_SCRIPT = (
    "🔥 Fire CONFIRMED by you. Call the fire brigade NOW from your phone:\n"
    "📞 {number}  (tel:{number})\n\n"
    "Read out:\n"
    "\"Hier ist {name}, {facility}, {address}. "
    "Brand in {zone}. Erkannt um {time}: {detected}.\"\n\n"
    "This system does NOT call anyone. The site alarm has been triggered.")


# ---------------------------------------------------------------- wiring from config

def build_escalation(cfg: dict, channel: str | None = None, notifier: Notifier | None = None,
                     log_path=None) -> tuple["EscalationManager", SmokeTrigger, Notifier]:
    """Manager + trigger + channel from config.yaml (and .env for Telegram).

    console: contact addresses are their role names, decisions come from demo keys.
    telegram: token and chat IDs from .env; a contact without a chat ID is left out
    (with a warning), but a primary contact is required.
    """
    from akkunet.notify import ConsoleNotifier, TelegramNotifier, load_env

    esc, paths = cfg["escalation"], cfg["paths"]
    channel = channel or esc.get("channel", "console")
    contacts = {}
    if channel == "console":
        for role, c in esc["contacts"].items():
            contacts[role] = Contact(role, c["name"], role)
        notifier = notifier or ConsoleNotifier()
    elif channel == "telegram":
        env = load_env(paths["env_file"])
        for role, c in esc["contacts"].items():
            chat = env.get(c.get("telegram_chat_id_env", ""), "")
            if chat:
                contacts[role] = Contact(role, c["name"], chat)
            else:
                print(f"WARNING: no {c.get('telegram_chat_id_env')} in .env: {role} contact ({c['name']}) left out")
        if "primary" not in contacts:
            raise ValueError("Telegram mode needs TELEGRAM_CHAT_ID_PRIMARY in .env (see .env.example)")
        notifier = notifier or TelegramNotifier(env.get("TELEGRAM_BOT_TOKEN", ""))
    else:
        raise ValueError(f"escalation channel must be console or telegram, not {channel!r}")

    f = esc["facility"]
    alarm_cfg = esc.get("site_alarm") or {}
    manager = EscalationManager(
        notifier=notifier, log=EventLog(log_path or paths["escalation_db"]), contacts=contacts,
        facility=Facility(f["name"], f["address"], str(f.get("emergency_number", "112"))),
        response_timeout=esc["response_timeout_seconds"],
        site_alarm=SiteAlarm(alarm_cfg.get("sound", SiteAlarm.sound), alarm_cfg.get("repeats", 10)),
        hard_negative_dir=paths["hard_negative_candidates"])
    return manager, SmokeTrigger(esc["trigger_min_frames"]), notifier
