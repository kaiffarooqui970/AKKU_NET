"""Notification channels for the fire/smoke escalation (human-in-the-loop).

HARD RULE: no channel can place a phone call. The interface only sends
messages and receives a person's decision. The emergency call (112) is always
made by a human from their own phone; we only hand them the number and a text
to read out.

Channels:
  ConsoleNotifier   demo/test default: prints messages; decisions come from
                    keys in the demo window (simulating the person's phone)
  TelegramNotifier  Telegram bot; token and chat IDs from a local .env file
                    (never committed). Later: SMS / push behind the same interface.
"""
import json
import os
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

CONFIRM = "confirm"
DISMISS = "dismiss"


@dataclass
class Contact:
    role: str        # "primary" / "secondary"
    name: str        # e.g. "Schichtleitung"
    address: str     # channel address: Telegram chat ID, phone number for SMS later, ...


@dataclass
class Response:
    """A person's decision on an incident."""
    incident_id: int
    action: str      # CONFIRM or DISMISS
    address: str     # who answered (channel address, checked against the contacts)
    received_at: float


class Notifier(ABC):
    """What every channel must do. Deliberately NO call/dial method."""

    @abstractmethod
    def send_alert(self, contact: Contact, incident_id: int, text: str, snapshot: Path | None) -> None:
        """Send the alert with two choices: 'Confirm fire' / 'False alarm'."""

    @abstractmethod
    def send_text(self, contact: Contact, text: str) -> None:
        """Send a plain follow-up message (call script, status update)."""

    @abstractmethod
    def responses(self) -> list[Response]:
        """Decisions received since the last call. Must not block."""

    def close(self) -> None:
        """Stop background work, if any."""


# ---------------------------------------------------------------- console (demo default)

class ConsoleNotifier(Notifier):
    """Prints every message. `inject()` simulates a person tapping a button."""

    def __init__(self, printer=print):
        self.printer = printer
        self.sent: list[tuple[str, str, str]] = []   # (kind, contact address, text), for tests
        self._pending: list[Response] = []

    def send_alert(self, contact, incident_id, text, snapshot):
        self.sent.append(("alert", contact.address, text))
        self.printer(f"\n[DEMO -> {contact.role} {contact.name}] {text}\n"
                     f"   snapshot: {snapshot}\n   buttons: [Confirm fire] [False alarm]")

    def send_text(self, contact, text):
        self.sent.append(("text", contact.address, text))
        self.printer(f"\n[DEMO -> {contact.role} {contact.name}]\n{text}")

    def inject(self, incident_id: int, action: str, address: str) -> None:
        self._pending.append(Response(incident_id, action, address, time.time()))

    def responses(self):
        out, self._pending = self._pending, []
        return out


# ---------------------------------------------------------------- Telegram

def load_env(path: Path) -> dict[str, str]:
    """Minimal .env reader (KEY=value per line, # comments). Real environment variables win."""
    values = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    values.update({k: v for k, v in os.environ.items() if k.startswith("TELEGRAM_")})
    return values


def telegram_http(url: str, data: dict, files: dict | None = None, timeout: float = 30) -> dict:
    """POST to the Telegram Bot API and return the JSON 'result'. Raises on API errors."""
    import requests

    reply = requests.post(url, data=data, files=files, timeout=timeout).json()
    if not reply.get("ok"):
        raise RuntimeError(f"Telegram API error: {reply.get('description', reply)}")
    return reply["result"]


class TelegramNotifier(Notifier):
    """Telegram bot channel.

    Alerts are photos with an inline keyboard; a tap sends a callback
    'confirm:<id>' / 'dismiss:<id>' that a background thread picks up via
    getUpdates (long polling). `http` is injectable so tests never touch the network.

    Note: Telegram buttons cannot open tel: links (only http(s)/tg). The call
    script therefore puts the number in the text, where phone apps usually make
    it tappable; check on the real phones before relying on it.
    """

    def __init__(self, token: str, http=telegram_http, poll_timeout: int = 20, start_polling: bool = True):
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is missing (put it in .env)")
        self.base = f"https://api.telegram.org/bot{token}"
        self.http = http
        self.poll_timeout = poll_timeout
        self.offset = 0
        self.alert_messages: dict[int, list[tuple[str, int]]] = {}   # incident -> [(chat, message_id)]
        self._queue: queue.Queue[Response] = queue.Queue()
        self._stop = threading.Event()
        self._thread = None
        if start_polling:
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()

    def send_alert(self, contact, incident_id, text, snapshot):
        keyboard = {"inline_keyboard": [[
            {"text": "🔥 Confirm fire", "callback_data": f"{CONFIRM}:{incident_id}"},
            {"text": "✅ False alarm", "callback_data": f"{DISMISS}:{incident_id}"},
        ]]}
        data = {"chat_id": contact.address, "caption": text, "reply_markup": json.dumps(keyboard)}
        if snapshot and Path(snapshot).exists():
            with open(snapshot, "rb") as f:
                msg = self.http(f"{self.base}/sendPhoto", data, files={"photo": f})
        else:
            data["text"] = data.pop("caption")
            msg = self.http(f"{self.base}/sendMessage", data)
        self.alert_messages.setdefault(incident_id, []).append((contact.address, msg["message_id"]))

    def send_text(self, contact, text):
        self.http(f"{self.base}/sendMessage", {"chat_id": contact.address, "text": text})

    def remove_buttons(self, incident_id: int) -> None:
        """After a decision, take the buttons off every alert of this incident (no double answers)."""
        for chat, message_id in self.alert_messages.get(incident_id, []):
            try:
                self.http(f"{self.base}/editMessageReplyMarkup",
                          {"chat_id": chat, "message_id": message_id, "reply_markup": json.dumps({})})
            except RuntimeError:
                pass  # message too old / already edited: harmless

    def handle_updates(self, updates: list[dict]) -> None:
        """Turn callback queries into Responses (public for tests)."""
        for u in updates:
            self.offset = max(self.offset, u["update_id"] + 1)
            cq = u.get("callback_query")
            if not cq:
                continue
            try:
                self.http(f"{self.base}/answerCallbackQuery", {"callback_query_id": cq["id"]})
            except RuntimeError:
                pass
            action, _, incident = (cq.get("data") or "").partition(":")
            if action in (CONFIRM, DISMISS) and incident.isdigit():
                chat = str(cq.get("message", {}).get("chat", {}).get("id", cq["from"]["id"]))
                self._queue.put(Response(int(incident), action, chat, time.time()))

    def _poll_loop(self):
        while not self._stop.is_set():
            try:
                updates = self.http(f"{self.base}/getUpdates",
                                    {"offset": self.offset, "timeout": self.poll_timeout,
                                     "allowed_updates": json.dumps(["callback_query"])},
                                    timeout=self.poll_timeout + 10)
                self.handle_updates(updates)
            except Exception as e:  # network hiccup: log and retry, never kill the demo
                print(f"WARNING: Telegram polling failed ({e}); retrying in 5 s")
                self._stop.wait(5)

    def responses(self):
        out = []
        while not self._queue.empty():
            out.append(self._queue.get_nowait())
        return out

    def close(self):
        self._stop.set()
