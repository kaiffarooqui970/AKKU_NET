# Fire / smoke escalation (human-in-the-loop)

## The hard rule

**This software never calls 112, or any other number, in any mode.** It alerts
a responsible person, waits for their decision, and if they confirm a fire it
gives them the emergency number and a text to read out. **The emergency call is
always made by a human, from their own phone.**

Why:

- **False alarms.** A camera model will fire on steam, dust, sun glare or (see
  `reports/finding_webcam_face_false_positives.md`) things that aren't there at
  all. Automatic calls would burden the Leitstelle (dispatch centre) and could
  make the facility liable for false-alarm costs.
- **Responsibility.** Calling the fire brigade is a judgement call that belongs
  to a person on site who can see the situation, give details and answer the
  dispatcher's questions.
- **The legal and technical route already exists.** Automatic fire alarms in
  Germany go through certified fire alarm systems, not through demo software
  (see below).

This is enforced in code, not just in documentation:

- The channel interface (`akkunet.notify.Notifier`) has only `send_alert`,
  `send_text` and `responses`. There is no call or dial method.
- `tests/test_escalation.py::test_no_code_can_place_a_phone_call` fails if any
  module mentions telephony APIs (Twilio, Vonage, …), modem dial commands, SIP,
  or launches macOS `open` (which could open a `tel:` link).

## Flow

```
smoke seen in >= trigger_min_frames consecutive frames
        │  (demo: key "s" toggles SIMULATED smoke; no smoke model yet)
        ▼
incident opened, snapshot saved, logged ─► alert to PRIMARY contact
        │                                   (snapshot, zone, time,
        │                                    [Confirm fire] [False alarm])
        ├── "Confirm fire" ─► call script to that person: 112, facility
        │                     address, zone, what was seen, as a text to read
        │                     out; LOCAL SITE ALARM sounds
        ├── "False alarm"  ─► logged; snapshot copied to
        │                     data/escalation/hard_negative_candidates/;
        │                     everyone alerted so far is told it's over
        └── no answer within response_timeout_seconds (default 90)
                          ─► same alert to SECONDARY contact + SITE ALARM;
                             either contact can still confirm or dismiss
```

- The **first decision wins**. Later answers are logged as `late_response` and
  ignored.
- Answers from people who are **not configured contacts** are ignored and logged.
- A failed message (network down) is logged as `notify_failed`, and the
  escalation continues (timeout, second contact, site alarm).

## Configuration

`config.yaml` → `escalation:` sets the trigger frames, timeout, zone, facility
name, address and emergency number, contact names, and the site alarm sound.

Secrets live in a local `.env` (git-ignored; template in `.env.example`):
`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID_PRIMARY`, `TELEGRAM_CHAT_ID_SECONDARY`.

**Demo/test mode is the default** (`channel: console`). Messages are printed in
the terminal, and in the demo window keys `c` / `f` stand in for the contact
pressing "Confirm fire" / "False alarm". No network, no phone. Tests mock the
Telegram API completely.

```bash
python -m akkunet.app --escalation console               # default: demo, no network
python -m akkunet.app --escalation telegram              # real Telegram messages (needs .env)
python -m akkunet.app --escalation off
```

### Telegram setup (demo channel)

1. In Telegram, open @BotFather → `/newbot` → copy the token into `.env`.
2. Each contact opens a chat with the bot and sends `/start`.
3. Open `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy each
   `"chat":{"id":…}` into `.env`.

**Limitation:** Telegram buttons can only open `http(s)`/`tg` links, not
`tel:`. The call script therefore puts `112` / `tel:112` in the message text,
where phone apps usually make it tappable. **Check this on the actual phones**
before relying on it. A later SMS or push channel can carry a real `tel:` link.

## Event log

SQLite at `data/escalation/events.sqlite` (git-ignored):

- `incidents`: zone, what was detected, detection time, snapshot, status
  (`open` / `escalated` / `confirmed` / `dismissed`), resolved by whom and when,
  `response_seconds` (from detection to decision)
- `events`: every step with a timestamp: `detected`, `notified` (who),
  `notify_failed`, `timeout`, `site_alarm`, `confirmed` / `dismissed` (who),
  `call_script_sent`, `hard_negative_saved`, `late_response`, `ignored_response`

```bash
sqlite3 data/escalation/events.sqlite "SELECT id, status, resolved_by, response_seconds FROM incidents"
```

## Adding SMS or push later

Implement `Notifier` (`send_alert`, `send_text`, `responses`) for the new
channel and select it in `build_escalation`. The escalation logic doesn't
change. Do not add a call method. The guard test will fail.

## Privacy

Snapshots can show employees and visitors. They stay local
(`data/escalation/`, git-ignored). On Telegram they leave the site, so a real
deployment needs a data-protection review (GDPR, works council /
Betriebsrat) and an agreed retention period. False-alarm snapshots are
collected as hard-negative *candidates*: review them before training, and
blur or exclude people.

## What a real installation must do instead

This workflow is a **demo of the human decision step**, not a fire alarm
system. A real facility must integrate with its **certified fire alarm system
(Brandmeldeanlage, BMA)**, planned, installed and operated under
**DIN 14675** together with **DIN VDE 0833-2**. The BMA forwards alarms to the
fire brigade's **Leitstelle** through a certified transmission unit
(Übertragungseinrichtung) under the local fire brigade's technical connection
rules (TAB).

For AkkuNet that means:

- A camera/AI detection should, at most, feed the BMA as an **additional,
  clearly marked input** (for example a pre-alarm or a verification aid for
  staff). Whether and how that's allowed must be decided with a certified BMA
  specialist (Fachfirma / Fachplaner nach DIN 14675) and the fire brigade.
- The certified detectors (flame and heat detectors, often thermal cameras
  certified for this use) stay the life-safety system. Software like this
  never replaces them.
- Telegram or SMS is **not a life-safety channel**: no delivery guarantee, no
  monitoring of the transmission path. Use it for information only.

The standards references above are a pointer, not legal advice. Confirm the
requirements for a specific site with a fire-protection specialist.
