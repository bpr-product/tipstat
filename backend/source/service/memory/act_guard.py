"""Backend checks on the assistant's write calls, run before anything reaches the world. The prompt asks the model to
behave; this makes the important rules hold even when it does not.

Checks (a blocked call returns advice to the model as the tool result; nothing is executed):
- information the user asked to forget (salted-hash tombstones kept in process memory only) must not appear in anything
  written; matching is on distinctive terms (names, numbers, adjacent word pairs), never on common words;
- private facts (the user asked to keep them private) must not appear in anything sent to other people, same matching;
- a message goes only to a recipient that appeared in the request, memory or an earlier tool result;
- a meeting is cancelled for everyone only if the user asked to cancel or delete something;
- the user's own quiet-hours rules (a structured constraint stored with the rule) hold for the recipient, the thread
  being answered, anyone mentioned, and the scheduled send time;
- calendar writes must not overlap a timed commitment that only memory knows (a flight, an appointment);
- an identical write is never done twice, and a job has a cap on changes.
Limits: matching is by words and wall-clock time (timezone offsets are ignored), recipient identity beyond names and
handles is not resolved, and a commitment with no stored time of day is not enforced.
"""
import json
import re
from datetime import date, datetime, timedelta

from backend.source.dal.repositories.memory_repository import MemoryRepository
from backend.source.utils.text import leak_terms, message_terms, words

MESSAGES_NOW = {"slack_send_message", "email_send", "email_reply"}
SENT_TO_OTHERS = MESSAGES_NOW | {"slack_schedule_message", "docs_comment", "tasks_update", "calendar_create_event", "calendar_update_event"}
WRITES = SENT_TO_OTHERS | {"reminders_create", "reminders_complete", "calendar_cancel_event", "calendar_respond"}
CALENDAR_WRITES = {"calendar_create_event", "calendar_update_event"}
MAX_WRITES = 12
RECIPIENT_ARGS = {"slack_send_message": ("to",), "slack_schedule_message": ("to",), "email_send": ("to", "cc")}
CANCEL_ASK = re.compile(r"cancel|delete|remove|call off|scrap|drop the (meeting|event)", re.IGNORECASE)
TRAVEL = re.compile(r"flight|train|\bbus\b|airport|boarding|depart|trip|travel|\bcab\b|drive", re.IGNORECASE)


MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
FULL = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]


def date_pattern(iso: str):
    """Regex for the common ways to write a date in a message: 2026-11-20, Nov 20, 20th November, ..."""
    try:
        d = date.fromisoformat(iso)
    except ValueError:
        return None
    mon, day = f"(?:{MONTHS[d.month - 1]}|{FULL[d.month - 1]})\\.?", rf"0?{d.day}(?:st|nd|rd|th)?"
    return re.compile(rf"{iso}|\b{mon}\s+{day}\b|\b{day}\s+(?:of\s+)?{mon}", re.IGNORECASE)


def _wall(s) -> datetime | None:
    """Wall-clock time of an ISO string (the offset is dropped on purpose; see module limits)."""
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _in_window(t: datetime, frm: str, until: str) -> bool:
    now, a, b = t.hour * 60 + t.minute, _minutes(frm), _minutes(until)
    return a <= now < b if a <= b else (now >= a or now < b)


class ActGuard:
    def __init__(self, repo: MemoryRepository, tools_url: str, call_tool, request: str = "", memory: str = "") -> None:
        self.tools_url, self.call_tool, self.request = tools_url, call_tool, request
        self.known = (request + " " + memory).lower()  # what the assistant legitimately knows about, grown by observe()
        self.now = _wall(repo.get_meta("now", ""))
        self.rules = []
        for r in repo.active_by_kind("RULE", "user"):  # only the user's own rules, never text someone else wrote
            c = (json.loads(r["data"]) if r["data"] else {}).get("constraint")
            if c:
                self.rules.append(c)
        self.private_uni, self.private_bi = set(), set()
        private = repo.sensitive_facts()
        if private:
            df = repo.doc_freq(3)
            for f in private:
                u, b = leak_terms(f["subject"], f["relation"], f["value"], f["quote"])
                self.private_uni |= {w for w in u if df.get(w, 0) <= 2}  # a name that appears all over is not a leak of this fact
                self.private_bi |= b
        self.forgotten, self.hash_term = repo.tombstones(), repo.hash_term
        self.commitments = []
        if self.now:
            today = self.now.date().isoformat()
            horizon = (self.now.date() + timedelta(days=60)).isoformat()
            for f in repo.upcoming(today, horizon, limit=80):
                start = _wall(f["when_ts"]) if f["when_ts"] else None
                if start:
                    text = f"{f['subject']} {f['relation']} {f['value']} {f['quote']}"
                    self.commitments.append((start, text, bool(TRAVEL.search(text)), words(text, 4), f"{f['relation']}: {f['value']}"))
        self.stale_dates = [(d, p) for d, p in ((d, date_pattern(d)) for d in sorted(repo.outdated_dates())) if p]
        self.seen, self.writes = set(), 0

    def observe(self, result: str) -> None:
        """Tool results extend what counts as known (contact handles, channel names) for the recipient check."""
        self.known += " " + str(result).lower()

    def __call__(self, name: str, args: dict) -> str | None:
        """Returns advice (the call is blocked) or None (allowed)."""
        if name not in WRITES:
            return None
        sig = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
        if sig in self.seen:
            return "blocked: this exact action was already done in this job; do not repeat it."
        if self.writes >= MAX_WRITES:
            return "blocked: too many changes in one job. Stop and report what was done."
        reason = self._check(name, args)
        if reason:
            return reason
        self.seen.add(sig)
        self.writes += 1
        return None

    def _check(self, name: str, args: dict) -> str | None:
        text = " ".join(str(v) for v in args.values())
        used, pairs = message_terms(text)
        if self.forgotten and (any(self.hash_term("u:" + w) in self.forgotten for w in used)
                               or any(self.hash_term("b:" + p) in self.forgotten for p in pairs)):
            return "blocked: this contains information you were asked to forget. Leave it out entirely."
        if name in SENT_TO_OTHERS and (used & self.private_uni or pairs & self.private_bi):
            return "blocked: this contains information the user marked private. Leave it out; keep the message to what is needed."
        if name in SENT_TO_OTHERS:
            for iso, pattern in self.stale_dates:
                if pattern.search(text):
                    return (f"blocked: this mentions {iso}, a date that was replaced and is out of date. State only the current "
                            "date or value; do not mention the old one.")
        for key in RECIPIENT_ARGS.get(name, ()):
            raw = args.get(key)
            for item in (raw if isinstance(raw, list) else re.split(r"[,;]", str(raw or ""))):
                item = str(item).strip().lower().lstrip("@#")  # "@kavya" is known if "kavya" appeared
                if item and item not in self.known:
                    return (f"blocked: '{item}' has not appeared in the request, memory or any result so far. Look the person up "
                            "first (contacts_lookup or a search) and use the exact handle or address it returns.")
        if name == "calendar_cancel_event" and not CANCEL_ASK.search(self.request):
            return ("blocked: cancelling removes the meeting for everyone and the user did not ask to cancel. To drop out of "
                    "someone else's meeting use calendar_respond with declined, or ask the user in your reply.")
        if self.rules and name in MESSAGES_NOW | {"slack_schedule_message"}:
            when = _wall(args.get("send_at")) if name == "slack_schedule_message" else self.now
            who_text = text.lower() + " " + self._thread_author(args)
            for rule in self.rules:
                if when and re.search(rf"(?<![a-z0-9]){re.escape(rule['who'])}(?![a-z0-9])", who_text) and _in_window(when, rule["from"], rule["until"]):
                    return (f"blocked: the user's standing rule: no contact with {rule['who']} between {rule['from']} and "
                            f"{rule['until']}, and this would reach them at {when:%H:%M}. Do not send it now: schedule it with "
                            f"slack_schedule_message for after {rule['until']} (or set a reminder) and tell the user.")
        if name in CALENDAR_WRITES:
            start = _wall(args.get("start"))
            if start:
                end = _wall(args.get("end")) or start + timedelta(hours=1)
                title_words = words(str(args.get("title", "")), 4)
                for c_start, _, travel, c_words, label in self.commitments:
                    lo = c_start - timedelta(hours=3) if travel else c_start - timedelta(minutes=30)
                    hi = c_start.replace(hour=23, minute=59) if travel else c_start + timedelta(minutes=90)
                    if start < hi and end > lo and not (title_words & c_words):
                        return (f"blocked: this overlaps the user's commitment ({label}, {c_start:%a %d %b %H:%M}) that is not on "
                                "the calendar. Skip this item or choose a clear time, and mention the clash to the user.")
        return None

    def _thread_author(self, args: dict) -> str:
        """Who wrote the message a reply is attached to: replying in someone's thread reaches them too."""
        thread_id = args.get("thread_id")
        if thread_id:
            try:
                out = json.loads(self.call_tool(self.tools_url, "slack_read_thread", {"message_id": thread_id}))
                return str((out.get("parent") or {}).get("from", "")).lower()
            except (ValueError, AttributeError):
                return ""
        return ""
