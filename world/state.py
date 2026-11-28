"""The mock world: Slack, Calendar, Email, Docs, Tasks, Reminders and Contacts for one persona.

State is plain JSON so it can be snapshotted, diffed and checked:

  initial.json        people, channels, messages, events, emails, docs, tasks, reminders
                      at the start (backlog the assistant can only see through tools)
  world_events.jsonl  dated operations that change the world over the period:
                      {"id", "date", "ts", "source": "inbound"|"user"|"assistant", "op", "data"}
                        inbound    other people (a colleague's message, an invite, a ticket update)
                        user       the user acting in an app themself (a DM they sent, an RSVP)
                        assistant  tool calls the assistant made during a chat
                      ops: add_message, add_channel, add_event, update_event, cancel_event,
                           rsvp_event, add_reminder, complete_reminder, add_person,
                           add_email, add_doc, update_doc, add_doc_comment,
                           add_task, update_task, add_task_comment

The harness loads the world "as of" a day (initial + every event up to that day),
sets the clock, lets a submission act through the tools, then reads the audit log
and final state to grade what it did.

Every list/search tool is capped and paged, like the real APIs: you cannot pull the
whole history in one call, and long bodies (email, docs) are clipped in list views.
"""
from __future__ import annotations

import copy
import datetime as dt
import itertools
import json
import threading
from pathlib import Path
from typing import Any

WORK_START, WORK_END = 9, 21  # hours searched by find_free_slots by default
MAX_LIST = 50                 # hard cap on any list/search result
SNIPPET = 140                 # characters of a body shown in list views
DOC_PAGE = 6000               # characters of a doc returned per docs_read call
RESPONSES = ("accepted", "tentative", "declined")
TASK_STATUSES = ("todo", "in_progress", "in_review", "blocked", "done", "wont_do")

_EVENTS_CACHE: dict[str, tuple[float, list[dict]]] = {}
_CACHE_LOCK = threading.Lock()


class ToolError(Exception):
    pass


def _parse(ts: str) -> dt.datetime:
    try:
        return dt.datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        raise ToolError(f"bad datetime '{ts}'; use ISO 8601 like 2026-11-09T18:00:00+05:30")


def _overlap(a_start, a_end, b_start, b_end) -> bool:
    return a_start < b_end and b_start < a_end


def _cap(limit, default: int = 20) -> int:
    try:
        n = int(limit if limit is not None else default)
    except (TypeError, ValueError):
        raise ToolError(f"limit must be an integer, got {limit!r}")
    return max(1, min(n, MAX_LIST))


def _clip(text: str, n: int = SNIPPET) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _words(q: str) -> list[str]:
    return [w for w in str(q or "").lower().split() if w]


def _load_events(path: Path) -> list[dict]:
    """world_events.jsonl is large (tens of thousands of lines); parse it once per mtime."""
    key = str(path.resolve())
    mtime = path.stat().st_mtime
    with _CACHE_LOCK:
        hit = _EVENTS_CACHE.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
    evs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    evs.sort(key=lambda e: (e["date"], e.get("ts", "")))
    with _CACHE_LOCK:
        _EVENTS_CACHE[key] = (mtime, evs)
    return evs


COLLECTIONS = ("people", "channels", "messages", "events", "reminders", "scheduled", "emails", "docs", "tasks")


class World:
    def __init__(self):
        self.lock = threading.RLock()
        self.state: dict[str, Any] = {"user": {"id": "user", "name": "User", "slack": "@user", "email": ""},
                                      **{k: [] for k in COLLECTIONS}}
        self.clock = "2026-01-01T09:00:00+05:30"
        self.tz = "+05:30"
        self.audit: list[dict] = []
        self._seq = itertools.count(1)
        self._ids = itertools.count(1)
        self.persona_dir: Path | None = None
        self._index()
        # The harness locks the tools outside /act windows, so a system cannot use the
        # world as a substitute memory while answering questions.
        self.locked = False

    # ------------------------------------------------------------ admin ---
    def load(self, persona_dir: str | Path, upto_date: str | None, clock: str | None = None,
             until: str | None = None) -> dict:
        """Reset to initial state + all world events dated <= upto_date (and, with `until`,
        not later than that moment)."""
        with self.lock:
            self.persona_dir = Path(persona_dir)
            wdir = self.persona_dir / "world"
            self.state = json.loads((wdir / "initial.json").read_text(encoding="utf-8"))
            for k in COLLECTIONS:
                self.state.setdefault(k, [])
            self.state["user"].setdefault("email", "")
            self.tz = self.state.get("utc_offset", "+05:30")
            self._index()
            evpath = wdir / "world_events.jsonl"
            if evpath.exists() and upto_date:
                for ev in _load_events(evpath):
                    if ev["date"] > upto_date or (until and _parse(ev["ts"]) > _parse(until)):
                        break
                    self.apply(ev)
            self.clock = clock or f"{upto_date or '2026-01-01'}T23:59:00{self.tz}"
            self.audit = []
            self._ids = itertools.count(1)
            self.locked = False
            return {"ok": True, "clock": self.clock, "counts": {k: len(v) for k, v in self.state.items()
                                                              if isinstance(v, list)}}

    def _index(self) -> None:
        s = self.state
        self._ev_ix = {e["id"]: e for e in s["events"]}
        self._doc_ix = {d["id"]: d for d in s["docs"]}
        self._task_ix = {t["key"]: t for t in s["tasks"]}
        self._msg_ix = {m["id"]: m for m in s["messages"] if m.get("id")}
        for d in s["docs"]:
            d.setdefault("comments", [])
        for t in s["tasks"]:
            t.setdefault("comments", [])

    def apply(self, ev: dict) -> None:
        op, d = ev["op"], copy.deepcopy(ev["data"])
        s = self.state
        if op == "add_message":
            s["messages"].append(d)
            if d.get("id"):
                self._msg_ix[d["id"]] = d
        elif op == "add_channel":
            s["channels"].append(d)
        elif op == "add_event":
            d.setdefault("status", "confirmed")
            s["events"].append(d)
            self._ev_ix[d["id"]] = d
        elif op == "update_event":
            self._event(d.pop("id")).update(d)
        elif op == "cancel_event":
            self._event(d["id"])["status"] = "cancelled"
        elif op == "rsvp_event":
            e = self._event(d["id"])
            e["my_response"] = d["response"]
            if d.get("note"):
                e["response_note"] = d["note"]
        elif op == "add_reminder":
            d.setdefault("status", "open")
            s["reminders"].append(d)
        elif op == "complete_reminder":
            for r in s["reminders"]:
                if r["id"] == d["id"]:
                    r["status"] = "done"
        elif op == "add_person":
            s["people"].append(d)
        elif op == "add_email":
            d.setdefault("folder", "inbox")
            d.setdefault("unread", d["folder"] == "inbox")
            s["emails"].append(d)
        elif op == "add_doc":
            d.setdefault("comments", [])
            s["docs"].append(d)
            self._doc_ix[d["id"]] = d
        elif op == "update_doc":
            doc = self._doc(d["id"])
            if "append" in d:
                doc["body"] = doc.get("body", "").rstrip() + "\n\n" + d["append"]
            if "body" in d:
                doc["body"] = d["body"]
            if "title" in d:
                doc["title"] = d["title"]
            doc["updated_at"] = d.get("updated_at", ev.get("ts"))
            if d.get("by"):
                doc["last_edited_by"] = d["by"]
        elif op == "add_doc_comment":
            self._doc(d.pop("doc_id"))["comments"].append(d)
        elif op == "add_task":
            d.setdefault("comments", [])
            s["tasks"].append(d)
            self._task_ix[d["key"]] = d
        elif op == "update_task":
            t = self._task(d.pop("key"))
            c = d.pop("comment", None)
            t.update(d)
            if c:
                t["comments"].append(c)
            t["updated_at"] = d.get("updated_at", ev.get("ts"))
        elif op == "add_task_comment":
            self._task(d.pop("key"))["comments"].append(d)
        else:
            raise ValueError(f"unknown world op {op}")

    def set_clock(self, clock: str) -> None:
        _parse(clock)
        self.clock = clock

    def snapshot(self) -> dict:
        with self.lock:
            return copy.deepcopy({"clock": self.clock, **self.state})

    def audit_since(self, seq: int = 0) -> list[dict]:
        with self.lock:
            return [a for a in self.audit if a["seq"] > seq]

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}_new{next(self._ids)}"

    # ------------------------------------------------------------- calls ---
    def call(self, name: str, args: dict) -> Any:
        if name not in TOOLS:
            raise ToolError(f"unknown tool '{name}'")
        with self.lock:
            entry = {"seq": next(self._seq), "clock": self.clock, "tool": name, "args": args,
                     "kind": TOOLS[name]["kind"]}
            try:
                if self.locked:
                    raise ToolError("the world's tools are only available during /act")
                result = getattr(self, "t_" + name)(**(args or {}))
                entry["ok"], entry["result"] = True, result
                return result
            except TypeError as e:
                entry["ok"], entry["error"] = False, f"bad arguments: {e}"
                raise ToolError(entry["error"])
            except (KeyError, ValueError, AttributeError) as e:
                entry["ok"], entry["error"] = False, f"invalid input: {e}"
                raise ToolError(entry["error"])
            except ToolError as e:
                entry["ok"], entry["error"] = False, str(e)
                raise
            finally:
                self.audit.append(entry)

    # helpers
    def _me(self) -> str:
        return self.state["user"]["slack"]

    def _person(self, ref: str) -> dict:
        r = str(ref).strip().lstrip("@").lower()
        for p in self.state["people"]:
            if p.get("slack", "").lstrip("@").lower() == r or p["id"].lower() == r or p.get("email", "").lower() == r:
                return p
        raise ToolError(f"no person with handle/id '{ref}'. Use contacts_lookup to find handles.")

    def _event(self, eid: str) -> dict:
        e = self._ev_ix.get(eid)
        if e is None:
            raise ToolError(f"no event '{eid}'")
        return e

    def _doc(self, did: str) -> dict:
        d = self._doc_ix.get(did)
        if d is None:
            raise ToolError(f"no doc '{did}'")
        return d

    def _task(self, key: str) -> dict:
        t = self._task_ix.get(str(key).upper()) or self._task_ix.get(key)
        if t is None:
            raise ToolError(f"no task '{key}'")
        return t

    def _resolve_target(self, to: str) -> dict:
        if to.startswith("#"):
            for c in self.state["channels"]:
                if c["name"] == to:
                    return {"channel": to}
            raise ToolError(f"no channel '{to}'")
        p = self._person(to)
        if not p.get("slack"):
            raise ToolError(f"{p['name']} is not on Slack")
        return {"to": p["slack"]}

    def _visible(self, items, key="ts"):
        return [x for x in items if x.get(key, "") <= self.clock]

    def _replies(self) -> dict:
        n: dict[str, int] = {}
        for m in self.state["messages"]:
            if m.get("thread_id") and m["ts"] <= self.clock:
                n[m["thread_id"]] = n.get(m["thread_id"], 0) + 1
        return n

    @staticmethod
    def _page(items: list, limit, before: str | None, key="ts") -> list:
        """Newest `limit` items older than `before`. `before` may be a timestamp or the id of
        an item in this list (agents naturally page with the oldest id they were given)."""
        items = sorted(items, key=lambda x: x.get(key, ""))
        if before:
            cut = str(before)
            if not cut[:4].isdigit():
                hit = next((x for x in items if x.get("id") == cut), None)
                if hit is None:
                    raise ToolError(f"before must be a timestamp or an id from this list, got '{before}'")
                cut = hit.get(key, "")
            items = [x for x in items if x.get(key, "") < cut]
        return items[-_cap(limit):]

    # ------------------------------------------------------------ slack ---
    def t_slack_list_channels(self):
        me = self._me()
        return [{"name": c["name"], "topic": c.get("topic", ""), "members": len(c.get("members", [])),
                 "is_member": me in c.get("members", [])} for c in self.state["channels"]]

    def t_slack_read_messages(self, conversation: str, limit: int = 20, before: str | None = None):
        """conversation: '#channel' or '@handle' (DMs with that person). Thread replies are
        folded into reply_count; read them with slack_read_thread."""
        me = self._me()
        if conversation.startswith("#"):
            self._resolve_target(conversation)
            msgs = [m for m in self.state["messages"] if m.get("channel") == conversation and not m.get("thread_id")]
        else:
            other = self._person(conversation)["slack"]
            msgs = [m for m in self.state["messages"] if not m.get("channel") and
                    {m.get("from"), m.get("to")} == {me, other}]
        rc = self._replies()
        out = []
        for m in self._page(self._visible(msgs), limit, before):
            m = dict(m)
            if rc.get(m.get("id")):
                m["reply_count"] = rc[m["id"]]
            out.append(m)
        return out

    def t_slack_read_thread(self, message_id: str):
        parent = self._msg_ix.get(message_id)
        if parent is None or parent["ts"] > self.clock:
            raise ToolError(f"no message '{message_id}'")
        replies = [m for m in self.state["messages"] if m.get("thread_id") == message_id]
        return {"parent": parent, "replies": self._page(self._visible(replies), MAX_LIST, None)}

    def t_slack_search_messages(self, query: str, limit: int = 20, from_user: str | None = None,
                                conversation: str | None = None):
        q = _words(query)
        if not q:
            raise ToolError("query is empty")
        frm = self._person(from_user)["slack"] if from_user else None
        hits = []
        for m in self._visible(self.state["messages"]):
            if frm and m.get("from") != frm:
                continue
            if conversation and conversation not in (m.get("channel"), m.get("to"), m.get("from")):
                continue
            t = m["text"].lower()
            if all(w in t for w in q):
                hits.append(m)
        return self._page(hits, limit, None)

    def t_slack_send_message(self, to: str, text: str, thread_id: str | None = None):
        """to: '#channel' or '@handle'. Sends immediately."""
        if not str(text).strip():
            raise ToolError("empty message")
        msg = {"id": self._new_id("msg"), "from": self._me(), "text": text, "ts": self.clock,
               **self._resolve_target(to)}
        if thread_id:
            if thread_id not in self._msg_ix:
                raise ToolError(f"no message '{thread_id}' to reply to")
            msg["thread_id"] = thread_id
        self.state["messages"].append(msg)
        self._msg_ix[msg["id"]] = msg
        return {"sent": True, "id": msg["id"], "ts": msg["ts"]}

    def t_slack_schedule_message(self, to: str, text: str, send_at: str):
        when = _parse(send_at)
        if when <= _parse(self.clock):
            raise ToolError(f"send_at must be in the future; it is now {self.clock}")
        item = {"id": self._new_id("sch"), "text": text, "send_at": send_at, **self._resolve_target(to)}
        self.state["scheduled"].append(item)
        return {"scheduled": True, "id": item["id"], "send_at": send_at}

    # --------------------------------------------------------- calendar ---
    @staticmethod
    def _ev_summary(e: dict) -> dict:
        out = {k: e.get(k) for k in ("id", "title", "start", "end", "organizer", "location", "status")}
        out["attendees"] = e.get("attendees", [])
        out["my_response"] = e.get("my_response", "accepted")
        if e.get("description"):
            out["description"] = _clip(e["description"])
        return out

    @staticmethod
    def _busy(e: dict) -> bool:
        return e.get("status") != "cancelled" and e.get("my_response", "accepted") != "declined"

    def t_calendar_list_events(self, start: str, end: str, limit: int = 50):
        a, b = _parse(start), _parse(end)
        out = [e for e in self.state["events"] if e.get("status") != "cancelled"
               and _overlap(_parse(e["start"]), _parse(e["end"]), a, b)]
        out.sort(key=lambda e: e["start"])
        return [self._ev_summary(e) for e in out[:_cap(limit, 50)]]

    def t_calendar_get_event(self, event_id: str):
        return copy.deepcopy(self._event(event_id))

    def t_calendar_find_free_slots(self, date: str, duration_minutes: int = 30, day_start: str = f"{WORK_START:02d}:00",
                                   day_end: str = f"{WORK_END:02d}:00"):
        """Free slots on a date based on the CALENDAR ONLY (it doesn't know things the user only said in chat).
        Events the user declined don't block time."""
        d0 = _parse(f"{date}T{day_start}:00{self.tz}")
        d1 = _parse(f"{date}T{day_end}:00{self.tz}")
        busy = sorted((_parse(e["start"]), _parse(e["end"])) for e in self.state["events"]
                      if self._busy(e) and _overlap(_parse(e["start"]), _parse(e["end"]), d0, d1))
        slots, cur, need = [], d0, dt.timedelta(minutes=int(duration_minutes))
        for bs, be in busy:
            if bs - cur >= need:
                slots.append({"start": cur.isoformat(), "end": bs.isoformat()})
            cur = max(cur, be)
        if d1 - cur >= need:
            slots.append({"start": cur.isoformat(), "end": d1.isoformat()})
        return slots

    def t_calendar_create_event(self, title: str, start: str, end: str, attendees: list | None = None,
                                location: str = "", notes: str = "", description: str = ""):
        a, b = _parse(start), _parse(end)
        if b <= a:
            raise ToolError("end must be after start")
        att = [self._person(x)["id"] for x in (attendees or [])]
        e = {"id": self._new_id("evt"), "title": title, "start": a.isoformat(), "end": b.isoformat(),
             "attendees": att, "location": location, "notes": notes, "description": description or notes,
             "organizer": self.state["user"]["id"], "my_response": "accepted", "status": "confirmed",
             "created_by": "assistant", "created_at": self.clock}
        self.state["events"].append(e)
        self._ev_ix[e["id"]] = e
        return e

    def t_calendar_update_event(self, event_id: str, title: str | None = None, start: str | None = None,
                                end: str | None = None, attendees: list | None = None, location: str | None = None,
                                description: str | None = None):
        e = self._event(event_id)
        if e.get("status") == "cancelled":
            raise ToolError("event is cancelled")
        if title is not None:
            e["title"] = title
        if start is not None:
            e["start"] = _parse(start).isoformat()
        if end is not None:
            e["end"] = _parse(end).isoformat()
        if _parse(e["end"]) <= _parse(e["start"]):
            raise ToolError("end must be after start")
        if attendees is not None:
            e["attendees"] = [self._person(x)["id"] for x in attendees]
        if location is not None:
            e["location"] = location
        if description is not None:
            e["description"] = description
        e["updated_at"] = self.clock
        return e

    def t_calendar_cancel_event(self, event_id: str):
        e = self._event(event_id)
        e["status"] = "cancelled"
        e["updated_at"] = self.clock
        return {"cancelled": True, "id": event_id}

    def t_calendar_respond(self, event_id: str, response: str, note: str = ""):
        """RSVP to an invite: accepted | tentative | declined. Declining frees the time
        without cancelling the meeting for everyone else."""
        if response not in RESPONSES:
            raise ToolError(f"response must be one of {RESPONSES}")
        e = self._event(event_id)
        if e.get("status") == "cancelled":
            raise ToolError("event is cancelled")
        e["my_response"] = response
        if note:
            e["response_note"] = note
        e["responded_at"] = self.clock
        return {"id": event_id, "my_response": response}

    # ------------------------------------------------------------ email ---
    @staticmethod
    def _mail_summary(m: dict) -> dict:
        return {"id": m["id"], "thread_id": m.get("thread_id", m["id"]), "folder": m.get("folder", "inbox"),
                "from": m.get("from"), "to": m.get("to", []), "subject": m.get("subject", ""),
                "snippet": _clip(m.get("body", "")), "ts": m["ts"], "unread": m.get("unread", False),
                "labels": m.get("labels", [])}

    def t_email_list(self, folder: str = "inbox", unread_only: bool = False, limit: int = 20,
                     before: str | None = None):
        mails = [m for m in self._visible(self.state["emails"])
                 if folder == "all" or m.get("folder", "inbox") == folder]
        if unread_only:
            mails = [m for m in mails if m.get("unread")]
        return [self._mail_summary(m) for m in reversed(self._page(mails, limit, before))]

    def t_email_search(self, query: str, limit: int = 20):
        q = _words(query)
        if not q:
            raise ToolError("query is empty")
        hits = []
        for m in self._visible(self.state["emails"]):
            hay = " ".join([m.get("subject", ""), m.get("body", ""), str(m.get("from", "")),
                            " ".join(m.get("to", []))]).lower()
            if all(w in hay for w in q):
                hits.append(m)
        return [self._mail_summary(m) for m in reversed(self._page(hits, limit, None))]

    def t_email_read_thread(self, thread_id: str):
        msgs = [m for m in self._visible(self.state["emails"]) if m.get("thread_id", m["id"]) == thread_id]
        if not msgs:
            raise ToolError(f"no email thread '{thread_id}'")
        for m in msgs:
            m["unread"] = False
        out = []
        for m in sorted(msgs, key=lambda m: m["ts"]):
            m = dict(m)
            if len(m.get("body", "")) > 4000:
                m["body"] = m["body"][:4000] + "\n[... clipped]"
            out.append(m)
        return out

    def _addr(self, x: str) -> str:
        x = str(x).strip()
        if "@" in x and not x.startswith("@"):
            return x
        p = self._person(x)
        if not p.get("email"):
            raise ToolError(f"{p['name']} has no email address")
        return p["email"]

    def t_email_send(self, to, subject: str, body: str, cc=None):
        to = [to] if isinstance(to, str) else list(to or [])
        if not to:
            raise ToolError("no recipient")
        if not str(body).strip():
            raise ToolError("empty body")
        mid = self._new_id("mail")
        m = {"id": mid, "thread_id": mid, "folder": "sent", "from": self.state["user"].get("email", ""),
             "to": [self._addr(x) for x in to], "cc": [self._addr(x) for x in (cc or [])],
             "subject": subject, "body": body, "ts": self.clock, "unread": False}
        self.state["emails"].append(m)
        return {"sent": True, "id": mid, "thread_id": mid}

    def t_email_reply(self, thread_id: str, body: str, reply_all: bool = False):
        msgs = sorted((m for m in self._visible(self.state["emails"]) if m.get("thread_id", m["id"]) == thread_id),
                      key=lambda m: m["ts"])
        if not msgs:
            raise ToolError(f"no email thread '{thread_id}'")
        if not str(body).strip():
            raise ToolError("empty body")
        me = self.state["user"].get("email", "")
        last = msgs[-1]
        to = list(last.get("to", [])) if last.get("from") == me else [last.get("from")]
        cc = [x for x in (last.get("to", []) + last.get("cc", [])) if x not in to and x != me] if reply_all else []
        subj = last.get("subject", "")
        m = {"id": self._new_id("mail"), "thread_id": thread_id, "folder": "sent", "from": me, "to": to, "cc": cc,
             "subject": subj if subj.lower().startswith("re:") else f"Re: {subj}", "body": body, "ts": self.clock,
             "unread": False}
        self.state["emails"].append(m)
        return {"sent": True, "id": m["id"], "thread_id": thread_id, "to": to}

    # ------------------------------------------------------------- docs ---
    def t_docs_search(self, query: str, limit: int = 10):
        q = _words(query)
        if not q:
            raise ToolError("query is empty")
        hits = []
        for d in self.state["docs"]:
            hay = (d.get("title", "") + " " + d.get("body", "")).lower()
            if all(w in hay for w in q):
                hits.append(d)
        hits.sort(key=lambda d: d.get("updated_at", ""), reverse=True)
        return [{"id": d["id"], "title": d["title"], "owner": d.get("owner"), "updated_at": d.get("updated_at"),
                 "snippet": _clip(d.get("body", ""))} for d in hits[:_cap(limit, 10)]]

    def t_docs_read(self, doc_id: str, offset: int = 0):
        d = self._doc(doc_id)
        body = d.get("body", "")
        o = max(0, int(offset or 0))
        nxt = o + DOC_PAGE if o + DOC_PAGE < len(body) else None
        return {"id": d["id"], "title": d["title"], "owner": d.get("owner"), "updated_at": d.get("updated_at"),
                "text": body[o:o + DOC_PAGE], "next_offset": nxt, "length": len(body),
                "comments": self._visible(d.get("comments", []))[-20:]}

    def t_docs_comment(self, doc_id: str, text: str):
        if not str(text).strip():
            raise ToolError("empty comment")
        c = {"id": self._new_id("cmt"), "author": self._me(), "text": text, "ts": self.clock}
        self._doc(doc_id)["comments"].append(c)
        return {"commented": True, **c}

    # ------------------------------------------------------------ tasks ---
    @staticmethod
    def _task_summary(t: dict) -> dict:
        return {k: t.get(k) for k in ("key", "title", "status", "assignee", "reporter", "due", "priority",
                                      "updated_at")}

    def t_tasks_list(self, assignee: str | None = None, status: str | None = None, project: str | None = None,
                     limit: int = 30):
        who = None
        if assignee:
            who = self._me() if assignee.lower() in ("me", "self") else self._person(assignee)["slack"]
        out = [t for t in self.state["tasks"]
               if (not who or t.get("assignee") == who) and (not status or t.get("status") == status)
               and (not project or t["key"].split("-")[0].lower() == project.lower())]
        out.sort(key=lambda t: t.get("updated_at", ""), reverse=True)
        return [self._task_summary(t) for t in out[:_cap(limit, 30)]]

    def t_tasks_get(self, key: str):
        t = copy.deepcopy(self._task(key))
        t["comments"] = self._visible(t.get("comments", []))[-20:]
        return t

    def t_tasks_search(self, query: str, limit: int = 20):
        q = _words(query)
        if not q:
            raise ToolError("query is empty")
        out = [t for t in self.state["tasks"]
               if all(w in (t["key"] + " " + t.get("title", "") + " " + t.get("description", "")).lower() for w in q)]
        out.sort(key=lambda t: t.get("updated_at", ""), reverse=True)
        return [self._task_summary(t) for t in out[:_cap(limit)]]

    def t_tasks_update(self, key: str, status: str | None = None, assignee: str | None = None,
                       due: str | None = None, comment: str | None = None):
        t = self._task(key)
        if status is not None:
            if status not in TASK_STATUSES:
                raise ToolError(f"status must be one of {TASK_STATUSES}")
            t["status"] = status
        if assignee is not None:
            t["assignee"] = self._person(assignee)["slack"]
        if due is not None:
            t["due"] = due
        if comment:
            t["comments"].append({"id": self._new_id("cmt"), "author": self._me(), "text": comment, "ts": self.clock})
        t["updated_at"] = self.clock
        return self._task_summary(t)

    # -------------------------------------------------------- reminders ---
    def t_reminders_create(self, text: str, due: str):
        if _parse(due) <= _parse(self.clock):
            raise ToolError(f"due must be in the future; it is now {self.clock}")
        r = {"id": self._new_id("rem"), "text": text, "due": _parse(due).isoformat(), "status": "open",
             "created_at": self.clock}
        self.state["reminders"].append(r)
        return r

    def t_reminders_list(self, status: str = "open"):
        return [r for r in self.state["reminders"] if status == "all" or r.get("status") == status][-MAX_LIST:]

    def t_reminders_complete(self, reminder_id: str):
        for r in self.state["reminders"]:
            if r["id"] == reminder_id:
                r["status"] = "done"
                return {"done": True}
        raise ToolError(f"no reminder '{reminder_id}'")

    # --------------------------------------------------------- contacts ---
    def t_contacts_lookup(self, query: str):
        q = str(query).lower().lstrip("@")
        if not q:
            raise ToolError("query is empty")
        out = [p for p in self.state["people"] if q in p["name"].lower() or q in p.get("slack", "").lower()
               or q in p.get("email", "").lower()]
        return [{k: p.get(k) for k in ("id", "name", "slack", "email", "phone", "title", "team")}
                for p in out[:MAX_LIST]]


def _s(t, **kw):
    return {"type": t, **kw}


_STR, _INT, _BOOL = _s("string"), _s("integer"), _s("boolean")
_LIST = _s("array", items=_s("string"))

# Descriptions are deliberately terse: every schema you send counts against the 8,000
# input cap, so a tool list is part of your context budget.
TOOLS: dict[str, dict] = {
    # slack
    "slack_list_channels": {"kind": "read", "description": "List Slack channels.", "params": {}, "required": []},
    "slack_read_messages": {"kind": "read", "description": "Recent messages in '#channel' or DMs with '@handle'. "
                            "Max 50; page back with before=<ts or oldest id>. Threads show reply_count.",
                            "params": {"conversation": _STR, "limit": _INT, "before": _STR}, "required": ["conversation"]},
    "slack_read_thread": {"kind": "read", "description": "A message and its thread replies.",
                          "params": {"message_id": _STR}, "required": ["message_id"]},
    "slack_search_messages": {"kind": "read", "description": "Search Slack (all words must match). "
                              "Optional from_user='@handle', conversation='#ch'|'@handle'.",
                              "params": {"query": _STR, "limit": _INT, "from_user": _STR, "conversation": _STR},
                              "required": ["query"]},
    "slack_send_message": {"kind": "write", "description": "Send NOW to '#channel' or '@handle' as the user. "
                           "thread_id replies in a thread.",
                           "params": {"to": _STR, "text": _STR, "thread_id": _STR}, "required": ["to", "text"]},
    "slack_schedule_message": {"kind": "write", "description": "Schedule a Slack message (send_at ISO 8601).",
                               "params": {"to": _STR, "text": _STR, "send_at": _STR},
                               "required": ["to", "text", "send_at"]},
    # calendar
    "calendar_list_events": {"kind": "read", "description": "Events between start and end (ISO 8601), max 50. "
                             "Descriptions clipped; use calendar_get_event.",
                             "params": {"start": _STR, "end": _STR, "limit": _INT}, "required": ["start", "end"]},
    "calendar_get_event": {"kind": "read", "description": "One event in full.",
                           "params": {"event_id": _STR}, "required": ["event_id"]},
    "calendar_find_free_slots": {"kind": "read",
                                 "description": "Free slots on date (YYYY-MM-DD) from the calendar only. "
                                                "day_start/day_end HH:MM.",
                                 "params": {"date": _STR, "duration_minutes": _INT,
                                            "day_start": _STR, "day_end": _STR}, "required": ["date"]},
    "calendar_create_event": {"kind": "write", "description": "Create an event. attendees: handles or person ids.",
                              "params": {"title": _STR, "start": _STR, "end": _STR, "attendees": _LIST,
                                         "location": _STR, "description": _STR}, "required": ["title", "start", "end"]},
    "calendar_update_event": {"kind": "write", "description": "Change an existing event.",
                              "params": {"event_id": _STR, "title": _STR, "start": _STR, "end": _STR,
                                         "attendees": _LIST, "location": _STR, "description": _STR},
                              "required": ["event_id"]},
    "calendar_cancel_event": {"kind": "write", "description": "Cancel an event for everyone.",
                              "params": {"event_id": _STR}, "required": ["event_id"]},
    "calendar_respond": {"kind": "write", "description": "RSVP to an invite: accepted|tentative|declined.",
                         "params": {"event_id": _STR, "response": _STR, "note": _STR},
                         "required": ["event_id", "response"]},
    # email
    "email_list": {"kind": "read", "description": "Newest emails in folder (inbox|sent|all), max 50; "
                   "page with before=<ts or oldest id>.",
                   "params": {"folder": _STR, "unread_only": _BOOL, "limit": _INT, "before": _STR}, "required": []},
    "email_search": {"kind": "read", "description": "Search email subject/body/addresses (all words must match).",
                     "params": {"query": _STR, "limit": _INT}, "required": ["query"]},
    "email_read_thread": {"kind": "read", "description": "Full messages of an email thread.",
                          "params": {"thread_id": _STR}, "required": ["thread_id"]},
    "email_send": {"kind": "write", "description": "Send a new email. to/cc: addresses or handles.",
                   "params": {"to": _LIST, "subject": _STR, "body": _STR, "cc": _LIST},
                   "required": ["to", "subject", "body"]},
    "email_reply": {"kind": "write", "description": "Reply to the latest message in a thread.",
                    "params": {"thread_id": _STR, "body": _STR, "reply_all": _BOOL}, "required": ["thread_id", "body"]},
    # docs
    "docs_search": {"kind": "read", "description": "Search docs by title/body (all words must match).",
                    "params": {"query": _STR, "limit": _INT}, "required": ["query"]},
    "docs_read": {"kind": "read", "description": "Read a doc, 6000 chars per call; continue with next_offset.",
                  "params": {"doc_id": _STR, "offset": _INT}, "required": ["doc_id"]},
    "docs_comment": {"kind": "write", "description": "Comment on a doc as the user.",
                     "params": {"doc_id": _STR, "text": _STR}, "required": ["doc_id", "text"]},
    # tasks
    "tasks_list": {"kind": "read", "description": "Tracker tickets. Filters: assignee ('me'|'@handle'), "
                   "status, project (e.g. PAY).",
                   "params": {"assignee": _STR, "status": _STR, "project": _STR, "limit": _INT}, "required": []},
    "tasks_get": {"kind": "read", "description": "One ticket with its comments.",
                  "params": {"key": _STR}, "required": ["key"]},
    "tasks_search": {"kind": "read", "description": "Search tickets (all words must match).",
                     "params": {"query": _STR, "limit": _INT}, "required": ["query"]},
    "tasks_update": {"kind": "write", "description": "Update a ticket: status (todo|in_progress|in_review|"
                     "blocked|done|wont_do), assignee, due, comment.",
                     "params": {"key": _STR, "status": _STR, "assignee": _STR, "due": _STR, "comment": _STR},
                     "required": ["key"]},
    # reminders + contacts
    "reminders_create": {"kind": "write", "description": "Create a reminder for the user at due (ISO 8601).",
                         "params": {"text": _STR, "due": _STR}, "required": ["text", "due"]},
    "reminders_list": {"kind": "read", "description": "List reminders. status: open|done|all.",
                       "params": {"status": _STR}, "required": []},
    "reminders_complete": {"kind": "write", "description": "Mark a reminder done.",
                           "params": {"reminder_id": _STR}, "required": ["reminder_id"]},
    "contacts_lookup": {"kind": "read", "description": "Find people by name, handle or email.",
                        "params": {"query": _STR}, "required": ["query"]},
}

APPS = ("slack", "calendar", "email", "docs", "tasks", "reminders", "contacts")


def json_schema(name: str) -> dict:
    t = TOOLS[name]
    return {"type": "object", "properties": t["params"], "required": t["required"], "additionalProperties": False}


def _select(apps: str | None) -> list[str]:
    """apps='slack,calendar' -> only those tools (lets an agent keep its schema budget small)."""
    if not apps:
        return list(TOOLS)
    want = {a.strip().lower() for a in apps.split(",") if a.strip()}
    return [n for n in TOOLS if n.split("_", 1)[0] in want]


def openai_tools(apps: str | None = None) -> list[dict]:
    return [{"type": "function", "function": {"name": n, "description": TOOLS[n]["description"],
                                              "parameters": json_schema(n)}} for n in _select(apps)]


def mcp_tools(apps: str | None = None) -> list[dict]:
    return [{"name": n, "description": TOOLS[n]["description"], "inputSchema": json_schema(n)} for n in _select(apps)]
