"""Deterministic handling of structured app events (no model call).

Each supported op maps to a stable entity id (e.g. "event:evt_dentist") and a field delta. The current
state of an entity is the merge of its earlier state and the delta; every change becomes a new fact
version that supersedes the previous one, so history is kept and the current value always wins.
Anything not listed here (Slack/email/doc text, comments) is free text and goes through extraction.
"""
from typing import NamedTuple


class Spec(NamedTuple):
    ext_id: str
    relation: str
    creates: bool      # False: only updates an entity we already know (a backlog item we never saw is skipped)
    delta: dict


def _clean(d: dict, drop=()) -> dict:
    return {k: v for k, v in d.items() if k not in drop and v not in (None, "", [], {})}


def spec(op: str, d: dict) -> Spec | None:
    if op in ("add_event", "update_event", "cancel_event", "rsvp_event") and d.get("id"):
        delta = _clean(d, drop=("id",))
        if op == "cancel_event":
            delta = {"status": "cancelled"}
        elif op == "rsvp_event":
            delta = _clean({"my_rsvp": d.get("response"), "rsvp_note": d.get("note")})
        return Spec(f"event:{d['id']}", "calendar event", op == "add_event", delta)
    if op in ("add_task", "update_task") and d.get("key"):
        return Spec(f"task:{d['key']}", "ticket", op == "add_task", _clean(d))
    if op in ("add_reminder", "complete_reminder") and d.get("id"):
        delta = {"status": "completed"} if op == "complete_reminder" else _clean(d, drop=("id",))
        return Spec(f"reminder:{d['id']}", "reminder", op == "add_reminder", delta)
    if op == "add_person" and (d.get("id") or d.get("name")):
        return Spec(f"person:{d.get('id') or d['name']}", "contact", True, _clean(d))
    return None


def render(data: dict) -> str:
    return "; ".join(f"{k}: {', '.join(map(str, v)) if isinstance(v, list) else v}" for k, v in data.items())


def subject(ext_id: str, data: dict) -> str:
    return str(data.get("title") or data.get("name") or data.get("summary") or data.get("text") or ext_id)


def render_event(m: dict) -> str:
    """One searchable, quotable line for any app event."""
    d = m.get("data") or {}
    body = "; ".join(f"{k}: {v}" for k, v in d.items() if v not in (None, "", [], {}))
    return f"{m.get('op', 'event')} by {m.get('source', '?')} - {body}"
