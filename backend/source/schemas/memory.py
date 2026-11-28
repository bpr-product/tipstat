"""Contracts for extracted facts and the deterministic gate every model proposal must pass before it is stored."""
import json
import re

from backend.source.utils.text import norm
from backend.source.utils.time import valid_date, valid_datetime

KINDS = {"STATE", "PREFERENCE", "EVENT", "PLAN", "TODO", "RULE", "DERIVED"}
ATTRIB = {"user", "other"}


PASTE_MARK = re.compile(r"^\s*(>|-{3,}\s*(forwarded|original)|begin forwarded|forwarded message|(from|to|cc|subject|sent|date):\s|on .{5,80} wrote:)",
                        re.IGNORECASE)
HIST_MARK = re.compile(r"\b(used to|formerly|previously|no longer|until|back then|at that time|former|ex-|had been|was|were|stopped|left|quit)\b|\bin (19|20)\d\d\b", re.IGNORECASE)
HHMM = re.compile(r"^([01]?\d|2[0-3]):[0-5]\d$")


def pasted_text(user_text: str) -> str:
    """Normalized text of anything the user pasted or forwarded: from the first mail/quote marker to the end, plus any
    quoted ('>') or fenced lines. Facts quoted from here are about that content, never the user's own."""
    lines, out, fence, on = (user_text or "").splitlines(), [], False, False
    for ln in lines:
        if ln.strip().startswith("```"):
            fence = not fence
            continue
        on = on or bool(PASTE_MARK.match(ln))
        if on or fence or ln.lstrip().startswith(">"):
            out.append(ln)
    return norm(" ".join(out))


def _constraint(x) -> dict | None:
    """A quiet-hours rule the backend can enforce: {who, from, until} as HH:MM. Anything else is dropped."""
    if not isinstance(x, dict) or not isinstance(x.get("who"), str) or not x["who"].strip():
        return None
    frm, until = str(x.get("from", "")).strip(), str(x.get("until") or "07:00").strip()
    if not (HHMM.match(frm) and HHMM.match(until)):
        return None
    return {"who": x["who"].strip().lstrip("@").lower()[:60], "from": frm.zfill(5), "until": until.zfill(5)}


QUIET = re.compile(r"(?:never|don'?t|do not)\s+(?:message|contact|ping|text|call|dm|email|slack)\s+@?([a-z][\w.]*(?:\s[a-z][\w.]*)?)"
                   r"\s+(?:after|past)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", re.IGNORECASE)


def _constraint_from_text(quote: str) -> dict | None:
    """Fallback when the model did not return a constraint: parse the plain phrasing 'never message <name> after <time>'."""
    m = QUIET.search(quote or "")
    if not m:
        return None
    hour, minute, ap = int(m.group(2)), int(m.group(3) or 0), (m.group(4) or "").lower()
    if ap == "pm" and hour < 12:
        hour += 12
    elif ap == "am" and hour == 12:
        hour = 0
    elif not ap and hour < 13:
        return None  # "after 9" with no am/pm is ambiguous: do not guess
    return _constraint({"who": m.group(1), "from": f"{hour:02d}:{minute:02d}", "until": "07:00"}) if hour < 24 else None


def validate_candidate(f, haystack: str, related_ids: set, force_other: bool = False, pasted: str = "") -> dict | None:
    """Text fields present, enums known, quote really in the source text, and a replace target only if it was
    one of the related facts we supplied. Others' text can never become a user fact (force_other)."""
    if not isinstance(f, dict):
        return None
    c = {k: f.get(k) for k in ("subject", "relation", "value", "quote")}
    if not all(isinstance(v, str) and v.strip() for v in c.values()):
        return None
    c = {k: v.strip() for k, v in c.items()}
    if norm(c["quote"]) not in haystack:
        return None
    kind = str(f.get("kind", "")).upper()
    attribution = "other" if force_other or (pasted and norm(c["quote"]) in pasted) else str(f.get("attribution", "")).lower()
    if kind not in KINDS or attribution not in ATTRIB:
        return None
    parents = sorted({int(x) for x in re.findall(r"\d+", str(f.get("derived_from") or "")) if int(x) in related_ids})
    if kind == "DERIVED" and not parents:
        return None  # an inference must name the stored facts it rests on; otherwise it is unsupported
    if parents:
        kind = "DERIVED"
    card = str(f.get("cardinality", "")).upper()
    m = re.search(r"\d+", str(f.get("replaces") or ""))
    rid = int(m.group(0)) if m and int(m.group(0)) in related_ids else None
    c.update(kind=kind, attribution=attribution, derived_from=parents, when=valid_date(f.get("when")) if f.get("when") else None,
             when_ts=valid_datetime(f.get("when")) if f.get("when") else None,
             sensitive=f.get("sensitive") is True or str(f.get("sensitive")).lower() == "true",
             historical=(f.get("historical") is True or str(f.get("historical")).lower() == "true") and bool(HIST_MARK.search(c["quote"])),  # the words must say it
             cardinality=card if card in ("EXCLUSIVE", "MULTI") else "MULTI", replaces=rid)
    constraint = (_constraint(f.get("constraint")) or _constraint_from_text(c["quote"])) if kind == "RULE" and attribution == "user" else None
    if constraint:  # only the user's own rules can carry an enforceable constraint
        c["data"] = json.dumps({"constraint": constraint})
    return c
