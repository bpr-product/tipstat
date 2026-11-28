"""What the assistant remembers that bears on a job: standing rules, relevant current facts, and the outdated
versions (labelled so they are never acted on). Reuses Ask's candidate discovery; no new retrieval logic."""
from datetime import date, timedelta

from backend.source.dal.repositories.memory_repository import MemoryRepository
from backend.source.dal.repositories.search_repository import SearchRepository
from backend.source.service.memory.hybrid_retrieval import retrieve
from backend.source.utils.text import clip, user_view


def _line(f, label: str) -> str:
    when = f" | date {f['when_ts'] or f['when_date']}" if f["when_date"] else ""
    if f["sensitive"]:
        label += ", PRIVATE: never put in a message to others"
    return (f"- ({label}; {f['attribution']}, {f['kind'].lower()}, noted {(f['created_ts'] or '')[:10]}) "
            f"{f['subject']} | {f['relation']} | {f['value']}{when} | said: \"{f['quote']}\"")


def build(repo: MemoryRepository, search: SearchRepository, llm, embedder, request: str, max_tokens: int = 1800):
    """Returns (memory text, trace entries, model calls used)."""
    facts, hist, srcs, trace = retrieve(search, llm, request, repo.has_data(), embedder)
    calls = 1 if any(t["step"] == "expand" for t in trace) else 0
    rules = repo.active_by_kind("RULE", "user")  # a rule written by someone else is never a standing rule
    today = repo.get_meta("now", "")[:10]
    try:
        horizon = (date.fromisoformat(today) + timedelta(days=30)).isoformat()
        upcoming = repo.upcoming(today, horizon)
    except ValueError:  # no message seen yet, so no "now"
        upcoming = []
    seen = {r["id"] for r in rules}
    lines = [_line(r, "STANDING RULE") for r in rules]
    lines += [_line(u, "UPCOMING - may not be on the calendar") for u in upcoming if u["id"] not in seen]
    seen |= {u["id"] for u in upcoming}
    lines += [_line(f, "CURRENT") for f in facts if f["id"] not in seen]
    lines += [_line(h, "OUTDATED - replaced " + (h["ended_ts"] or "?")[:10] + ", do not act on it") for h in hist]
    lines += [f"- (source {s['message_id']}, {s['origin']}, {s['ts']}) {(user_view(s['text']) if s['kind'] == 'chat' else s['text'])[:300]}" for s in srcs[:3]
              if s["kind"] != "event" or not any(s["message_id"] == f["first_message_id"] for f in facts)]
    trace.append({"step": "act_memory", "rules": len(rules), "facts": len(facts), "outdated": len(hist), "upcoming": len(upcoming)})
    note = repo.people_note()
    return clip(("\n".join(([note] if note else []) + lines)) or "nothing relevant stored", max_tokens), trace, calls
