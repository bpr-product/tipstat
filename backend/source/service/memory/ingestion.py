"""Remember: route one incoming message, store it as a source, and turn it into facts.

Chat and free-text app events: one extraction call. Structured app events: applied in code, no model call.
The model proposes; deterministic code validates (schemas), reconciles with what is stored (reconciliation: replace
targets, paraphrase duplicates, conflicts) and writes. Embeddings are indexed afterwards and separately: a failed
index never loses a fact.
"""
import json
import sys

from backend.source.dal.repositories.memory_repository import MemoryRepository
from backend.source.dal.repositories.search_repository import SearchRepository
from backend.source.schemas.memory import pasted_text, validate_candidate
from backend.source.service.memory import app_events
from backend.source.service.memory.extraction_pipeline import propose_facts
from backend.source.service.memory.forgetting import handle_forget_request
from backend.source.service.memory.intent import may_ask_to_forget
from backend.source.service.memory.reconciliation import reconcile
from backend.source.service.memory.write_context import related_previous_facts
from backend.source.utils.text import norm
from backend.source.utils.time import valid_date

ACK = {"ok", "okay", "k", "thanks", "thank", "thx", "ty", "yes", "no", "yep", "nope", "hi", "hello", "hey", "bye", "cool",
       "nice", "great", "lol", "hmm", "sure", "fine", "good", "morning", "night", "please", "you", "welcome", "got", "it"}
FREE_TEXT_OPS = {"add_message", "add_email", "add_doc", "update_doc", "add_doc_comment", "add_task_comment"}


def ingest(repo: MemoryRepository, search: SearchRepository, llm, embedder, message: dict) -> None:
    _ingest(repo, search, llm, embedder, message)
    _index_message(repo, embedder, message["id"])


def _ingest(repo, search, llm, embedder, message: dict) -> None:
    mid, ts = message["id"], message.get("ts", "")
    if message.get("type") == "chat":
        turns = message.get("turns") or []
        user_text = "\n".join(t["content"] for t in turns if t.get("role") == "user")
        convo = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in turns)
        if not repo.put_source(mid, ts, "chat", "user", convo):
            return  # replay of an id we already ingested: no duplicate facts
        _extract(repo, search, llm, embedder, mid, ts, "a chat between the user and their assistant", convo, user_text, user_text, False)
        if may_ask_to_forget(user_text):  # only the user's own words can ask; the model decides, code verifies
            handle_forget_request(repo, search, llm, embedder, mid, ts, user_text)
        return
    text, op, origin = app_events.render_event(message), message.get("op", ""), message.get("source", "inbound")
    if not repo.put_source(mid, ts, "event", origin, text):
        return
    sp = app_events.spec(op, message.get("data") or {})
    if sp is not None and _apply_structured(repo, message, sp):
        return repo.set_extraction(mid, "structured")
    if op in FREE_TEXT_OPS:
        who = {"user": "written by the user", "assistant": "by the assistant"}.get(origin, "written by someone else, not the user")
        _extract(repo, search, llm, embedder, mid, ts, f"an app event ({op}) {who}", text, text, text, origin != "user")
        if origin == "user" and may_ask_to_forget(text):  # inbound or assistant events can never trigger erasure
            handle_forget_request(repo, search, llm, embedder, mid, ts, text)
    else:
        repo.set_extraction(mid, "skipped")


def _apply_structured(repo: MemoryRepository, m: dict, sp: app_events.Spec) -> bool:
    """Calendar/task/reminder/contact ops: new state = old state + delta, stored as a new fact version."""
    prior = repo.active_by_ext(sp.ext_id)
    if prior is None and not sp.creates:
        return False  # an update to something we never saw: only the source is kept
    merged = {**(json.loads(prior["data"]) if prior else {}), **sp.delta}
    fact = {"kind": {"calendar event": "EVENT", "reminder": "TODO"}.get(sp.relation, "STATE"),
            "subject": app_events.subject(sp.ext_id, merged), "relation": sp.relation, "value": app_events.render(merged),
            "attribution": "other" if m.get("source") == "inbound" else "user",
            "when": valid_date(merged.get("start")) if merged.get("start") else None,
            "quote": app_events.render_event(m), "cardinality": "EXCLUSIVE", "ext_id": sp.ext_id, "data": json.dumps(merged)}
    repo.add_fact(fact, m["id"], m.get("ts", ""), replaces=prior["id"] if prior else None)
    return True


def _extract(repo, search, llm, embedder, mid, ts, header, body, quote_text, search_text, force_other) -> None:
    if _trivial(search_text):
        return repo.set_extraction(mid, "skipped")
    related, related_text = related_previous_facts(search, embedder, search_text)
    try:
        facts = propose_facts(llm, ts, header, body, related_text)
    except Exception as e:  # model/network/format failure: source stays searchable, fact not saved
        print(f"extraction failed for {mid}: {type(e).__name__}: {e}", file=sys.stderr)
        return repo.set_extraction(mid, "failed")
    haystack, related_ids, pasted = norm(quote_text), {f["id"] for f in related}, pasted_text(quote_text)
    for f in facts:
        clean = validate_candidate(f, haystack, related_ids, force_other, pasted)
        if not clean:
            continue
        clean, duplicate_of = reconcile(repo, clean, related)
        if duplicate_of:  # a paraphrase of something already stored: more evidence, not a second fact
            repo.add_evidence(duplicate_of, mid, clean["quote"])
            if clean.get("sensitive"):  # a later "keep it private" must not be lost by merging
                repo.escalate_sensitive(duplicate_of)
            if clean.get("historical"):  # "I used to ..." about something stored as current closes it
                repo.close_fact(duplicate_of, clean.get("when") or ts[:10])
            continue
        repo.add_fact(clean, mid, ts, replaces=clean.pop("replaces"))
    repo.set_extraction(mid, "ok")


def _trivial(text: str) -> bool:
    """Nothing to remember: empty, or only greetings/acknowledgements. A short message with any real content is not trivial."""
    ws = norm(text).split()
    return not ws or all(w in ACK for w in ws)


def _index_message(repo: MemoryRepository, embedder, mid: str) -> None:
    """Embed the message's source text and its facts (one batch). Optional: without a model, or if it fails, lexical
    search still covers everything and the facts are untouched."""
    if not embedder:
        return
    src = repo.source(mid)
    refs = {f"f:{f['id']}": f"{f['subject']} {f['relation']} {f['value']} {f['quote']}" for f in repo.facts_from_message(mid)}
    if src and src["text"]:
        refs[f"s:{mid}"] = src["text"][:500]
    vecs = embedder.embed(list(refs.values())) if refs else None
    if vecs:
        for ref, vec in zip(refs, vecs):
            repo.put_embedding(ref, vec)
    elif refs and getattr(embedder, "available", False):
        print(f"embedding index skipped for {mid}", file=sys.stderr)
