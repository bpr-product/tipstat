"""Evaluation runner.

Sends a persona's messages to a participant's service one at a time, in order. At fixed
points in the stream it asks QUESTIONS and gives JOBS (actions, run against the mock
Slack/Calendar/Email/Docs/Tasks/Reminders/Contacts world as it is at that moment).

  # questions only
  python -m harness.run --system http://localhost:8000 --data data/dev/persona_a --judge offline --out runs/me

  # questions + actions with an embedded mock world
  python -m harness.run --system http://localhost:8000 --data data/dev/persona_a --world auto --judge offline --out runs/me

See docs/SPEC.md for the HTTP contract.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import socket
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

from .checks import evaluate
from .common import approx_tokens, http_json, read_json, read_jsonl, write_json, write_jsonl
from .checks import has_positive, validate_checks
from .judge import make_action_judge, make_judge
from .report import summarize, summary_markdown


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def _message_ids(evidence) -> list[str]:
    return [str(e["message_id"]) for e in evidence or [] if isinstance(e, dict) and e.get("message_id")]


def _clip(x, n=400):
    s = json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x
    return s if len(s) <= n else s[:n] + "..."


class Proxy:
    def __init__(self, url):
        self.url = url.rstrip("/") if url else None

    def mark(self, budget: int = 0, token_budget: int = 0, phase: str = "", label: str = ""):
        """Open a new budget window. Every model call until the next mark counts against it."""
        if not self.url:
            return None
        try:
            return http_json(f"{self.url}/calls/mark", {"budget": budget, "token_budget": token_budget,
                                                        "phase": phase, "label": label}, timeout=5).get("mark")
        except Exception:
            return None

    def calls_since(self, mark):
        if not self.url or mark is None:
            return []
        try:
            return http_json(f"{self.url}/calls?since={mark}", timeout=5).get("calls", [])
        except Exception:
            return []


class WorldClient:
    def __init__(self, url: str, public: str | None):
        self.url, self.public = url.rstrip("/"), (public or url).rstrip("/")
        tok = os.environ.get("WORLD_ADMIN_TOKEN", "")
        self.h = {"X-Admin-Token": tok} if tok else {}

    def admin(self, path, body=None):
        return http_json(f"{self.url}/admin/{path}", body, headers=self.h, timeout=60)

    def lock(self, locked: bool = True):
        """Outside /act the tools refuse calls, so the world can't stand in for memory."""
        try:
            self.admin("lock", {"locked": locked})
        except Exception:
            pass


def start_embedded_world() -> str:
    from world.server import serve
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = serve(port, "127.0.0.1")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}/t/eval"


def ask(system: str, pid: str, q: dict, timeout: float, proxy: Proxy | None,
        call_budget: int = 0, token_budget: int = 0) -> dict:
    t0 = time.time()
    rec = {"id": q["id"], "kind": "question", "question": q["question"], "as_of": q["as_of"]}
    mark = proxy.mark(call_budget, token_budget, "answer", q["id"]) if proxy else None
    try:
        out = http_json(f"{system}/answer", {"question": q["question"]}, timeout=timeout)
        rec.update(answer=str(out.get("answer", "")), evidence=(out.get("evidence") or [])[:20],
                   trace=(out.get("trace") or [])[:50], error=None)
    except Exception as e:
        rec.update(answer="", evidence=[], trace=[], error=f"{type(e).__name__}: {e}")
    rec["latency_s"] = round(time.time() - t0, 3)
    rec["evidence_message_ids"] = _message_ids(rec["evidence"])
    if proxy:
        rec["llm_calls"] = proxy.calls_since(mark)
    return rec


def act(system: str, pid: str, a: dict, world: WorldClient, data_dir: Path, timeout: float,
        proxy: Proxy | None, call_budget: int = 0, token_budget: int = 0) -> dict:
    """Load the world as of the latest message, let the system act, diff the world."""
    world.admin("load", {"persona_dir": str(data_dir.resolve()), "date": a["as_of"][:10], "clock": a["as_of"],
                         "until": a["as_of"]})
    before = world.admin("state")
    wmark = world.admin("mark")["mark"]
    pmark = proxy.mark(call_budget, token_budget, "act", a["id"]) if proxy else None
    rec = {"id": a["id"], "kind": "action", "request": a["request"], "as_of": a["as_of"]}
    t0 = time.time()
    try:
        out = http_json(f"{system}/act", {"request": a["request"],
                                          "mcp_url": f"{world.public}/mcp", "tools_url": f"{world.public}/tools"},
                        timeout=timeout)
        rec.update(reply=str(out.get("reply", out.get("answer", ""))), trace=(out.get("trace") or [])[:60], error=None)
    except Exception as e:
        rec.update(reply="", trace=[], error=f"{type(e).__name__}: {e}")
    rec["latency_s"] = round(time.time() - t0, 3)
    world.lock(True)
    audit = world.admin(f"audit?since={wmark}")["audit"]
    after = world.admin("state")
    rec["tool_calls"] = [{"tool": x["tool"], "args": x["args"], "ok": x.get("ok"),
                          "result": _clip(x.get("result", x.get("error")), 300)} for x in audit]
    rec["checks"] = evaluate(a.get("checks", []), before, after, audit, rec.get("reply", ""))
    if proxy:
        rec["llm_calls"] = proxy.calls_since(pmark)
    return rec


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", required=True, help="Base URL of the submission, e.g. http://localhost:8000")
    p.add_argument("--data", required=True, help="Persona folder (persona.json, stream.jsonl, questions.jsonl, "
                                                 "optional actions.jsonl + world/)")
    p.add_argument("--out", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--judge", choices=["llm", "offline", "none"], default="llm")
    p.add_argument("--votes", type=int, default=3)
    p.add_argument("--world", default=None, help="Mock world base URL (e.g. http://host:8800/t/alice) or 'auto' "
                                                 "to start one in-process. Needed for action tasks.")
    p.add_argument("--world-public", default=None, help="World URL as seen by the submission (e.g. from inside Docker)")
    p.add_argument("--no-actions", action="store_true", help="Skip action tasks")
    p.add_argument("--ingest-budget", type=float, default=20 * 60, help="Total seconds for all /message calls")
    p.add_argument("--wall-clock", type=float, default=30 * 60,
                   help="Hard total budget for one submission. When it runs out we stop and score "
                        "what finished, so one stuck system cannot hold up the whole batch.")
    p.add_argument("--message-timeout", type=float, default=30)
    p.add_argument("--answer-timeout", type=float, default=30)
    p.add_argument("--act-timeout", type=float, default=90)
    p.add_argument("--concurrency", type=int, default=1,
                   help="Questions sent at once when there is no --proxy (1 = like the evaluation)")
    p.add_argument("--message-calls", type=int, default=2, help="Model calls allowed per /message (0 = unlimited)")
    p.add_argument("--message-tokens", type=int, default=10000, help="Input tokens allowed per /message")
    p.add_argument("--answer-calls", type=int, default=2, help="Model calls allowed per /answer (0 = unlimited)")
    p.add_argument("--answer-tokens", type=int, default=4000,
                   help="Input tokens allowed per /answer across all its calls: the history is many times "
                        "larger, so it cannot be sent with the question")
    p.add_argument("--act-calls", type=int, default=12, help="Model calls allowed per /act")
    p.add_argument("--act-tokens", type=int, default=60000, help="Input tokens allowed per /act")
    p.add_argument("--proxy", default=None, help="LLM proxy URL (for usage + per-question LLM call capture)")
    p.add_argument("--trace-llm", action="store_true", help="Ask questions one at a time and attach the proxy's LLM calls")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args(argv)

    system, data, out = a.system.rstrip("/"), Path(a.data), Path(a.out)
    name = a.name or out.name
    persona = read_json(data / "persona.json")
    pid = persona["persona_id"]
    stream = read_jsonl(data / "stream.jsonl")
    questions = read_jsonl(data / "questions.jsonl")
    actions = [] if a.no_actions or not (data / "actions.jsonl").exists() else read_jsonl(data / "actions.jsonl")
    if a.limit:
        questions, actions = questions[: a.limit], actions[: a.limit]
    # "now" for an item is the time of the message it follows; only the grader and the mock
    # world use it - the system under test sees timestamps on its messages, nothing else
    when = {m["id"]: m["ts"] for m in stream}
    for it in questions + actions:
        it["as_of"] = when[it["after"]]
    proxy = Proxy(a.proxy) if a.proxy else None

    world = None
    if actions:
        if not a.world:
            log("actions.jsonl found but no --world given: skipping action tasks (use --world auto)")
            actions = []
        else:
            url = start_embedded_world() if a.world == "auto" else a.world
            world = WorldClient(url, a.world_public or (url if a.world == "auto" else None))
            log(f"mock world at {world.url} (submission sees {world.public})")

    schedule: dict = defaultdict(list)
    for item in questions + actions:
        schedule[item["after"]].append(item)

    for t in actions:
        validate_checks(t.get("checks"), where=f"{t['id']} in {data}")
        if not has_positive(t.get("checks")):
            raise SystemExit(f"{t['id']}: only guard checks, so doing nothing would score 1.0. "
                             f"Add at least one positive check.")
    msg_tokens = {m["id"]: approx_tokens(json.dumps(m, ensure_ascii=False)) for m in stream}
    log(f"{name}: {pid}, {len(stream)} messages, {len(questions)} questions, {len(actions)} actions")
    log(f"history: {sum(msg_tokens.values()):,} tokens, largest message {max(msg_tokens.values()):,}")
    if proxy and a.answer_calls:
        log(f"budgets: message {a.message_calls} calls / {a.message_tokens:,} tokens, answer {a.answer_calls} calls / "
            f"{a.answer_tokens:,} tokens, act {a.act_calls} calls / {a.act_tokens:,} tokens (everything runs "
            f"serially so budgets can be enforced)")
    try:
        http_json(f"{system}/health", timeout=10)
    except Exception as e:
        log(f"Health check failed: {e}")
        return 2
    http_json(f"{system}/reset", {}, timeout=120)   # first, so stores created at reset exist
    # Anything can answer 200 on /health. Prove this really is a submission before we
    # spend 30 minutes scoring it - a port collision otherwise produces a silent zero.
    try:
        probe = http_json(f"{system}/answer", {"question": "harness connectivity probe"},
                          timeout=max(15.0, a.answer_timeout))
        if not isinstance(probe, dict) or "answer" not in probe:
            log(f"ABORT: {system} responded to /answer without an 'answer' field "
                f"(keys: {sorted(probe)[:8] if isinstance(probe, dict) else type(probe).__name__}). "
                f"Wrong service on this port?")
            return 2
    except Exception as e:
        # A 500 here is a bug in handling a question with an empty memory: their /answer exists,
        # so warn and carry on rather than scoring the whole run zero.
        if getattr(e, "code", None) != 500:
            log(f"ABORT: /answer probe failed: {type(e).__name__}: {e}")
            return 2
        log("WARNING: /answer returned 500 on the connectivity probe (empty memory?); continuing")
    if proxy:
        try:
            http_json(f"{proxy.url}/usage/reset", {}, timeout=10)
        except Exception as e:
            log(f"(could not reset proxy usage: {e})")
    if world:
        world.lock(True)

    records: list[dict] = []
    ingest_used, ingest_log, budget_blown = 0.0, [], False
    deadline = time.time() + a.wall_clock
    out_of_time = {"hit": False}

    def time_left() -> float:
        return deadline - time.time()

    def skip(item: dict) -> dict:
        """Record an item we never got to, so it scores 0 rather than vanishing."""
        is_action = item.get("type") == "action"
        base = {"id": item["id"], "kind": "action" if is_action else "question",
                "as_of": item.get("as_of"), "latency_s": 0.0,
                "error": "wall clock exceeded: not attempted"}
        if is_action:
            base.update(request=item.get("request", ""), reply="", trace=[], tool_calls=[],
                        checks={"hard_score": 0.0, "harmful": False, "results": [], "writes": []})
        else:
            base.update(question=item.get("question", ""), answer="", evidence=[],
                        evidence_message_ids=[], trace=[])
        return base

    def checkpoint(slot):
        items = schedule.get(slot, [])
        qs = [i for i in items if i.get("type") != "action"]
        acts = [i for i in items if i.get("type") == "action"]
        if time_left() <= 0:
            if items and not out_of_time["hit"]:
                out_of_time["hit"] = True
                log("WALL CLOCK EXHAUSTED: remaining questions and actions are scored as missed")
            records.extend(skip(i) for i in items)
            return
        if qs:
            # A call budget needs one open window at a time, so questions go one by one.
            # That also gives the judges the LLM trace behind every answer.
            if proxy and (a.trace_llm or a.answer_calls):
                records.extend(ask(system, pid, q, a.answer_timeout, proxy, a.answer_calls, a.answer_tokens)
                               for q in qs)
            else:
                with cf.ThreadPoolExecutor(a.concurrency) as ex:
                    records.extend(ex.map(lambda q: ask(system, pid, q, a.answer_timeout, None), qs))
        for t in acts:  # actions are sequential: they share one world
            if time_left() <= 0:
                records.append(skip(t))
                continue
            records.append(act(system, pid, t, world, data,
                               max(5.0, min(a.act_timeout, time_left())), proxy, a.act_calls, a.act_tokens))
        if items:
            log(f"  {len(qs)} questions, {len(acts)} actions after {slot}")

    failed = 0
    for n, m in enumerate(stream, start=1):
        if not budget_blown and time_left() <= 0:
            budget_blown = True
            log("WALL CLOCK EXHAUSTED during the stream: remaining messages skipped, questions still asked")
        if not budget_blown:
            t0, err = time.time(), None
            mark = proxy.mark(a.message_calls, a.message_tokens, "message", m["id"]) if proxy else None
            try:
                http_json(f"{system}/message", {"message": m}, retries=0,
                          timeout=max(1.0, min(a.message_timeout, a.ingest_budget - ingest_used, time_left())))
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                failed += 1
            took = time.time() - t0
            ingest_used += took
            entry = {"id": m["id"], "seconds": round(took, 2), "error": err, "payload_tokens": msg_tokens[m["id"]]}
            if proxy:
                calls = proxy.calls_since(mark)
                entry.update(llm_calls=len(calls), llm_tokens=sum(c.get("prompt_tokens", 0) for c in calls),
                             rejected=sum(1 for c in calls if c.get("status") != 200))
            ingest_log.append(entry)
            if err and failed <= 5:
                log(f"message {m['id']} ERROR {err}")
            if n % 25 == 0 or n == len(stream):
                log(f"sent {n}/{len(stream)} messages ({ingest_used:.0f}s, {failed} errors)")
            if ingest_used >= a.ingest_budget:
                budget_blown = True
                log("MESSAGE TIME BUDGET EXHAUSTED: remaining messages skipped, questions still asked")
        checkpoint(m["id"])

    order = {x["id"]: i for i, x in enumerate(questions + actions)}
    records.sort(key=lambda r: order[r["id"]])
    write_jsonl(out / "answers.jsonl", records)

    usage = None
    if proxy:
        try:
            usage = http_json(f"{proxy.url}/usage", timeout=10)
        except Exception as e:
            log(f"(could not read proxy usage: {e})")
    if a.judge == "none":
        log(f"Answers written to {out/'answers.jsonl'} (not graded)")
        return 0

    qjudge, ajudge = make_judge(a.judge, votes=a.votes), make_action_judge(a.judge, votes=a.votes)
    items = {x["id"]: x for x in questions + actions}
    gold = {m["id"]: m if m["type"] == "chat" else
            {"turns": [{"role": f"app:{m['source']}", "content": json.dumps(m["data"], ensure_ascii=False)}]}
            for m in stream}
    log(f"grading {len(records)} items with {a.judge} judge ...")

    def grade(r):
        it = items[r["id"]]
        base = {**r, "category": it.get("category"), "difficulty": it.get("difficulty", 1)}
        if r["kind"] == "action":
            g = ajudge.grade(it, r) if not r["error"] else {
                "label": "WRONG", "process": 0, "hard_score": r["checks"]["hard_score"], "harmful": r["checks"]["harmful"],
                "score": -0.25 if r["checks"]["harmful"] else 0.0, "votes": [], "reason": f"system error: {r['error']}",
                "judge": "n/a"}
            return {**base, "expected": it["expected_behavior"], **g}
        g = qjudge.grade(it, r, gold) if not r["error"] else {
            "label": "WRONG", "grounding": 0, "usefulness": 0, "score": 0.0, "votes": [],
            "reason": f"system error: {r['error']}", "judge": "n/a"}
        return {**base, "reference_answer": it["reference_answer"], **g}

    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        graded = list(ex.map(grade, records))
    write_jsonl(out / "graded.jsonl", graded)
    summary = summarize(graded, name=name, persona_id=pid, ingest_log=ingest_log, ingest_seconds=ingest_used,
                        budget_blown=budget_blown, usage=usage, judge=a.judge)
    write_json(out / "summary.json", summary)
    (out / "summary.md").write_text(summary_markdown(summary, graded), encoding="utf-8")
    log(f"SCORE {summary['score']:.1f} / 100 (questions {summary['question_score']}, actions "
        f"{summary['action_score']})  ->  {out/'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
