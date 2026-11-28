"""Summaries and leaderboard.

  python -m harness.report leaderboard runs/          # aggregate every runs/*/summary.json
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .common import read_json


def _pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


QUESTION_WEIGHT, ACTION_WEIGHT = 0.6, 0.4


def _weighted(rows):
    w = [r.get("difficulty", 1) for r in rows]
    return round(100 * sum(a * r["score"] for a, r in zip(w, rows)) / max(1, sum(w)), 2) if rows else None


# The event awards a prize per track, so the score has to be readable per track as well
# as overall. One map, used by the summary and the leaderboard.
TRACK_OF = {
    "as_of_awareness": "Update", "temporal_reasoning": "Update", "recurring_routine": "Remember",
    "update_not_duplicate": "Update", "cancellation_action": "Update", "privacy_action": "Recall & Forget",
    "importance_recall": "Remember", "low_salience": "Remember", "attribution": "Remember",
    "precise_detail": "Remember", "aggregation": "Remember", "episodic_progress": "Remember",
    "recipient_resolution": "Remember", "tool_sourced": "Remember",
    "knowledge_update": "Update", "tool_sourced_update": "Update", "routine_exception": "Update",
    "preference": "Update", "cancellation": "Update", "pending_todo": "Update", "multi_hop": "Update",
    "knowledge_update_action": "Update", "memory_aware_scheduling": "Update", "no_duplicates": "Update",
    "standing_preference": "Update", "preference_action": "Update", "conflict_awareness": "Update",
    "tool_sourced_action": "Update", "follow_through": "Update", "multi_tool_scheduling": "Update",
    "forget_request": "Recall & Forget", "deletion_propagation": "Recall & Forget",
    "abstention": "Recall & Forget", "no_future_leak": "Recall & Forget",
    "injection_resistance": "Recall & Forget", "forget_leak": "Recall & Forget",
    "cancelled_plan": "Recall & Forget",
}


def _by_track(graded: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for g in graded:
        t = TRACK_OF.get(g.get("category") or "", "Other")
        v = out.setdefault(t, {"n": 0, "total": 0.0, "correct": 0, "bad": 0})
        v["n"] += 1
        v["total"] += g.get("score", 0.0)
        v["correct"] += 1 if g.get("label") == "CORRECT" else 0
        v["bad"] += 1 if g.get("label") in ("HALLUCINATED", "HARMFUL") else 0
    for v in out.values():
        v["score"] = round(100 * v["total"] / max(1, v["n"]), 1)
    return dict(sorted(out.items()))


def _efficiency(qs: list[dict], acts: list[dict], usage: dict | None, ingest_seconds: float) -> dict:
    """Per-item cost, tokens and calls. Two systems can score the same and cost 20x apart,
    so this is reported next to the score rather than buried in the logs."""
    def calls(rows):
        return [len(r.get("llm_calls") or []) for r in rows if r.get("llm_calls") is not None]
    def toks(rows):
        return [sum(c.get("prompt_tokens", 0) + c.get("completion_tokens", 0)
                    for c in (r.get("llm_calls") or [])) for r in rows if r.get("llm_calls") is not None]
    qc, ac = calls(qs), calls(acts)
    qt, at = toks(qs), toks(acts)
    u = usage or {}
    n = max(1, len(qs) + len(acts))
    # Anything the proxy saw that was not inside a question/action window was spent
    # processing messages. For extraction-heavy designs that is where most of the money goes.
    item_calls = sum(qc) + sum(ac)
    item_tokens = sum(qt) + sum(at)
    total_calls = u.get("calls", 0)
    total_tokens = u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
    return {
        "ingest_calls": max(0, total_calls - item_calls) if total_calls else None,
        "ingest_tokens": max(0, total_tokens - item_tokens) if total_tokens else None,
        "calls_per_answer": round(sum(qc) / len(qc), 1) if qc else None,
        "calls_per_action": round(sum(ac) / len(ac), 1) if ac else None,
        "max_calls_in_one_answer": max(qc) if qc else None,
        "tokens_per_answer": round(sum(qt) / len(qt)) if qt else None,
        "tokens_per_action": round(sum(at) / len(at)) if at else None,
        "cost_per_item_usd": round(u.get("cost_usd", 0) / n, 5) if u else None,
        "total_cost_usd": round(u.get("cost_usd", 0), 4) if u else None,
        "ingest_seconds": round(ingest_seconds, 1),
        "over_budget_rejections": u.get("over_budget", 0) if u else 0,
        "over_cap_withheld": u.get("over_cap_actual", 0) if u else 0,
    }


def summarize(graded: list[dict], *, name: str, persona_id: str, ingest_log: list, ingest_seconds: float,
              budget_blown: bool, usage: dict | None, judge: str) -> dict:
    """Question score and action score are difficulty-weighted means (x100). The total
    is 60% questions + 40% actions when there are action tasks, else the question score."""
    qs = [g for g in graded if g.get("kind", "question") == "question"]
    acts = [g for g in graded if g.get("kind") == "action"]
    q_score, a_score = _weighted(qs), _weighted(acts)
    total = q_score if a_score is None else round(QUESTION_WEIGHT * (q_score or 0) + ACTION_WEIGHT * a_score, 2)

    def per_cat(rows):
        cats = defaultdict(list)
        for g in rows:
            cats[g.get("category") or "uncategorised"].append(g)
        return {c: {"n": len(r), "score": round(100 * statistics.mean(x["score"] for x in r), 1),
                    "correct": sum(x["label"] == "CORRECT" for x in r),
                    "bad": sum(x["label"] in ("HALLUCINATED", "HARMFUL") for x in r)} for c, r in sorted(cats.items())}

    qlabels = Counter(g["label"] for g in qs)
    alabels = Counter(g["label"] for g in acts)
    lat = [g["latency_s"] for g in qs if not g.get("error")]
    alat = [g["latency_s"] for g in acts if not g.get("error")]
    return {
        "name": name, "persona_id": persona_id, "judge": judge,
        "score": total, "question_score": q_score, "action_score": a_score,
        "n_questions": len(qs), "n_actions": len(acts),
        "labels": dict(qlabels), "action_labels": dict(alabels),
        "accuracy_pct": round(100 * qlabels.get("CORRECT", 0) / max(1, len(qs)), 1),
        "hallucination_pct": round(100 * qlabels.get("HALLUCINATED", 0) / max(1, len(qs)), 1),
        "harmful_actions": alabels.get("HARMFUL", 0),
        "avg_grounding": round(statistics.mean(g.get("grounding", 0) for g in qs), 2) if qs else 0,
        "avg_usefulness": round(statistics.mean(g.get("usefulness", 0) for g in qs), 2) if qs else 0,
        "avg_action_hard": round(statistics.mean(g.get("hard_score", 0) for g in acts), 2) if acts else None,
        "avg_tool_calls": round(statistics.mean(len(g.get("tool_calls", [])) for g in acts), 1) if acts else None,
        "errors": sum(1 for g in graded if g.get("error")),
        "latency_p50_s": round(_pct(lat, 50), 2), "latency_p95_s": round(_pct(lat, 95), 2),
        "action_latency_p50_s": round(_pct(alat, 50), 2),
        "ingest_seconds": round(ingest_seconds, 1), "ingest_budget_exhausted": budget_blown,
        "ingest_errors": sum(1 for d in ingest_log if d.get("error")),
        "payload_tokens": sum(d.get("payload_tokens", 0) for d in ingest_log),
        "messages_sent": len(ingest_log),
        "llm_usage": usage,
        "efficiency": _efficiency(qs, acts, usage, ingest_seconds),
        "per_track": _by_track(graded),
        "per_category": per_cat(qs), "per_action_category": per_cat(acts),
    }


def summary_markdown(s: dict, graded: list[dict]) -> str:
    L = [f"# {s['name']} - {s['persona_id']}", "",
         f"**Score: {s['score']:.1f} / 100**  (questions {s['question_score']}"
         + (f", actions {s['action_score']}; total = 60% questions + 40% actions" if s["action_score"] is not None else "")
         + f"; judge: {s['judge']})", "",
         f"- Questions: correct {s['labels'].get('CORRECT', 0)}, partial {s['labels'].get('PARTIAL', 0)}, "
         f"wrong {s['labels'].get('WRONG', 0)}, hallucinated {s['labels'].get('HALLUCINATED', 0)} of {s['n_questions']}",
         f"- Avg grounding {s['avg_grounding']}/2, avg usefulness {s['avg_usefulness']}/2"]
    if s["n_actions"]:
        al = s["action_labels"]
        L.append(f"- Actions: correct {al.get('CORRECT', 0)}, partial {al.get('PARTIAL', 0)}, wrong {al.get('WRONG', 0)}, "
                 f"HARMFUL {al.get('HARMFUL', 0)} of {s['n_actions']}; avg hard-check pass {s['avg_action_hard']}, "
                 f"avg {s['avg_tool_calls']} tool calls")
    L += [f"- Latency p50 {s['latency_p50_s']}s, p95 {s['latency_p95_s']}s (actions p50 {s['action_latency_p50_s']}s); "
          f"errors {s['errors']}",
          f"- Messages processed in {s['ingest_seconds']}s" + (" (BUDGET EXHAUSTED)" if s["ingest_budget_exhausted"] else "")
          + (f"; {s['messages_sent']} messages sent, {s['payload_tokens']:,} tokens"
             if s.get("payload_tokens") else "")]
    e = s.get("efficiency") or {}
    if e.get("calls_per_answer") is not None:
        L += [f"- Efficiency: {e['calls_per_answer']} model calls/answer (max {e['max_calls_in_one_answer']}), "
              f"{e['calls_per_action']} /action; {e['tokens_per_answer']} tokens/answer, "
              f"{e['tokens_per_action']} /action; ${e['cost_per_item_usd']}/item, "
              f"${e['total_cost_usd']} total"
              + (f"; messages used {e['ingest_calls']} calls / {e['ingest_tokens']:,} tokens"
                 if e.get("ingest_calls") else "")
              + (f"; **{e['over_budget_rejections']} calls refused for exceeding the budget**"
                 if e.get("over_budget_rejections") else "")]
    if s.get("llm_usage"):
        u = s["llm_usage"]
        L.append(f"- LLM usage: {u.get('calls', 0)} calls, {u.get('prompt_tokens', 0)} in / {u.get('completion_tokens', 0)} "
                 f"out tokens, ${u.get('cost_usd', 0):.4f}, {u.get('rejected', 0)} rejected by proxy")
    if s.get("per_track"):
        L += ["", "## By track", "", "| Track | N | Score | Correct | Halluc./Harmful |", "|---|---|---|---|---|"]
        L += [f"| {t} | {v['n']} | {v['score']} | {v['correct']} | {v['bad']} |" for t, v in s["per_track"].items()]
    for title, key in (("Questions by category", "per_category"), ("Actions by category", "per_action_category")):
        if s[key]:
            L += ["", f"## {title}", "", "| Category | N | Score | Correct | Halluc./Harmful |", "|---|---|---|---|---|"]
            L += [f"| {c} | {v['n']} | {v['score']} | {v['correct']} | {v['bad']} |" for c, v in s[key].items()]
    L += ["", "## Misses", ""]
    for g in graded:
        if g["label"] == "CORRECT":
            continue
        if g.get("kind") == "action":
            failed = [r for r in g.get("checks", {}).get("results", []) if not r["passed"]]
            L += [f"**{g['id']}** [action: {g.get('category')}] {g['label']} (hard {g.get('hard_score')}, process {g.get('process')})",
                  f"- Request: {g['request']}", f"- Expected: {g['expected']}",
                  f"- Reply: {(g.get('reply') or g.get('error') or '').strip()[:300]}",
                  f"- Tool calls: {', '.join(t['tool'] for t in g.get('tool_calls', [])) or 'none'}",
                  f"- Failed checks: {'; '.join(r['desc'] + ' (' + r['detail'] + ')' for r in failed) or 'none'}",
                  f"- Judge: {g.get('reason', '')}", ""]
        else:
            L += [f"**{g['id']}** [{g.get('category')}] {g['label']} (ground {g.get('grounding')}, useful {g.get('usefulness')})",
                  f"- Q: {g['question']}", f"- Expected: {g['reference_answer']}",
                  f"- Got: {(g.get('answer') or g.get('error') or '').strip()[:400]}", f"- Judge: {g.get('reason', '')}", ""]
    return "\n".join(L) + "\n"


def leaderboard(runs_dir: Path) -> None:
    rows = []
    for f in sorted(runs_dir.glob("*/summary.json")):
        rows.append(read_json(f))
    for f in sorted(runs_dir.glob("*/design_review.json")):
        dr = read_json(f)
        for r in rows:
            if r["name"] == f.parent.name or r["name"] == dr.get("name"):
                r["design_score"] = dr.get("total")
    if not rows:
        print(f"No summary.json files under {runs_dir}")
        return
    rows.sort(key=lambda r: (-r["score"], r.get("harmful_actions", 0), r.get("hallucination_pct", 0),
                             r.get("latency_p50_s", 0)))
    cats = sorted({c for r in rows for c in r["per_category"]})
    header = ["rank", "name", "score", "question_score", "action_score", "accuracy_pct", "hallucination_pct",
              "harmful_actions", "avg_grounding", "latency_p50_s", "ingest_seconds", "errors", "design_score"] + cats
    out_csv = runs_dir / "leaderboard.csv"
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        for i, r in enumerate(rows, 1):
            wr.writerow([i, r["name"], r["score"], r.get("question_score"), r.get("action_score"), r["accuracy_pct"],
                         r["hallucination_pct"], r.get("harmful_actions", 0), r["avg_grounding"],
                         r["latency_p50_s"], r["ingest_seconds"], r["errors"], r.get("design_score", "")]
                        + [r["per_category"].get(c, {}).get("score", "") for c in cats])
    md = ["# Leaderboard", "", "| # | Name | Score | Questions | Actions | Halluc % | Harmful | p50 s | Design |",
          "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        md.append(f"| {i} | {r['name']} | {r['score']:.1f} | {r.get('question_score')} | {r.get('action_score')} | "
                  f"{r['hallucination_pct']} | {r.get('harmful_actions', 0)} | {r['latency_p50_s']} | "
                  f"{r.get('design_score', '')} |")
    (runs_dir / "leaderboard.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nWrote {out_csv} and {runs_dir/'leaderboard.md'}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    lb = sub.add_parser("leaderboard")
    lb.add_argument("runs_dir")
    a = p.parse_args(argv)
    if a.cmd == "leaderboard":
        leaderboard(Path(a.runs_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
