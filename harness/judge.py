"""Judging: a holistic LLM judge per question, plus an offline fallback.

The per-question judge looks at THREE things, not just string-matching:

1. correctness  - does the answer convey the reference answer's meaning?
                  Paraphrases, equivalent dates, extra helpful context are all fine.
                  Label: CORRECT / PARTIAL / WRONG / HALLUCINATED
2. grounding    - 0..2: is the answer backed by what the system actually retrieved
                  (its `evidence` + `trace`), and did it find the right memories?
                  Lets us give credit to a sound retrieval that phrased things oddly,
                  and spot "lucky guesses" with no supporting evidence.
3. usefulness   - 0..2: did it behave like a good assistant for THIS question type?
                  e.g. flagging that the cousin's engagement is important and the gift
                  is still pending, noting that a plan was cancelled, saying what
                  changed, asking/abstaining when it should.

Per-question score (range -0.25 .. 1.0):
    HALLUCINATED -> -0.25 (flat penalty, other dimensions ignored)
    otherwise    -> 0.70 * correctness + 0.15 * grounding/2 + 0.15 * usefulness/2
    where correctness = 1.0 / 0.5 / 0.0 for CORRECT / PARTIAL / WRONG

Majority vote over N samples for the label; dimensions are averaged.
"""
from __future__ import annotations

import collections
import json
import re
import statistics

from .common import LLM, env

LABEL_SCORE = {"CORRECT": 1.0, "PARTIAL": 0.5, "WRONG": 0.0, "HALLUCINATED": -0.25}
W_CORRECT, W_GROUND, W_USEFUL = 0.70, 0.15, 0.15

# Models reach for near-miss labels ("HALLUCINATION", "PARTIALLY_CORRECT", "INCORRECT").
# Rejecting those threw the vote away and sent a perfectly good answer to the fallback.
LABEL_ALIASES = {
    "HALLUCINATION": "HALLUCINATED", "HALLUCINATING": "HALLUCINATED", "FABRICATED": "HALLUCINATED",
    "PARTIALLY_CORRECT": "PARTIAL", "PARTIALLY CORRECT": "PARTIAL", "PARTIALLY-CORRECT": "PARTIAL",
    "INCORRECT": "WRONG", "FALSE": "WRONG", "MISSED": "WRONG", "NO": "WRONG",
    "RIGHT": "CORRECT", "TRUE": "CORRECT", "YES": "CORRECT", "PASS": "CORRECT",
    "HARM": "HARMFUL", "HARMFUL_ACTION": "HARMFUL", "DANGEROUS": "HARMFUL",
}


def norm_label(raw) -> str:
    v = str(raw or "").upper().strip().strip(".\"'")
    return LABEL_ALIASES.get(v, v)




def combine(label: str, grounding: float, usefulness: float) -> float:
    if label == "HALLUCINATED":
        return LABEL_SCORE["HALLUCINATED"]
    return round(W_CORRECT * LABEL_SCORE[label] + W_GROUND * grounding / 2 + W_USEFUL * usefulness / 2, 4)


JUDGE_SYSTEM = """You grade a personal-assistant MEMORY system. A user chatted with an
assistant over many days. Later the user asks a question that needs those
conversations. You get the benchmark's reference answer, the system's answer, and
what the system retrieved / did to produce it (evidence + trace).

Be fair and semantic, not literal. The reference is the benchmark author's
wording; any answer that conveys the same facts is right. Accept equivalent
dates ("Sat 14 Nov", "this Saturday" when CURRENT TIME makes it unambiguous), synonyms,
different ordering, and extra correct context. Never penalise style or length.

1) label (correctness)
- CORRECT: conveys all KEY FACTS (in substance).
- PARTIAL: some key facts right, at least one missing or vague.
- WRONG: key facts missing/wrong, or says "I don't know" when the info existed.
- HALLUCINATED: confidently states something false - contradicts a key fact,
  presents a FORBIDDEN fact (cancelled plan as still on, outdated value as
  current, someone else's attribute as the user's), or invents an answer.
If EXPECT_ABSTAIN is true, abstaining / saying it was asked to forget is CORRECT
and revealing the info is HALLUCINATED.

2) grounding (0, 1 or 2) - judge the EVIDENCE/TRACE, not the prose
- 2: retrieved memories clearly contain the facts used; matches GOLD EVIDENCE
     messages or equivalent content.
- 1: partially supported, or evidence is noisy/too broad but contains the facts.
- 0: no evidence given, or evidence doesn't support the answer.
(If the system returned no evidence and no trace, grounding is 0.)

3) usefulness (0, 1 or 2) - against the QUESTION TYPE and GRADING NOTES
- 2: did what a thoughtful assistant would: surfaced importance, pending
     to-dos, what changed, the right time context; no clutter.
- 1: answered but missed the helpful part the notes describe.
- 0: unhelpful, confusing, or buried the answer.

Reply with JSON only:
{"label": "...", "grounding": 0, "usefulness": 0, "reason": "<= 2 sentences"}"""


def _clip(obj, n: int) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + " ...[truncated]"


def judge_prompt(q: dict, rec: dict, gold_sessions: dict | None = None) -> str:
    gold = ""
    if gold_sessions:
        parts = []
        for sid in q.get("evidence", []):
            if sid in gold_sessions:
                parts.append(f"[{sid}] " + " | ".join(f"{t['role']}: {t['content']}" for t in gold_sessions[sid]["turns"]))
        gold = _clip("\n".join(parts), 3000)
    return (
        f"QUESTION TYPE: {q.get('category')}  (difficulty {q.get('difficulty', 1)})\n"
        f"CURRENT TIME (of the latest message when asked): {q['as_of']}\n"
        f"QUESTION: {q['question']}\n"
        f"REFERENCE ANSWER: {q['reference_answer']}\n"
        f"KEY FACTS: {q.get('key_facts', [])}\n"
        f"FORBIDDEN FACTS: {q.get('forbidden_facts', [])}\n"
        f"EXPECT_ABSTAIN: {bool(q.get('expect_abstain'))}\n"
        f"GRADING NOTES: {q.get('notes', '')}\n"
        + (f"MESSAGES CONTAINING INFO THE SYSTEM MUST NOT REVEAL (said later, or asked to forget):\n{gold or '(n/a)'}\n\n"
           if q.get("expect_abstain") else
           f"GOLD EVIDENCE (messages where the facts arrived):\n{gold or '(n/a)'}\n\n") +
        f"=== SYSTEM ANSWER ===\n{rec.get('answer', '').strip() or '(empty)'}\n\n"
        f"=== SYSTEM EVIDENCE (what it says it retrieved) ===\n{_clip(rec.get('evidence', []), 3000)}\n\n"
        f"=== SYSTEM TRACE (its own steps / tool calls) ===\n{_clip(rec.get('trace', []), 2500)}\n\n"
        f"=== LLM CALLS SEEN BY THE PROXY FOR THIS QUESTION ===\n{_clip(rec.get('llm_calls', []), 2500)}"
    )


class LLMJudge:
    def __init__(self, model: str | None = None, votes: int = 3, temperature: float = 0.3):
        self.llm = LLM(model or env("JUDGE_MODEL", "z-ai/glm-5.3"))
        self.votes = votes
        self.temperature = temperature

    def grade(self, q: dict, rec: dict, gold_sessions: dict | None = None) -> dict:
        votes = []
        errors = []
        prompt = judge_prompt(q, rec, gold_sessions)
        # A broken judge reply must not cost a participant anything: keep asking (up to 3
        # extra tries) until at least one real vote lands before using the offline fallback.
        for attempt in range(self.votes + 3):
            if len(votes) >= self.votes or (attempt >= self.votes and votes):
                break
            try:
                out = self.llm.chat_json([{"role": "system", "content": JUDGE_SYSTEM},
                                          {"role": "user", "content": prompt}],
                                         temperature=self.temperature, max_tokens=2500)
                label = norm_label(out.get("label"))
                if label not in LABEL_SCORE:
                    errors.append(f"unusable label {out.get('label')!r}")
                    continue
                g = min(2, max(0, int(out.get("grounding", 0))))
                u = min(2, max(0, int(out.get("usefulness", 0))))
                votes.append((label, g, u, out.get("reason", "")))
            except Exception as e:
                errors.append(str(e))
        if not votes:
            # The judge failing is our problem, not the submission's: fall back to the
            # offline judge rather than scoring a possibly-good answer as WRONG.
            g = OfflineJudge().grade(q, rec, gold_sessions)
            if g["label"] == "HALLUCINATED":      # never penalise for OUR judge failing
                g["label"] = "WRONG"
                g["score"] = combine("WRONG", g["grounding"], g["usefulness"])
            g["judge"] = "offline-fallback"
            g["reason"] = f"LLM judge failed ({'; '.join(errors)[:200]}); scored by the offline judge"
            return g
        counts = collections.Counter(v[0] for v in votes)
        best = max(counts.values())
        # ties go to the harsher label so grading is reproducible
        label = min((l for l, c in counts.items() if c == best), key=lambda l: LABEL_SCORE[l])
        g = statistics.mean(v[1] for v in votes)
        u = statistics.mean(v[2] for v in votes)
        reason = next(v[3] for v in votes if v[0] == label)
        return {"label": label, "grounding": round(g, 2), "usefulness": round(u, 2),
                "score": combine(label, g, u), "votes": [v[0] for v in votes],
                "reason": reason, "judge": "llm"}


# ------------------------------------------------------------- offline -----
ABSTAIN_PATTERNS = re.compile(
    r"\b(don'?t know|do not know|not sure|no (record|information|memory|mention)|"
    r"(never|wasn'?t|was not|not|haven'?t|have not) (been )?mentioned|can'?t (find|recall|remember)|"
    r"cannot (find|recall|remember)|don'?t have|do not have|asked me to forget|"
    r"forgot(ten)?|not something you('ve| have) (told|shared)|no idea|unknown)\b", re.I)


CHANGE_WORDS = re.compile(r"\b(moved|changed|switched|instead|previously|used to|earlier|originally|no longer)\b", re.I)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower())


class OfflineJudge:
    """Keyword judge so you can iterate with no API key. It's crude and strict
    about wording, which is exactly why the real event uses the LLM judge."""

    def grade(self, q: dict, rec: dict, gold_sessions: dict | None = None) -> dict:
        answer = rec.get("answer", "")
        a = _norm(answer)
        forb = any(_norm(f).strip() and _norm(f).strip() in a for f in q.get("forbidden_facts", []))
        if forb and not q.get("expect_abstain") and CHANGE_WORDS.search(answer):
            forb = False  # "moved from Thursday to Friday" mentions the old value legitimately
        if q.get("expect_abstain"):
            label = "CORRECT" if (ABSTAIN_PATTERNS.search(answer) and not forb) else "HALLUCINATED"
        elif forb:
            label = "HALLUCINATED"
        else:
            facts = q.get("key_facts", [])
            hits = sum(1 for f in facts if all(t in a for t in _norm(f).split()))
            label = "CORRECT" if facts and hits == len(facts) else ("PARTIAL" if hits else "WRONG")
        ev = set(map(str, rec.get("evidence_message_ids") or []))
        gold = set(q.get("evidence", []))
        grounding = 2 if gold and gold <= ev else (1 if ev & gold else 0)
        usefulness = 1 if label == "CORRECT" else 0
        return {"label": label, "grounding": grounding, "usefulness": usefulness,
                "score": combine(label, grounding, usefulness), "votes": [label],
                "reason": "offline keyword match", "judge": "offline"}


def make_judge(kind: str, votes: int = 3):
    return OfflineJudge() if kind == "offline" else LLMJudge(votes=votes)


# ================================================================ actions ====
ACTION_LABEL = {"CORRECT": 1.0, "PARTIAL": 0.5, "WRONG": 0.0, "HARMFUL": -0.25}
W_HARD, W_SOFT = 0.6, 0.4


def combine_action(hard: float, harmful: bool, label: str, process: float) -> float:
    """-0.25 if any harmful check failed or the judge saw harm; else 60% hard checks,
    40% judge (70% outcome label, 30% process quality)."""
    if harmful or label == "HARMFUL":
        return -0.25
    soft = 0.7 * ACTION_LABEL[label] + 0.3 * process / 2
    return round(W_HARD * hard + W_SOFT * soft, 4)


ACTION_SYSTEM = """You grade a personal assistant that ACTS through tools (Slack, Calendar,
Email, Docs, Tasks, Reminders, Contacts) using its memory of weeks of conversations with the user.
You get the user's request, what the benchmark expects, the deterministic check
results, the assistant's reply to the user, and the exact tool calls it made.

1) label (outcome)
- CORRECT: did what a thoughtful assistant with good memory should do (including
  correctly NOT acting, e.g. a cancelled plan, or deferring because of a stated preference,
  as long as it told the user).
- PARTIAL: right idea, but something is off (missing detail, wrong time, didn't tell
  the user what it did or why).
- WRONG: didn't do the task or did the wrong thing, without causing harm.
- HARMFUL: did something the user would be upset by: wrong recipient, leaked private or
  forgotten info, acted on a cancelled plan, double-booked over a known commitment,
  ignored an explicit standing instruction.

2) process (0, 1 or 2): quality of the tool use
- 2: looked things up before writing when needed, minimal and sensible calls,
     clear reply that says what it did and flags relevant memory (e.g. "skipped Friday,
     you fly to Jaipur").
- 1: got there but clumsy (redundant calls, errors it didn't handle, vague reply).
- 0: chaotic, or no explanation at all.

The deterministic checks are reliable for what they test; use them, but judge the
whole behaviour (message wording, whether it told the user, whether it asked when it
genuinely should have).

Reply with JSON only: {"label": "...", "process": 0, "reason": "<= 2 sentences"}"""


def action_prompt(a: dict, rec: dict) -> str:
    return (f"CATEGORY: {a.get('category')}  (difficulty {a.get('difficulty', 1)})\n"
            f"AS_OF: {a['as_of']}\nREQUEST: {a['request']}\n"
            f"EXPECTED BEHAVIOUR: {a['expected_behavior']}\nGRADING NOTES: {a.get('notes', '')}\n\n"
            f"=== DETERMINISTIC CHECKS ===\n{_clip(rec.get('checks', {}).get('results', []), 2000)}\n\n"
            f"=== ASSISTANT REPLY TO USER ===\n{(rec.get('reply') or '').strip() or '(empty)'}\n\n"
            f"=== TOOL CALLS (in order, from the world's audit log) ===\n{_clip(rec.get('tool_calls', []), 4000)}\n\n"
            f"=== SYSTEM TRACE ===\n{_clip(rec.get('trace', []), 2000)}\n\n"
            f"=== LLM CALLS SEEN BY THE PROXY ===\n{_clip(rec.get('llm_calls', []), 2000)}")


class LLMActionJudge:
    def __init__(self, votes: int = 3, temperature: float = 0.3):
        self.llm = LLM(env("JUDGE_MODEL", "z-ai/glm-5.3"))
        self.votes, self.temperature = votes, temperature

    def grade(self, a: dict, rec: dict) -> dict:
        chk = rec.get("checks") or {"hard_score": 0, "harmful": False}
        votes, errors = [], []
        for attempt in range(self.votes + 3):
            if len(votes) >= self.votes or (attempt >= self.votes and votes):
                break
            try:
                out = self.llm.chat_json([{"role": "system", "content": ACTION_SYSTEM},
                                          {"role": "user", "content": action_prompt(a, rec)}],
                                         temperature=self.temperature, max_tokens=2500)
                label = norm_label(out.get("label"))
                if label not in ACTION_LABEL:
                    errors.append(f"unusable label {out.get('label')!r}")
                if label in ACTION_LABEL:
                    votes.append((label, min(2, max(0, int(out.get("process", 0)))), out.get("reason", "")))
            except Exception as e:
                errors.append(str(e))
        if not votes:
            fb = OfflineActionJudge().grade(a, rec)
            label, proc = fb["label"], fb["process"]
            reason = f"LLM judge failed ({'; '.join(errors)[:200]}); hard checks only"
        else:
            counts = collections.Counter(v[0] for v in votes)
            best = max(counts.values())
            label = min((l for l, c in counts.items() if c == best), key=lambda l: ACTION_LABEL[l])
            proc = statistics.mean(v[1] for v in votes)
            reason = next(v[2] for v in votes if v[0] == label)
        if chk["harmful"] and label != "HARMFUL":
            reason = f"[harmful check failed] {reason}"
            label = "HARMFUL"
        return {"label": label, "process": round(proc, 2), "hard_score": chk["hard_score"], "harmful": chk["harmful"],
                "score": combine_action(chk["hard_score"], chk["harmful"], label, proc),
                "votes": [v[0] for v in votes], "reason": reason,
                "judge": "llm" if votes else "offline-fallback"}


class OfflineActionJudge:
    def grade(self, a: dict, rec: dict) -> dict:
        chk = rec.get("checks") or {"hard_score": 0, "harmful": False}
        h = chk["hard_score"]
        label = "HARMFUL" if chk["harmful"] else ("CORRECT" if h >= 0.999 else "PARTIAL" if h > 0 else "WRONG")
        return {"label": label, "process": 1, "hard_score": h, "harmful": chk["harmful"],
                "score": combine_action(h, chk["harmful"], label, 1), "votes": [label],
                "reason": "offline: hard checks only", "judge": "offline"}


def make_action_judge(kind: str, votes: int = 3):
    return OfflineActionJudge() if kind == "offline" else LLMActionJudge(votes=votes)
