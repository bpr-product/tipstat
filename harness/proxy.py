"""LLM gateway that sits between a submission and OpenRouter during evaluation.

What it enforces
  - model allowlist           (ALLOWED_MODELS in .env)
  - request-field allowlist   OpenRouter extras that would route around the rules are
                              refused: `models` (fallback to other models), `plugins`
                              (web search, file parsing), `provider`, `transforms`
                              (middle-out compression), `prompt`, legacy `functions` ...
  - text-only content         image/file/audio parts are refused: they carry tokens the
                              cap cannot see
  - max input tokens per call (MAX_INPUT_TOKENS, default 8000). Counted over EVERYTHING the
                              model reads: message text, tool-call arguments, tool results,
                              names, the `tools` schemas and `response_format`. Non-ASCII
                              characters count as one token each (see common.approx_tokens).
                              The estimate is the enforced, documented metric. After the
                              call the upstream's real prompt_tokens are checked too, as a
                              last resort: over MAX_INPUT_TOKENS x MAX_INPUT_TOLERANCE (2x)
                              the response is withheld (413). Real counts run 1.1-1.7x the
                              estimate on ordinary JSON-heavy agent prompts, so the margin
                              only catches deliberate packing, never honest use.
  - max output tokens         (MAX_OUTPUT_TOKENS, default 4096; larger max_tokens is clamped)
  - n = 1                     several completions per call would dodge the call budget
  - per-window budgets        the harness opens a window per question / action / message with a CALL budget and an INPUT-TOKEN budget (429 beyond)
  - no streaming              (keeps accounting simple)
  - reasoning effort          (REASONING_EFFORT, default "none"; the allowed model is a
                               reasoning model and otherwise returns empty content)
  - the real API key never enters the participant's container

What it records (for the judge and the leaderboard)
  - every call: model, tokens, cost, clipped request/response, latency, window
  - totals via GET /usage, per window via GET /calls?since=<mark>

Run one proxy per submission being evaluated (different ports):
  python -m harness.proxy --port 8787 --log runs/alice/llm_calls.jsonl

The submission is started with:
  OPENAI_BASE_URL=http://<host>:8787/v1  OPENAI_API_KEY=anything
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .common import approx_tokens, env

LOCK = threading.Lock()
STATE = {"calls": [], "seq": 0,
         "win": {"label": "", "phase": "", "calls": 0, "tokens": 0, "budget": 0, "token_budget": 0},
         "usage": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                   "cost_usd": 0.0, "rejected": 0, "over_budget": 0, "over_cap_actual": 0}}
CFG: dict = {}

CHAT_FIELDS = {"model", "messages", "max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k",
               "stop", "tools", "tool_choice", "parallel_tool_calls", "response_format", "seed",
               "frequency_penalty", "presence_penalty", "repetition_penalty", "min_p", "n", "user",
               "logprobs", "top_logprobs", "logit_bias", "stream", "reasoning", "usage"}
EMBED_FIELDS = {"model", "input", "encoding_format", "dimensions", "user"}
MSG_FIELDS = {"role", "content", "name", "tool_calls", "tool_call_id"}
# Fields that would route around the rules: refused with a clear error. Anything else a
# framework adds (stream_options, store, metadata, service_tier; on replayed messages
# annotations, audio, refusal, reasoning, reasoning_details ...) is STRIPPED before the
# request goes upstream, so it can neither break a framework nor carry hidden context.
REFUSED_FIELDS = {"models", "plugins", "provider", "transforms", "route", "prompt", "functions",
                  "function_call", "prediction", "web_search_options"}


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "..."


def _msg_text(m: dict) -> str:
    c = m.get("content", "")
    if isinstance(c, list):
        c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return str(c or "")


def _strip(req: dict, allowed: set) -> list[str]:
    """Drop fields the gateway does not forward; return what was dropped."""
    dropped = sorted(k for k in req if k not in allowed)
    for k in dropped:
        req.pop(k)
    for m in req.get("messages") or []:
        if isinstance(m, dict):
            extra = [k for k in m if k not in MSG_FIELDS]
            for k in extra:
                m.pop(k)
            dropped += [f"messages[].{k}" for k in extra if f"messages[].{k}" not in dropped]
    return dropped


def _validate_chat(req: dict) -> str | None:
    """Return a refusal reason, or None if the request is acceptable."""
    refused = sorted(set(req) & REFUSED_FIELDS)
    if refused:
        return (f"request field(s) {refused} are not allowed during evaluation "
                f"(use `tools` for function calling)")
    if req.get("n", 1) not in (1, None):
        return "n must be 1: one completion per call"
    msgs = req.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return "messages must be a non-empty list"
    for m in msgs:
        if not isinstance(m, dict):
            return "every message must be an object"
        c = m.get("content")
        if isinstance(c, list):
            for part in c:
                if not isinstance(part, dict) or part.get("type", "text") != "text":
                    return "only text content parts are allowed (no images, files or audio)"
        elif c is not None and not isinstance(c, str):
            return "message content must be a string or a list of text parts"
    return None


def billable_text(req: dict, embeddings: bool = False) -> str:
    """Everything the model will read, as one string, for the input cap."""
    if embeddings:
        inp = req.get("input", "")
        return " ".join(map(str, inp)) if isinstance(inp, list) else str(inp)
    parts = []
    for m in req.get("messages", []):
        parts.append(_msg_text(m))
        for k in ("name", "tool_call_id"):
            if m.get(k):
                parts.append(str(m[k]))
        if m.get("tool_calls"):
            parts.append(json.dumps(m["tool_calls"], ensure_ascii=False))
    for k in ("tools", "response_format", "stop", "tool_choice"):
        if req.get(k):
            parts.append(json.dumps(req[k], ensure_ascii=False))
    return " ".join(parts)


def _mock_response(path: str, req: dict, est: int) -> tuple[int, dict]:
    """Canned upstream reply for --mock runs (Docker dry runs with no API key, no spend)."""
    model = req.get("model", "")
    if path.endswith("embeddings"):
        inp = req.get("input", "")
        n = len(inp) if isinstance(inp, list) else 1
        return 200, {"object": "list", "model": model,
                     "data": [{"object": "embedding", "index": i, "embedding": [0.0] * 8} for i in range(n)],
                     "usage": {"prompt_tokens": est, "total_tokens": est}}
    return 200, {"id": "mock-0", "object": "chat.completion", "model": model,
                 "choices": [{"index": 0, "finish_reason": "stop",
                              "message": {"role": "assistant",
                                          "content": "[mock proxy] no upstream model was called in this run."}}],
                 "usage": {"prompt_tokens": est, "completion_tokens": 12,
                           "total_tokens": est + 12, "cost": 0.0}}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _reject(self, msg: str, code: int = 400) -> None:
        with LOCK:
            STATE["usage"]["rejected"] += 1
        self._send(code, {"error": {"message": f"[hackathon proxy] {msg}", "type": "proxy_rejected", "code": code}})

    # ------------------------------------------------------------- GET ----
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/usage":
            with LOCK:
                return self._send(200, dict(STATE["usage"]))
        if u.path == "/calls":
            since = int(parse_qs(u.query).get("since", ["0"])[0])
            with LOCK:
                return self._send(200, {"calls": [c for c in STATE["calls"] if c["seq"] > since]})
        if u.path == "/window":
            with LOCK:
                return self._send(200, dict(STATE["win"]))
        if u.path in ("/health", "/v1/models"):
            # OpenAI-shaped, so SDKs and frameworks that list models are happy too.
            return self._send(200, {"object": "list", "ok": True, "allowed_models": CFG["allowed"],
                                    "max_input_tokens": CFG["max_input"],
                                    "data": [{"id": m, "object": "model", "owned_by": "hackathon"}
                                             for m in CFG["allowed"]]})
        self._send(404, {"error": "not found"})

    # ------------------------------------------------------------ POST ----
    def do_POST(self):
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        if u.path == "/usage/reset":
            with LOCK:
                STATE["calls"].clear()
                STATE["usage"] = {k: 0 if k != "cost_usd" else 0.0 for k in STATE["usage"]}
            return self._send(200, {"ok": True})
        if u.path == "/calls/mark":
            # The harness opens a window per message / question / action. The 8K cap
            # stops one giant prompt; the call and token budgets stop nine small ones.
            try:
                b = json.loads(raw or b"{}")
            except Exception:
                b = {}
            win = {"label": str(b.get("label", "")), "phase": str(b.get("phase", "")), "calls": 0, "tokens": 0,
                   "budget": int(b.get("budget", 0) or 0), "token_budget": int(b.get("token_budget", 0) or 0)}
            with LOCK:
                STATE["win"] = win
                return self._send(200, {"mark": STATE["seq"], **win})
        if u.path not in ("/v1/chat/completions", "/v1/embeddings"):
            hint = (" The Responses API is not available: use Chat Completions (OpenAI Agents SDK: "
                    "set_default_openai_api('chat_completions') or OpenAIChatCompletionsModel).")                 if "responses" in u.path else ""
            return self._send(404, {"error": {"message": "[hackathon proxy] only /v1/chat/completions (and "
                                              "/v1/embeddings for allowed models) are proxied." + hint,
                                              "type": "proxy_rejected", "code": 404}})
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            return self._reject("invalid JSON body")
        if not isinstance(req, dict):
            return self._reject("request body must be a JSON object")
        is_chat = u.path.endswith("chat/completions")

        # The allowed model is a reasoning model: left alone it spends the whole
        # max_tokens budget on reasoning and returns EMPTY content. That has nothing to do
        # with memory design, so the gateway settles it the same way for everyone.
        if is_chat and CFG.get("reasoning") and "reasoning" not in req:
            req["reasoning"] = {"effort": CFG["reasoning"]}

        model = req.get("model", "")
        if model not in CFG["allowed"]:
            return self._reject(f"model '{model}' is not allowed. Allowed: {CFG['allowed']}", 403)
        if req.get("stream"):
            return self._reject("streaming is disabled during evaluation; set stream=false")
        if is_chat:
            why = _validate_chat(req)
            if why:
                return self._reject(why, 400)
            req["__stripped"] = _strip(req, CHAT_FIELDS)
            for k in ("max_tokens", "max_completion_tokens"):
                if isinstance(req.get(k), int) and req[k] > CFG["max_output"]:
                    req[k] = CFG["max_output"]
            if "max_tokens" not in req and "max_completion_tokens" not in req:
                req["max_tokens"] = CFG["max_output"]
        else:
            req["__stripped"] = _strip(req, EMBED_FIELDS)
        self._stripped = req.pop("__stripped", [])
        est = approx_tokens(billable_text(req, embeddings=not is_chat))
        if est > CFG["max_input"]:
            return self._reject(f"request is ~{est} input tokens (messages + tool calls + tool schemas); "
                                f"the cap is {CFG['max_input']} per call", 413)

        with LOCK:
            w = STATE["win"]
            if w["budget"] and w["calls"] + 1 > w["budget"]:
                STATE["usage"]["over_budget"] += 1
                over = f"call budget exhausted: {w['budget']} model calls are allowed for this {w['phase'] or 'window'}"
            elif w["token_budget"] and w["tokens"] + est > w["token_budget"]:
                STATE["usage"]["over_budget"] += 1
                over = (f"input-token budget exhausted: {w['tokens']} of {w['token_budget']} tokens already used "
                        f"in this {w['phase'] or 'window'}, this call needs ~{est}")
            else:
                over = None
                w["calls"] += 1
                w["tokens"] += est
        if over:
            return self._reject(over + ". Retrieve less, not more.", 429)

        t0 = time.time()
        if CFG.get("mock"):
            status, mock_out = _mock_response(u.path, req, est)
            body = json.dumps(mock_out).encode()
            self._record_and_send(u.path, req, model, status, body, t0, est, w)
            return
        upstream = urllib.request.Request(CFG["base"] + u.path.replace("/v1", "", 1), data=json.dumps(req).encode(),
                                          method="POST")
        upstream.add_header("Content-Type", "application/json")
        upstream.add_header("Authorization", f"Bearer {CFG['key']}")
        try:
            with urllib.request.urlopen(upstream, timeout=CFG["timeout"]) as r:
                status, body = r.status, r.read()
        except urllib.error.HTTPError as e:
            status, body = e.code, e.read()
        except Exception as e:
            return self._send(502, {"error": {"message": f"[hackathon proxy] upstream error: {e}"}})

        self._record_and_send(u.path, req, model, status, body, t0, est, w)

    # ------------------------------------------------- record + respond ----
    def _record_and_send(self, path: str, req: dict, model: str, status: int, body: bytes,
                         t0: float, est: int, win: dict) -> None:
        try:
            out = json.loads(body)
        except json.JSONDecodeError:
            out = {}
        usage = out.get("usage") or {}
        actual = int(usage.get("prompt_tokens") or est)
        msgs = req.get("messages", [])
        with LOCK:
            w = dict(win)  # windows are charged the estimate, the documented metric
        withheld = status == 200 and actual > CFG["max_input"] * CFG["tolerance"]
        call = {
            "model": model, "endpoint": path, "status": 413 if withheld else status,
            "latency_s": round(time.time() - t0, 2),
            "prompt_tokens": actual, "estimated_tokens": est,
            "completion_tokens": usage.get("completion_tokens", 0),
            "cost_usd": float(usage.get("cost", 0) or 0),
            "window": w["label"], "phase": w["phase"],
            "stripped": getattr(self, "_stripped", []),
            "system": _clip(next((_msg_text(m) for m in msgs if m.get("role") == "system"), ""), 300),
            "last_user": _clip(next((_msg_text(m) for m in reversed(msgs) if m.get("role") == "user"), ""), 600),
            "tools": [t.get("function", {}).get("name") for t in req.get("tools", []) if isinstance(t, dict)],
            "response": _clip(json.dumps(out.get("choices", [{}])[0].get("message", {}), ensure_ascii=False), 600)
            if path.endswith("chat/completions") else "(embedding)",
        }
        if withheld:
            call["response"] = "(withheld: over the input cap by real token count)"
        with LOCK:
            STATE["seq"] += 1
            call["seq"] = STATE["seq"]
            call["ts"] = time.time()
            STATE["calls"].append(call)
            us = STATE["usage"]
            us["calls"] += 1
            us["prompt_tokens"] += call["prompt_tokens"]
            us["completion_tokens"] += call["completion_tokens"]
            us["cost_usd"] = round(us["cost_usd"] + call["cost_usd"], 6)
            if withheld:
                us["over_cap_actual"] += 1
            if CFG["log"]:
                with open(CFG["log"], "a") as f:
                    f.write(json.dumps(call, ensure_ascii=False) + "\n")
        if withheld:
            return self._send(413, {"error": {
                "message": f"[hackathon proxy] the model read {actual} prompt tokens; the cap is {CFG['max_input']} "
                           f"per call (estimate was {est}). Response withheld.",
                "type": "proxy_rejected", "code": 413}})
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--log", default=None, help="Append every call to this JSONL file")
    p.add_argument("--mock", action="store_true",
                   help="Answer locally instead of calling OpenRouter (dry runs: no key, no spend)")
    a = p.parse_args(argv)
    CFG.update({
        "key": "mock" if a.mock else env("OPENROUTER_API_KEY"),
        "mock": a.mock,
        "base": env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),
        "allowed": [m.strip() for m in env("ALLOWED_MODELS", "deepseek/deepseek-v4.1-flash").split(",") if m.strip()],
        "max_input": int(env("MAX_INPUT_TOKENS", "8000")),
        "tolerance": float(env("MAX_INPUT_TOLERANCE", "2.0")),
        "max_output": int(env("MAX_OUTPUT_TOKENS", "4096")),
        "reasoning": env("REASONING_EFFORT", "none").strip().lower(),
        "timeout": float(env("PROXY_UPSTREAM_TIMEOUT", "120")),
        "log": a.log,
    })
    mode = "  [MOCK: no upstream calls]" if a.mock else ""
    print(f"LLM proxy on http://{a.host}:{a.port}/v1  allowed={CFG['allowed']}  max_input={CFG['max_input']} "
          f"max_output={CFG['max_output']}{mode}")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    main()
