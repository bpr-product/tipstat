"""A bounded tool-calling loop over the mock world's plain-HTTP tools (GET {tools_url}, POST {tools_url}/<name>).

Hard limits mirror the official /act budget: at most 12 model calls (we use at most MAX_CALLS here, the caller may
already have spent some), 60,000 input tokens in total (estimated with the proxy's rule), 8,000 per call, 90 s.
The loop never trusts the model with the tool list: it can only call tools the world listed.
"""
import json
import time
import urllib.request

from backend.source.utils.text import est_tokens

PER_CALL_CAP = 6800      # proxy limit is 8,000 per call; keep a margin for estimate error
TOTAL_CAP = 52000        # proxy limit is 60,000 for the whole job
RESULT_CHARS = 1800      # a single tool result is clipped to this
MAX_TOOL_RUNS = 30
TRIMMED = "[older result trimmed]"


def fetch_tools(tools_url: str) -> list:
    with urllib.request.urlopen(tools_url, timeout=15) as r:
        return json.loads(r.read())["tools"]


def call_tool(tools_url: str, name: str, args: dict) -> str:
    """Run one tool. Failures come back as text for the model to react to; they never raise."""
    try:
        req = urllib.request.Request(f"{tools_url.split('?')[0].rstrip('/')}/{name}", data=json.dumps(args).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=15) as r:
            out = json.loads(r.read())
        text = json.dumps(out["result"] if "result" in out else {"error": out.get("error", "unknown error")}, ensure_ascii=False)
    except Exception as e:
        text = json.dumps({"error": f"{type(e).__name__}: {e}"})
    return text[:RESULT_CHARS]


def _cost(messages, tools) -> float:
    return est_tokens(json.dumps(messages, ensure_ascii=False)) + est_tokens(json.dumps(tools, ensure_ascii=False))


def run(chat, tools_url: str, tools: list, messages: list, max_calls: int, deadline: float, spent: float = 0.0, guard=None):
    """Returns (final reply or None, trace, calls used, tokens spent). `chat(messages, tools, tool_choice, timeout)`."""
    names = {t["function"]["name"] for t in tools}
    trace, calls, runs, nudged = [], 0, 0, False
    while calls < max_calls:
        while _cost(messages, tools) > PER_CALL_CAP:  # shrink the oldest tool results first
            old = next((m for m in messages if m["role"] == "tool" and m["content"] != TRIMMED), None)
            if old is None:
                break
            old["content"] = TRIMMED
        cost = _cost(messages, tools)
        left = deadline - time.time()
        if cost > PER_CALL_CAP or spent + cost > TOTAL_CAP or left < 8:
            trace.append({"step": "stop", "reason": "budget or time", "calls": calls, "est_tokens": round(spent)})
            return None, trace, calls, spent
        if not nudged and (max_calls - calls <= 3 or spent > TOTAL_CAP * 0.55):
            nudged = True  # tell the model the budget is nearly gone so it performs the writes instead of exploring
            messages.append({"role": "user", "content": f"Budget: about {max(1, max_calls - calls)} model call(s) left. "
                                                        "If the request is not done yet, make the required changes now and then reply."})
            cost = _cost(messages, tools)
        last = calls == max_calls - 1 or runs >= MAX_TOOL_RUNS
        try:
            msg = chat(messages, tools, "none" if last else "auto", min(30, int(left) - 3))
        except Exception as e:
            trace.append({"step": "model_error", "error": f"{type(e).__name__}: {e}"})
            return None, trace, calls + 1, spent + cost
        calls, spent = calls + 1, spent + cost
        messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")} | {"role": "assistant"})
        if not msg.get("tool_calls"):
            return (msg.get("content") or "").strip() or None, trace, calls, spent
        for tc in msg["tool_calls"]:
            fn, runs = tc.get("function") or {}, runs + 1
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
                assert isinstance(args, dict)
            except (ValueError, AssertionError):
                result = json.dumps({"error": "arguments must be a JSON object"})
            else:
                blocked = guard(name, args) if guard and name in names else None  # backend rules run before the world sees it
                if blocked:
                    result = json.dumps({"error": blocked})
                    trace.append({"step": "blocked", "name": name, "args": json.dumps(args, ensure_ascii=False)[:300], "reason": blocked[:240]})
                else:
                    result = call_tool(tools_url, name, args) if name in names else json.dumps({"error": f"unknown tool {name}"})
                    if guard:
                        guard.observe(result)
                    trace.append({"step": "tool", "name": name, "args": json.dumps(args, ensure_ascii=False)[:300],
                                  "result": result[:200]})
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})
    return None, trace, calls, spent
