"""Model adapter. Every runtime LLM call goes through here: OPENAI_BASE_URL + OPENAI_API_KEY, one allowed model."""
import json
import urllib.request

from backend.source.utils import config
from backend.source.utils import metrics

# The allowed model is a reasoning model: with a small max_tokens it can spend everything on thinking and
# return "". The evaluation proxy turns reasoning off; we also ask for it so direct OpenRouter calls behave.


def complete(messages, max_tokens: int = 500, temperature: float = 0.2, timeout: int = 25) -> str:
    """One chat completion through the allowed model. Do not call any other provider."""
    req = urllib.request.Request(
        f"{config.BASE_URL}/chat/completions",
        data=json.dumps({"model": config.MODEL, "messages": messages,
                         "max_tokens": max_tokens, "temperature": temperature,
                         "reasoning": {"effort": "none"}}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {config.API_KEY}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    if metrics.current():
        metrics.current().report(data.get("usage"))
    return data["choices"][0]["message"]["content"] or ""


def chat_with_tools(messages, tools, max_tokens: int = 700, temperature: float = 0, timeout: int = 30,
                    tool_choice: str = "auto") -> dict:
    """One tool-calling completion through the allowed model. Returns the assistant message (content, tool_calls)."""
    req = urllib.request.Request(
        f"{config.BASE_URL}/chat/completions",
        data=json.dumps({"model": config.MODEL, "messages": messages, "tools": tools, "tool_choice": tool_choice,
                         "max_tokens": max_tokens, "temperature": temperature,
                         "reasoning": {"effort": "none"}}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {config.API_KEY}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    if metrics.current():
        metrics.current().report(data.get("usage"))
    return data["choices"][0]["message"]
