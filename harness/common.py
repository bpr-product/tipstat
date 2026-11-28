"""Shared helpers: config, JSONL I/O, a tiny stdlib-only OpenAI-compatible client.

No third-party dependencies, so this runs on any machine with Python 3.10+.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- config ----
def load_env(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from .env into os.environ (without overriding)."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def env(name: str, default: str | None = None) -> str:
    load_env()
    val = os.environ.get(name, default)
    if val is None:
        raise SystemExit(f"Missing required setting {name} (set it in .env)")
    return val


# ------------------------------------------------------------------- I/O ----
def read_jsonl(path: Path | str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise SystemExit(f"{path}:{i}: bad JSON: {e}")
    return rows


def write_jsonl(path: Path | str, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------ HTTP ----
# Transient failures we retry: a submission's single-threaded server dropping a
# connection, a port race, or a 5xx while it is still warming up. A 4xx is a real
# answer from the submission and is never retried.
_TRANSIENT = (ConnectionResetError, ConnectionRefusedError, ConnectionAbortedError,
              BrokenPipeError, urllib.error.URLError)


def _is_timeout(e: Exception) -> bool:
    """A timeout means the submission is slow, not that the connection glitched.
    Retrying it would multiply the per-item budget, so we never do."""
    if isinstance(e, TimeoutError):
        return True
    reason = getattr(e, "reason", None)
    return isinstance(reason, TimeoutError) or "timed out" in str(reason or e).lower()


def http_json(url: str, payload: dict | None = None, *, headers: dict | None = None,
              timeout: float = 60.0, method: str | None = None, retries: int = 2) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    last: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"))
        req.add_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode() or "{}"
                return json.loads(body)
        except urllib.error.HTTPError as e:
            if e.code >= 500 and attempt < retries:
                last = e
                time.sleep(0.4 * (attempt + 1))
                continue
            raise
        except TimeoutError:
            raise
        except _TRANSIENT as e:
            last = e
            if _is_timeout(e):
                raise
            if attempt < retries:
                time.sleep(0.4 * (attempt + 1))
                continue
            raise
    raise last if last else RuntimeError("http_json: unreachable")


# ------------------------------------------------------------------- LLM ----
class LLM:
    """Minimal chat-completions client for OpenRouter (or any OpenAI-compatible API)."""

    def __init__(self, model: str, *, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 120.0, retries: int = 4):
        self.model = model
        self.api_key = api_key or env("OPENROUTER_API_KEY")
        self.base_url = (base_url or env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def chat_message(self, messages: list[dict], *, tools: list | None = None, **kw) -> dict:
        """Like chat(), but returns the whole assistant message (content + tool_calls)."""
        return self.chat(messages, tools=tools, _whole=True, **kw)

    def chat(self, messages: list[dict], *, temperature: float = 0.7, max_tokens: int = 4000,
             json_mode: bool = False, tools: list | None = None, _whole: bool = False):
        payload: dict[str, Any] = {"model": self.model, "messages": messages,
                                   "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            payload["tools"] = tools
        # Reasoning models spend max_tokens on reasoning and can return empty content, or
        # JSON cut off half way. Unless asked otherwise we turn that off for our own calls.
        effort = env("REASONING_EFFORT", "none").strip().lower()
        if effort and effort != "default" and not getattr(self, "_no_reasoning_param", False):
            payload["reasoning"] = {"effort": effort}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        last_err: Exception | None = None
        for attempt in range(self.retries):
            try:
                out = http_json(f"{self.base_url}/chat/completions", payload,
                                headers={"Authorization": f"Bearer {self.api_key}"},
                                timeout=self.timeout)
                if "error" in out:
                    raise RuntimeError(out["error"])
                u = out.get("usage") or {}
                self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
                self.usage["completion_tokens"] += u.get("completion_tokens", 0)
                self.usage["calls"] += 1
                msg = out["choices"][0]["message"]
                return msg if _whole else (msg["content"] or "")
            except urllib.error.HTTPError as e:
                # The reason lives in the response body, not in str(e). Some models
                # (z-ai/glm-5.3) refuse to have reasoning switched off: drop the
                # parameter for this client and retry rather than failing the call.
                try:
                    body = e.read().decode()[:400]
                except Exception:
                    body = ""
                if "reasoning" in body.lower() and "reasoning" in payload:
                    payload.pop("reasoning", None)
                    self._no_reasoning_param = True
                    continue
                last_err = RuntimeError(f"HTTP {e.code}: {body or e}")
                time.sleep(2 ** attempt)
            except (urllib.error.URLError, RuntimeError, KeyError, TimeoutError) as e:
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after {self.retries} attempts: {last_err}")

    def chat_json(self, messages: list[dict], **kw) -> Any:
        """Chat and parse a JSON object out of the reply (tolerates ``` fences)."""
        for attempt in range(3):
            text = self.chat(messages, json_mode=True, **kw)
            try:
                return parse_json_loose(text)
            except ValueError:
                messages = messages + [{"role": "assistant", "content": text},
                                       {"role": "user", "content": "That was not valid JSON. Reply with ONLY the JSON object."}]
        raise ValueError("Model did not return valid JSON")


def parse_json_loose(text: str) -> Any:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    raise ValueError("no JSON found")


def approx_tokens(text: str) -> int:
    """Conservative token estimate for budget caps.

    ASCII text runs ~4 characters per token, but CJK, Devanagari, emoji and other
    non-ASCII scripts run close to ONE character per token. Counting everything as
    chars/4 lets a system pack ~4x more context into a "capped" call by writing its
    memory in another script, so non-ASCII characters count as a token each.
    """
    ascii_n = sum(1 for ch in text if ord(ch) < 128)
    return max(1, ascii_n // 4 + (len(text) - ascii_n))
