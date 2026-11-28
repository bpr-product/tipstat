"""Per-operation model usage for the product UI: what a Tell, Ask or Do actually spent.

Numbers come from the provider's own `usage` field when the response had one ("provider"); otherwise they are estimated
with the proxy's counting rule ("estimated"). Recording never makes a model call and is a no-op unless an operation is
being metered, so the official service path pays nothing.
"""
import contextvars
import json
import time
from contextlib import contextmanager

from backend.source.utils.text import est_tokens

_current = contextvars.ContextVar("usage_meter", default=None)


class Meter:
    def __init__(self) -> None:
        self.calls, self._reported, self.started = [], None, time.monotonic()

    def report(self, usage) -> None:
        """Called by the provider with the response's usage object (or None)."""
        self._reported = usage if isinstance(usage, dict) else None

    def record(self, prompt_est: float, completion_est: float, seconds: float, ok: bool = True) -> None:
        u, self._reported = self._reported, None
        if u and isinstance(u.get("prompt_tokens"), int) and isinstance(u.get("completion_tokens"), int):
            p, c, basis = u["prompt_tokens"], u["completion_tokens"], "provider"
        else:
            p, c, basis = round(prompt_est), round(completion_est), "estimated"
        self.calls.append({"input": p, "output": c, "basis": basis, "seconds": round(seconds, 3), "ok": ok})

    def summary(self) -> dict:
        bases = {c["basis"] for c in self.calls}
        inp, out = sum(c["input"] for c in self.calls), sum(c["output"] for c in self.calls)
        return {"model_calls": len(self.calls), "failed_calls": sum(not c["ok"] for c in self.calls),
                "input_tokens": inp, "output_tokens": out, "total_tokens": inp + out,
                "basis": "none" if not bases else bases.pop() if len(bases) == 1 else "mixed",
                "model_ms": round(sum(c["seconds"] for c in self.calls) * 1000),
                "latency_ms": round((time.monotonic() - self.started) * 1000)}


@contextmanager
def metering():
    meter = Meter()
    token = _current.set(meter)
    try:
        yield meter
    finally:
        _current.reset(token)


def current():
    return _current.get()


def record_call(messages, output: str, seconds: float, ok: bool = True, extra: str = "") -> None:
    meter = _current.get()
    if meter:
        meter.record(est_tokens(json.dumps(messages, ensure_ascii=False) + extra), est_tokens(output or ""), seconds, ok)
