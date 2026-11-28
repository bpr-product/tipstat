"""HTTP wrapper around your MemorySystem. You normally don't need to edit this file.

  python server.py                      # serves memory.py:MemorySystem on :8000
  MEMORY_IMPL=my_module:MyMemory python server.py

Endpoints (see SPEC.md):
  GET  /health
  POST /reset    {}
  POST /message  {"message": {...}}                          one message at a time, in order
  POST /answer   {"question"}                             -> {"answer", "evidence", "trace"}
  POST /act      {"request", "mcp_url", "tools_url"}      -> {"reply", "trace"}
"""
import importlib
import json
import os
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Local convenience: read KEY=VALUE lines from .env (never overrides variables already set,
# so the evaluation's OPENAI_BASE_URL always wins). .env is git-ignored - never commit it.
if os.path.exists(".env"):
    for _line in open(".env", encoding="utf-8"):
        _k, _sep, _v = _line.strip().partition("=")
        if _sep and _k and not _k.startswith("#"):
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

mod_name, cls_name = os.environ.get("MEMORY_IMPL", "memory:MemorySystem").split(":")
SYSTEM = getattr(importlib.import_module(mod_name), cls_name)()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if os.environ.get("VERBOSE"):
            super().log_message(fmt, *args)

    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/reset":
                SYSTEM.reset()
                return self._send(200, {"ok": True})
            if self.path == "/message":
                SYSTEM.add_message(req["message"])
                return self._send(200, {"ok": True})
            if self.path == "/answer":
                out = SYSTEM.answer(req["question"])
                if isinstance(out, str):
                    out = {"answer": out}
                return self._send(200, {"answer": out.get("answer", ""), "evidence": out.get("evidence", []),
                                        "trace": out.get("trace", [])})
            if self.path == "/act":
                out = SYSTEM.act(req["request"], {"mcp_url": req.get("mcp_url"), "tools_url": req.get("tools_url")})
                if isinstance(out, str):
                    out = {"reply": out}
                return self._send(200, {"reply": out.get("reply", ""), "trace": out.get("trace", [])})
            self._send(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"Memory service ({mod_name}:{cls_name}) on http://0.0.0.0:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
