"""Mock-world server: the same tools over MCP and over plain HTTP, plus admin
endpoints the harness uses to load state, move the clock and read the audit log.

  python -m world.server --port 8800 --persona data/dev/persona_a --date 2026-11-03

Every path can be prefixed with /t/<tenant> for an isolated world (e.g. one per
participant): /t/alice/mcp, /t/alice/tools, /t/alice/admin/load ...

Participant-facing
  POST /mcp                  MCP (Streamable HTTP, JSON responses): initialize, tools/list, tools/call
  GET  /tools[?apps=a,b]     tool schemas in OpenAI function-calling format (apps: slack, calendar,
                             email, docs, tasks, reminders, contacts; default all 29 tools)
  POST /tools/<name>         call a tool with a JSON object of arguments -> {"result"} or {"error"}
  GET  /health

Harness-only (need header X-Admin-Token = WORLD_ADMIN_TOKEN if that env var is set)
  POST /admin/load    {"persona_dir", "date", "clock"?}
  POST /admin/clock   {"clock"}
  GET  /admin/state
  GET  /admin/audit?since=N
  GET  /admin/mark    -> {"mark": last audit seq}
  POST /admin/lock    {"locked": true|false}  tools refuse calls while locked (outside /act)
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import threading
import uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .state import TOOLS, ToolError, World, mcp_tools, openai_tools

# One isolated World per tenant, so 40 participants (or 40 submissions under eval)
# can share one server without seeing each other's changes. A tenant is selected
# by URL prefix /t/<tenant>/... ; requests without a prefix use tenant "default".
WORLDS: dict[str, World] = {}
WORLDS_LOCK = threading.Lock()
DEFAULTS: dict = {"persona": None, "date": None, "clock": None, "max_tenants": 500}
TENANT_RE = re.compile(r"^/t/([A-Za-z0-9_.-]{1,64})(/.*)$")


def get_world(tenant: str) -> World:
    with WORLDS_LOCK:
        w = WORLDS.get(tenant)
        if w is None:
            if len(WORLDS) >= DEFAULTS["max_tenants"]:
                raise ToolError("too many tenants on this server")
            w = World()
            if DEFAULTS["persona"]:
                w.load(DEFAULTS["persona"], DEFAULTS["date"], DEFAULTS["clock"])
            WORLDS[tenant] = w
        return w


ADMIN_TOKEN = os.environ.get("WORLD_ADMIN_TOKEN", "")
DATA_ROOT = Path(os.environ.get("WORLD_DATA_ROOT", Path(__file__).resolve().parent.parent / "data" / "dev"))


def tenant_token(tenant: str) -> str:
    """Per-tenant admin token derived from the master token."""
    return hmac.new(ADMIN_TOKEN.encode(), tenant.encode(), hashlib.sha256).hexdigest()[:20] if ADMIN_TOKEN else ""
PROTOCOL_VERSIONS = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        if os.environ.get("VERBOSE"):
            super().log_message(*a)

    def _send(self, code: int, obj=None, headers: dict | None = None):
        body = b"" if obj is None else json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        if obj is not None:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        return json.loads(raw) if raw else {}

    def _resolve_persona(self, persona_dir: str) -> str:
        """Master token (or no token): any existing path. Tenant tokens: only personas
        under WORLD_DATA_ROOT (default data/dev), looked up by folder name, so a
        participant can't load the hidden test persona from a shared server."""
        p = Path(persona_dir)
        is_master = not ADMIN_TOKEN or hmac.compare_digest(self.headers.get("X-Admin-Token", ""), ADMIN_TOKEN)
        if is_master and (p / "world" / "initial.json").exists():
            return str(p)
        cand = DATA_ROOT / p.name
        if (cand / "world" / "initial.json").exists():
            return str(cand)
        raise ToolError(f"persona '{p.name}' not available on this server (looked in {DATA_ROOT})")

    def _admin_ok(self) -> bool:
        """With WORLD_ADMIN_TOKEN set, admin calls need either the master token or the
        tenant's own token (see --print-tokens), so participants on a shared server can
        reset/inspect only their own world."""
        if not ADMIN_TOKEN:
            return True
        tok = self.headers.get("X-Admin-Token", "")
        if hmac.compare_digest(tok, ADMIN_TOKEN) or hmac.compare_digest(tok, tenant_token(self.tenant)):
            return True
        self._send(403, {"error": "admin token required (X-Admin-Token)"})
        return False

    def _route(self):
        u = urlparse(self.path)
        m = TENANT_RE.match(u.path)
        tenant, path = (m.group(1), m.group(2)) if m else ("default", u.path)
        self.tenant = tenant
        try:
            self.world = get_world(tenant)
        except ToolError as e:
            self._send(503, {"error": str(e)})
            return None, None
        return u, path

    # ------------------------------------------------------------------ GET
    def do_GET(self):
        self._safe(self._get)

    def do_POST(self):
        self._safe(self._post)

    def _safe(self, fn):
        try:
            fn()
        except Exception as e:  # never drop the connection on a bug: return JSON 500
            try:
                self._send(500, {"error": f"internal error: {type(e).__name__}: {e}"})
            except Exception:
                pass

    def _get(self):
        u, path = self._route()
        if u is None:
            return
        WORLD = self.world
        if path == "/health":
            return self._send(200, {"ok": True, "clock": WORLD.clock})
        if path == "/tools":
            apps = parse_qs(u.query).get("apps", [None])[0]
            return self._send(200, {"tools": openai_tools(apps), "clock": WORLD.clock})
        if path == "/mcp":
            return self._send(405, {"error": "SSE stream not supported; POST JSON-RPC to /mcp"}, {"Allow": "POST"})
        if path.startswith("/admin/"):
            if not self._admin_ok():
                return
            if path == "/admin/state":
                return self._send(200, WORLD.snapshot())
            if path == "/admin/audit":
                since = int(parse_qs(u.query).get("since", ["0"])[0])
                return self._send(200, {"audit": WORLD.audit_since(since)})
            if path == "/admin/mark":
                return self._send(200, {"mark": WORLD.audit[-1]["seq"] if WORLD.audit else 0})
        self._send(404, {"error": "not found"})

    def do_DELETE(self):  # MCP session termination: sessions are stateless here
        self._send(200, {})

    # ----------------------------------------------------------------- POST
    def _post(self):
        u, path = self._route()
        if u is None:
            return
        WORLD = self.world
        try:
            body = self._body()
        except json.JSONDecodeError:
            return self._send(400, {"error": "invalid JSON"})
        if path == "/mcp":
            return self._mcp(body)
        if path.startswith("/tools/"):
            name = path.split("/", 2)[2]
            try:
                return self._send(200, {"result": WORLD.call(name, body)})
            except ToolError as e:
                return self._send(400, {"error": str(e)})
        if path.startswith("/admin/"):
            if not self._admin_ok():
                return
            try:
                if path == "/admin/load":
                    return self._send(200, WORLD.load(self._resolve_persona(body["persona_dir"]), body.get("date"),
                                                       body.get("clock"), body.get("until")))
                if path == "/admin/lock":
                    WORLD.locked = bool(body.get("locked", True))
                    return self._send(200, {"ok": True, "locked": WORLD.locked})
                if path == "/admin/clock":
                    WORLD.set_clock(body["clock"])
                    return self._send(200, {"ok": True, "clock": WORLD.clock})
            except (KeyError, ToolError, FileNotFoundError) as e:
                return self._send(400, {"error": str(e)})
        self._send(404, {"error": "not found"})

    # ------------------------------------------------------------------ MCP
    def _mcp(self, msg):
        if isinstance(msg, list):  # batch
            out = [r for r in (self._rpc(m) for m in msg) if r is not None]
            return self._send(200, out) if out else self._send(202)
        resp = self._rpc(msg)
        if resp is None:
            return self._send(202)
        headers = {}
        if msg.get("method") == "initialize":
            headers["Mcp-Session-Id"] = uuid.uuid4().hex
        self._send(200, resp, headers)

    def _rpc(self, m: dict):
        mid, method, params = m.get("id"), m.get("method", ""), m.get("params") or {}
        if mid is None:  # notification
            return None

        def ok(result):
            return {"jsonrpc": "2.0", "id": mid, "result": result}

        if method == "initialize":
            req = params.get("protocolVersion")
            ver = req if req in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[1]
            return ok({"protocolVersion": ver, "capabilities": {"tools": {"listChanged": False}},
                       "serverInfo": {"name": "hackathon-world", "version": "1.0.0"},
                       "instructions": "Mock Slack, Calendar, Email, Docs, Tasks, Reminders and Contacts "
                                       "for the user. Lists are capped at 50 and paged. "
                                       f"Current time: {self.world.clock}."})
        if method == "ping":
            return ok({})
        if method == "tools/list":
            return ok({"tools": mcp_tools((params or {}).get("apps"))})
        if method == "tools/call":
            name, args = params.get("name"), params.get("arguments") or {}
            if name not in TOOLS:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"unknown tool {name}"}}
            try:
                res = self.world.call(name, args)
                structured = res if isinstance(res, dict) else {"items": res}
                return ok({"content": [{"type": "text", "text": json.dumps(res, ensure_ascii=False)}],
                           "structuredContent": structured, "isError": False})
            except ToolError as e:
                return ok({"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True})
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256


def serve(port: int, host: str = "0.0.0.0", persona: str | None = None, date: str | None = None,
          clock: str | None = None) -> ThreadingHTTPServer:
    DEFAULTS.update({"persona": persona, "date": date, "clock": clock})
    if persona:
        get_world("default")
    return Server((host, port), Handler)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8800)
    p.add_argument("--persona", help="Persona folder to load (must contain world/initial.json)")
    p.add_argument("--date", help="Load world events up to this date (YYYY-MM-DD)")
    p.add_argument("--clock", help="Current time (ISO 8601)")
    p.add_argument("--print-tokens", nargs="+", metavar="TENANT",
                   help="Print each tenant's admin token (needs WORLD_ADMIN_TOKEN) and exit")
    a = p.parse_args(argv)
    if a.print_tokens:
        if not ADMIN_TOKEN:
            raise SystemExit("Set WORLD_ADMIN_TOKEN first")
        for t in a.print_tokens:
            print(f"{t}\t{tenant_token(t)}")
        return
    srv = serve(a.port, a.host, a.persona, a.date, a.clock)
    print(f"Mock world on http://{a.host}:{a.port}  (MCP at /mcp, REST at /tools; per-tenant: /t/<name>/mcp)"
          + ("  [admin token required]" if ADMIN_TOKEN else "  [admin OPEN - set WORLD_ADMIN_TOKEN when sharing]"))
    srv.serve_forever()


if __name__ == "__main__":
    main()
