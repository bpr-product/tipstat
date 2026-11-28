#!/usr/bin/env python3
"""One-command smoke test: checks that the mock world, harness, starter and
(optionally) the LLM path all work on this machine. Run it from the repo root.

  python3 scripts/smoke_test.py                 # offline: world + MCP + harness + starter + checker
  python3 scripts/smoke_test.py --llm           # also proxy + reference agent + LLM judge (needs .env key)
  python3 scripts/smoke_test.py --load          # also a 40-tenant load test on the world
  python3 scripts/smoke_test.py --world-url http://192.168.1.20:8800 --tenant alice --token <tenant-token>
                                                # check a world hosted on another laptop (run from a participant laptop)

Exit code 0 = everything passed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PERSONA = ROOT / "data" / "dev" / "persona_a"
STARTER = ROOT / "starter" if (ROOT / "starter" / "server.py").exists() else ROOT   # organizer repo vs participant kit
PY = sys.executable
PROCS: list[subprocess.Popen] = []
RESULTS: list[tuple[str, bool, str]] = []


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def start(args, env=None, cwd=ROOT, log=None) -> subprocess.Popen:
    out = open(log, "w") if log else subprocess.DEVNULL
    p = subprocess.Popen(args, cwd=cwd, env={**os.environ, **(env or {})}, stdout=out, stderr=subprocess.STDOUT)
    PROCS.append(p)
    return p


def stop_all():
    for p in PROCS:
        if p.poll() is None:
            p.terminate()
    for p in PROCS:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


def http(url, body=None, headers=None, timeout=20):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET",
                                 headers={"Content-Type": "application/json",
                                          "Accept": "application/json, text/event-stream", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return r.status, (json.loads(raw) if raw else {})


def wait_up(url, secs=20) -> bool:
    end = time.time() + secs
    while time.time() < end:
        try:
            http(url, timeout=2)
            return True
        except Exception:
            time.sleep(0.3)
    return False


def check(name):
    def deco(fn):
        def run(*a, **k):
            try:
                detail = fn(*a, **k) or ""
                RESULTS.append((name, True, str(detail)))
                print(f"  PASS  {name}  {detail}")
                return True
            except Exception as e:
                RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
                print(f"  FAIL  {name}  {type(e).__name__}: {e}")
                return False
        return run
    return deco


# ---------------------------------------------------------------- world ----
def world_checks(base: str, admin_headers: dict, other_tenant_base: str | None):
    @check("world: health")
    def _h():
        _, out = http(f"{base}/health")
        return f"clock={out.get('clock')}"

    @check("world: REST tool list (29 tools, ?apps= filter)")
    def _t():
        _, out = http(f"{base}/tools")
        n = len(out["tools"])
        assert n == 29, f"expected 29 tools, got {n}"
        _, few = http(f"{base}/tools?apps=slack,calendar")
        assert 0 < len(few["tools"]) < n
        return f"{n} tools, {len(few['tools'])} for apps=slack,calendar"

    @check("world: email, docs, tasks, threads, RSVP on persona A")
    def _apps():
        _, em = http(f"{base}/tools/email_list", {"limit": 5})
        assert em["result"], "inbox is empty"
        _, th = http(f"{base}/tools/email_read_thread", {"thread_id": em["result"][0]["thread_id"]})
        _, tk = http(f"{base}/tools/tasks_list", {"limit": 5})
        _, tg = http(f"{base}/tools/tasks_get", {"key": tk["result"][0]["key"]})
        _, ds = http(f"{base}/tools/docs_search", {"query": "the"})
        _, dr = http(f"{base}/tools/docs_read", {"doc_id": ds["result"][0]["id"]})
        _, ev = http(f"{base}/tools/calendar_list_events", {"start": "2026-11-04T00:00:00+05:30",
                                                             "end": "2026-11-07T00:00:00+05:30"})
        _, rs = http(f"{base}/tools/calendar_respond", {"event_id": ev["result"][0]["id"], "response": "tentative"})
        _, ch = http(f"{base}/tools/slack_read_messages", {"conversation": "#eng", "limit": 50})
        assert len(ch["result"]) <= 50
        return (f"{len(em['result'])} emails, ticket {tg['result']['key']}, doc '{dr['result']['title'][:30]}', "
                f"rsvp {rs['result']['my_response']}")

    @check("world: tools refuse calls while locked (outside /act)")
    def _lock():
        http(f"{base}/admin/lock", {"locked": True}, admin_headers)
        try:
            http(f"{base}/tools/contacts_lookup", {"query": "kavya"})
            raise AssertionError("locked world answered a tool call")
        except urllib.error.HTTPError as e:
            assert e.code == 400, e.code
        finally:
            http(f"{base}/admin/lock", {"locked": False}, admin_headers)
        return "400 while locked, ok after unlock"

    @check("world: REST tool call")
    def _c():
        _, out = http(f"{base}/tools/contacts_lookup", {"query": "kavya"})
        names = [p["name"] for p in out["result"]]
        assert len(names) == 2, names
        return ", ".join(names)

    @check("world: MCP JSON-RPC (initialize, tools/list, tools/call)")
    def _m():
        _, init = http(f"{base}/mcp", {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                       "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                                  "clientInfo": {"name": "smoke", "version": "1"}}})
        _, lst = http(f"{base}/mcp", {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        _, call = http(f"{base}/mcp", {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                       "params": {"name": "slack_list_channels", "arguments": {}}})
        assert not call["result"]["isError"]
        return f"protocol {init['result']['protocolVersion']}, {len(lst['result']['tools'])} tools"

    @check("world: official MCP SDK client")
    def _sdk():
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
        except ImportError:
            return "skipped (pip install mcp to test with the SDK)"

        async def go():
            async with streamable_http_client(f"{base}/mcp") as streams:
                async with ClientSession(streams[0], streams[1]) as s:
                    await s.initialize()
                    r = await s.call_tool("calendar_find_free_slots", {"date": "2026-11-13", "duration_minutes": 30})
                    return r.content[0].text[:60]
        return asyncio.run(go())

    @check("world: admin load + audit log")
    def _a():
        _, out = http(f"{base}/admin/load", {"persona_dir": str(PERSONA), "date": "2026-11-03",
                                             "clock": "2026-11-10T09:00:00+05:30"}, admin_headers)
        http(f"{base}/tools/slack_send_message", {"to": "@kavya", "text": "smoke test"})
        _, aud = http(f"{base}/admin/audit?since=0", headers=admin_headers)
        assert aud["audit"] and aud["audit"][-1]["tool"] == "slack_send_message"
        return f"{out['counts']['events']} events, audit ok"

    for fn in (_h, _a, _t, _c, _apps, _lock, _m, _sdk):
        fn()

    if other_tenant_base:
        @check("world: tenants are isolated")
        def _iso():
            http(f"{other_tenant_base}/admin/load", {"persona_dir": str(PERSONA), "date": "2026-11-03"}, admin_headers)
            _, st = http(f"{other_tenant_base}/admin/state", headers=admin_headers)
            assert not any(m.get("text") == "smoke test" for m in st["messages"]), "message leaked across tenants"
            return "no leakage"
        _iso()


# ---------------------------------------------------------------- proxy ----
def proxy_checks(logs: Path):
    """The gateway's rules, against a --mock proxy: no key and no spend needed."""
    port = free_port()
    start([PY, "-m", "harness.proxy", "--port", str(port), "--host", "127.0.0.1", "--mock"], log=logs / "proxy_mock.log",
          env={"MAX_INPUT_TOKENS": "8000", "ALLOWED_MODELS": "deepseek/deepseek-v4.1-flash"})
    base = f"http://127.0.0.1:{port}"
    model = "deepseek/deepseek-v4.1-flash"

    def code(body, path="/v1/chat/completions"):
        try:
            http(base + path, body)
            return 200
        except urllib.error.HTTPError as e:
            return e.code

    def msg(text):
        return {"model": model, "messages": [{"role": "user", "content": text}]}

    @check("proxy: refuses route-around fields, images, n>1, other models")
    def _fields():
        assert wait_up(f"{base}/health"), "mock proxy did not start"
        cases = {"models": {**msg("hi"), "models": ["openai/gpt-5"]}, "plugins": {**msg("hi"), "plugins": [{"id": "web"}]},
                 "transforms": {**msg("hi"), "transforms": ["middle-out"]}, "n=2": {**msg("hi"), "n": 2},
                 "image": {"model": model, "messages": [{"role": "user", "content": [
                     {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]},
                 "model": {**msg("hi"), "model": "openai/gpt-5"}}
        got = {k: code(v) for k, v in cases.items()}
        bad = {k: c for k, c in got.items() if c not in (400, 403)}
        assert not bad, f"accepted: {bad}"
        assert code(msg("hello")) == 200
        return ", ".join(f"{k}={c}" for k, c in got.items())

    @check("proxy: 8K cap counts tool schemas, tool calls and non-ASCII text")
    def _cap():
        stuffed_tools = {**msg("hi"), "tools": [{"type": "function", "function": {
            "name": "memory", "description": "x " * 30000, "parameters": {"type": "object", "properties": {}}}}]}
        stuffed_args = {"model": model, "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function", "function": {
                "name": "note", "arguments": json.dumps({"memory": "y " * 30000})}}]},
            {"role": "tool", "tool_call_id": "1", "content": "ok"}]}
        cjk = msg("记忆" * 7000)          # 14K characters: ~3.5K tokens as chars/4, ~14K for real
        got = [code(stuffed_tools), code(stuffed_args), code(cjk)]
        assert got == [413, 413, 413], got
        return "tools 413, tool_calls 413, CJK 413"

    @check("proxy: per-window call and token budgets (429)")
    def _budget():
        http(f"{base}/calls/mark", {"budget": 2, "token_budget": 0, "phase": "answer", "label": "smoke"})
        a, b, c = code(msg("one")), code(msg("two")), code(msg("three"))
        assert (a, b, c) == (200, 200, 429), (a, b, c)
        http(f"{base}/calls/mark", {"budget": 0, "token_budget": 3000, "phase": "message", "label": "smoke"})
        d, e = code(msg("z " * 4000)), code(msg("z " * 4000))
        assert (d, e) == (200, 429), (d, e)
        http(f"{base}/calls/mark", {"budget": 0, "token_budget": 0})
        return "3rd call 429, token budget 429"

    for fn in (_fields, _cap, _budget):
        fn()



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", action="store_true", help="Also test proxy + reference agent + LLM judge (uses .env)")
    ap.add_argument("--load", action="store_true", help="Also run a 40-tenant load test on the world")
    ap.add_argument("--world-url", help="Test an already-running (e.g. hosted) world instead of starting one")
    ap.add_argument("--tenant", default=None, help="Tenant on the hosted world (default: smoke-<time>)")
    ap.add_argument("--token", default=os.environ.get("WORLD_ADMIN_TOKEN", ""), help="Admin or tenant token")
    a = ap.parse_args()
    os.chdir(ROOT)
    logs = Path(tempfile.mkdtemp(prefix="smoke_"))
    print(f"Smoke test (logs in {logs})\n")

    @check("python >= 3.10")
    def _py():
        assert sys.version_info >= (3, 10), sys.version
        return sys.version.split()[0]
    _py()

    @check("dev persona files present")
    def _files():
        need = ["persona.json", "stream.jsonl", "questions.jsonl", "actions.jsonl", "world/initial.json",
                "world/world_events.jsonl"]
        missing = [n for n in need if not (PERSONA / n).exists()]
        assert not missing, missing
        return "persona_a ok"
    _files()

    tenant = a.tenant or f"smoke-{int(time.time())}"
    hdr = {"X-Admin-Token": a.token} if a.token else {}
    if a.world_url:
        root = a.world_url.rstrip("/")
        print(f"\n[1] Hosted world at {root} (tenant {tenant})")
        world_root = root
    else:
        wport = free_port()
        start([PY, "-m", "world.server", "--port", str(wport), "--host", "127.0.0.1"], log=logs / "world.log",
              env={"WORLD_ADMIN_TOKEN": a.token} if a.token else None)
        world_root = f"http://127.0.0.1:{wport}"
        print(f"\n[1] Local world on {world_root}")
        if not wait_up(f"{world_root}/health"):
            print("  FAIL  world server did not start (see world.log)")
            return 1
    world_checks(f"{world_root}/t/{tenant}", hdr, None if a.world_url else f"{world_root}/t/{tenant}-other")
    print("\n[1b] LLM gateway rules (mock proxy, no key needed)")
    proxy_checks(logs)
    if a.world_url and a.token:
        @check("world: tenant token can't touch other tenants")
        def _scope():
            try:
                http(f"{world_root}/t/{tenant}-intruder-check/admin/state", headers=hdr)
            except urllib.error.HTTPError as e:
                assert e.code == 403, e.code
                return "403 as expected"
            return "accepted (you are using the MASTER token, which is fine for organizers)"
        _scope()

    print("\n[2] Starter system + harness (offline judge, questions + actions)")
    sport = free_port()
    start([PY, "server.py"], cwd=STARTER, env={"PORT": str(sport), "OPENAI_API_KEY": ""},
          log=logs / "starter.log")
    sys_url = f"http://127.0.0.1:{sport}"

    @check("starter: /health")
    def _sh():
        assert wait_up(f"{sys_url}/health"), "starter did not start (see starter.log)"
    _sh()

    @check("harness: full run on persona A")
    def _run():
        out = logs / "run_starter"
        env = {**os.environ, "WORLD_ADMIN_TOKEN": a.token} if a.token else None
        r = subprocess.run([PY, "-m", "harness.run", "--system", sys_url, "--data", str(PERSONA), "--judge", "offline",
                            "--world", f"{world_root}/t/{tenant}", "--out", str(out)],
                           capture_output=True, text=True, timeout=300, env=env)
        (logs / "run_starter.log").write_text(r.stdout + r.stderr)
        assert r.returncode == 0, f"exit {r.returncode}: {(r.stdout + r.stderr)[-300:]}"
        s = json.loads((out / "summary.json").read_text())
        assert s["n_questions"] == 20 and s["n_actions"] == 8, (s["n_questions"], s["n_actions"])
        assert s["errors"] == 0, f"{s['errors']} errors"
        errs = sum(1 for l in open(out / "graded.jsonl") if '"error": "' in l)
        assert errs == 0, f"{errs} item(s) failed with a system error"
        return (f"ran clean: {s['n_questions']} questions + {s['n_actions']} actions, 0 errors, "
                f"score {s['score']}")
    _run()

    @check("checker: starter passes except prompt_history")
    def _chk():
        r = subprocess.run([PY, "scripts/check_submission.py", str(STARTER), "--live", sys_url], capture_output=True,
                           text=True, timeout=200)
        fails = [l for l in r.stdout.splitlines() if l.startswith("FAIL")]
        assert all("prompt_history" in f for f in fails), fails
        return "ok (prompt_history is expected to be empty in the template)"
    _chk()

    if a.llm and not (ROOT / "reference" / "agent.py").exists():
        print("\n[3] LLM path skipped: the reference agent is only in the organizer toolkit")
    elif a.llm:
        print("\n[3] LLM path: proxy + reference agent + LLM judge")
        sys.path.insert(0, str(ROOT))
        from harness.common import load_env
        load_env()
        pport, rport = free_port(), free_port()

        @check(".env has OPENROUTER_API_KEY")
        def _key():
            k = os.environ.get("OPENROUTER_API_KEY", "")
            assert k.startswith("sk-"), "set OPENROUTER_API_KEY in .env"
            return f"allowed models: {os.environ.get('ALLOWED_MODELS', '(default)')}; judge: {os.environ.get('JUDGE_MODEL', '(default)')}"
        if _key():
            start([PY, "-m", "harness.proxy", "--port", str(pport), "--host", "127.0.0.1", "--log", str(logs / "llm_calls.jsonl")],
                  log=logs / "proxy.log")
            model = os.environ.get("ALLOWED_MODELS", "deepseek/deepseek-v4.1-flash").split(",")[0].strip()
            start([PY, str(STARTER / "server.py")], env={"PYTHONPATH": str(ROOT), "MEMORY_IMPL": "reference.agent:ReferenceAgent",
                                                  "OPENAI_BASE_URL": f"http://127.0.0.1:{pport}/v1", "OPENAI_API_KEY": "x",
                                                  "MODEL": model, "PORT": str(rport)}, log=logs / "reference.log")

            @check("proxy blocks a disallowed model")
            def _blk():
                assert wait_up(f"http://127.0.0.1:{pport}/health"), "proxy did not start"
                try:
                    http(f"http://127.0.0.1:{pport}/v1/chat/completions",
                         {"model": "openai/not-allowed", "messages": [{"role": "user", "content": "hi"}]})
                except urllib.error.HTTPError as e:
                    assert e.code == 403
                    return "403 as expected"
                raise AssertionError("disallowed model was accepted")
            _blk()

            @check("reference agent: 5 questions + 5 actions with LLM judge")
            def _ref():
                assert wait_up(f"http://127.0.0.1:{rport}/health"), "reference agent did not start"
                out = logs / "run_reference"
                r = subprocess.run([PY, "-m", "harness.run", "--system", f"http://127.0.0.1:{rport}", "--data", str(PERSONA),
                                    "--judge", "llm", "--votes", "1", "--limit", "5", "--proxy", f"http://127.0.0.1:{pport}",
                                    "--world", f"{world_root}/t/{tenant}-ref", "--out", str(out)],
                                   capture_output=True, text=True, timeout=900,
                                   env={**os.environ, **({"WORLD_ADMIN_TOKEN": a.token} if a.token else {})})
                (logs / "run_reference.log").write_text(r.stdout + r.stderr)
                assert r.returncode == 0, (r.stdout + r.stderr)[-400:]
                s = json.loads((out / "summary.json").read_text())
                u = s.get("llm_usage") or {}
                assert u.get("calls", 0) > 0, "no LLM calls went through the proxy"
                judged = [json.loads(l) for l in open(out / "graded.jsonl")]
                assert all(g.get("judge") in ("llm", "n/a") for g in judged), "LLM judge did not run"
                return (f"score {s['score']}, {u.get('calls')} LLM calls, ${u.get('cost_usd', 0):.4f}, "
                        f"errors {s['errors']} -> see {out/'summary.md'}")
            _ref()

    if a.load:
        print("\n[4] Load test: 40 tenants x 50 calls, concurrent")

        @check("world: 40 concurrent tenants")
        def _load():
            if a.world_url:
                target = world_root
            else:
                lport = free_port()
                start([PY, "-m", "world.server", "--port", str(lport), "--host", "127.0.0.1", "--persona", str(PERSONA),
                       "--date", "2026-11-03"], log=logs / "world_load.log")
                target = f"http://127.0.0.1:{lport}"
                wait_up(f"{target}/health")
            r = subprocess.run([PY, "world/loadtest.py", "40", "50"], capture_output=True, text=True, timeout=300,
                               env={**os.environ, "WORLD_URL": target})
            line = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
            assert r.returncode == 0 and " 0 errors" in line, line
            return line
        _load()

    stop_all()
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{'ALL PASSED' if not failed else f'{len(failed)} FAILED'}: {len(RESULTS) - len(failed)}/{len(RESULTS)} checks. Logs: {logs}")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        stop_all()
