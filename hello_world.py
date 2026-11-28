"""Connectivity check for the mock world. Run this first.

  python hello_world.py http://localhost:8800/t/<your-name>

It lists the tools and calls one, both over plain HTTP and over MCP (if the `mcp`
package is installed). If both print results, your setup is good.
"""
import json
import sys
import urllib.request

base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8800").rstrip("/")


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


# 1) Plain HTTP: GET /tools gives OpenAI function-calling schemas; POST /tools/<name> calls one.
tools = json.loads(urllib.request.urlopen(f"{base}/tools", timeout=10).read())["tools"]
print(f"HTTP: {len(tools)} tools:", ", ".join(t["function"]["name"] for t in tools))
print("HTTP: contacts_lookup('kavya') ->", post(f"{base}/tools/contacts_lookup", {"query": "kavya"})["result"])

# 2) MCP (Streamable HTTP) with the official SDK: pip install mcp
try:
    import asyncio
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async def mcp_demo():
        async with streamable_http_client(f"{base}/mcp") as streams:
            async with ClientSession(streams[0], streams[1]) as s:
                await s.initialize()
                listed = await s.list_tools()
                res = await s.call_tool("calendar_list_events", {"start": "2026-11-02T00:00:00+05:30",
                                                                  "end": "2026-11-03T00:00:00+05:30"})
                print(f"MCP: {len(listed.tools)} tools; calendar_list_events ->", res.content[0].text[:200])

    asyncio.run(mcp_demo())
except ImportError:
    print("MCP: `pip install mcp` to try the MCP endpoint (optional; plain HTTP works too)")
