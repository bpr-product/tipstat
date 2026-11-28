"""Load test for the mock world: N concurrent tenants x M calls (mixed MCP + REST).

  python -m world.server --port 8800 --persona data/dev/persona_a --date 2026-11-03 &
  python world/loadtest.py 40 100          # WORLD_URL=http://host:8800 to target another server
"""
import json, time, threading, urllib.request, statistics, sys
import os
BASE=os.environ.get("WORLD_URL","http://localhost:8800").rstrip("/"); N=int(sys.argv[1]); CALLS=int(sys.argv[2])
lat=[]; errs=[0]; L=threading.Lock()
def post(url, body):
    r=urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type":"application/json","Accept":"application/json, text/event-stream"})
    t=time.time(); urllib.request.urlopen(r, timeout=30).read(); return time.time()-t
def user(i):
    t=f"{BASE}/t/load{i}"
    for k in range(CALLS):
        try:
            if k%2:
                d=post(f"{t}/mcp", {"jsonrpc":"2.0","id":k,"method":"tools/call","params":{"name":"calendar_list_events","arguments":{"start":"2026-11-02T00:00:00+05:30","end":"2026-11-14T00:00:00+05:30"}}})
            elif k%5==0:
                d=post(f"{t}/tools/reminders_create", {"text":f"load {k}","due":"2027-01-01T09:00:00+05:30"})
            else:
                d=post(f"{t}/tools/slack_search_messages", {"query":"migration"})
            with L: lat.append(d)
        except Exception as e:
            with L: errs[0]+=1
t0=time.time(); th=[threading.Thread(target=user,args=(i,)) for i in range(N)]
[x.start() for x in th]; [x.join() for x in th]; el=time.time()-t0
lat.sort(); print(f"{N} clients x {CALLS} calls: {len(lat)} ok, {errs[0]} errors in {el:.1f}s -> {len(lat)/el:.0f} req/s; p50 {lat[len(lat)//2]*1000:.1f} ms, p95 {lat[int(len(lat)*.95)]*1000:.1f} ms, p99 {lat[int(len(lat)*.99)]*1000:.1f} ms")
