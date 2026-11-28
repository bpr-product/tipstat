"""Stage 06 deterministic tests for /act: tool loop limits, memory context, honest failure. A real local HTTP server
plays the mock world (same GET /tools + POST /tools/<name> protocol); the model is scripted."""
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_stage02 import Base, chat, fact  # noqa: E402
from backend.source.service.ai import tool_loop  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": n, "description": n, "parameters": {"type": "object", "properties": {}}}}
         for n in ("contacts_lookup", "slack_send_message", "reminders_create")]


class World(BaseHTTPRequestHandler):
    calls: list = []
    result = {"ok": True}

    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send({"tools": TOOLS})

    def do_POST(self):
        args = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        World.calls.append((self.path.rsplit("/", 1)[-1], args))
        self._send({"result": World.result})


def tc(i, name, args):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}


class Script:
    """Scripted model: yields assistant messages in order and records every request it was sent."""
    def __init__(self, *msgs):
        self.msgs, self.requests = list(msgs), []

    def __call__(self, messages, tools, tool_choice, timeout):
        self.requests.append(([dict(m) for m in messages], tool_choice))
        m = self.msgs.pop(0) if self.msgs else {"role": "assistant", "content": "done"}
        if isinstance(m, Exception):
            raise m
        return m


class ActTests(Base):
    def setUp(self):
        super().setUp()
        World.calls, World.result = [], {"ok": True}
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), World)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.addCleanup(self.srv.shutdown)
        self.world = {"tools_url": f"http://127.0.0.1:{self.srv.server_port}/tools", "mcp_url": None}

    def system(self, *msgs):
        m = self.mem()
        m.chat = Script(*msgs)
        return m

    def test_runs_tools_in_order_against_the_world_and_returns_the_models_reply(self):
        m = self.system({"role": "assistant", "content": None, "tool_calls": [tc(1, "contacts_lookup", {"query": "kavya"})]},
                        {"role": "assistant", "content": None, "tool_calls": [tc(2, "slack_send_message", {"to": "@kavya", "text": "15 min late"})]},
                        {"role": "assistant", "content": "Told Kavya Rao you'll be 15 minutes late."})
        out = m.act("Tell Kavya I'll be 15 mins late", self.world)
        self.assertEqual([c[0] for c in World.calls], ["contacts_lookup", "slack_send_message"])
        self.assertEqual(World.calls[1][1]["to"], "@kavya")
        self.assertEqual(out["reply"], "Told Kavya Rao you'll be 15 minutes late.")
        self.assertEqual([t["name"] for t in out["trace"] if t["step"] == "tool"], ["contacts_lookup", "slack_send_message"])
        # the tool result went back to the model as a tool message
        self.assertEqual(m.chat.requests[1][0][-1]["role"], "tool")

    def test_prompt_carries_standing_rules_current_facts_and_labels_outdated_versions(self):
        m = self.mem({"facts": [fact("user", "Nikhil messaging", "never after 9 pm", "never message Nikhil after 9 pm", "RULE")]},
                     {"facts": [fact("user", "dentist", "Thu 5 pm", "dentist for Thursday 5 pm", "EVENT")]},
                     {"facts": [dict(fact("user", "dentist", "Fri 6 pm", "dentist moved to Friday 6 pm", "EVENT"), replaces=2)]})
        m.add_message(chat("m1", "Note for future: never message Nikhil after 9 pm, he hates late pings."))
        m.add_message(chat("m2", "Booked the dentist for Thursday 5 pm, remember it."))
        m.add_message(chat("m3", "Dentist moved to Friday 6 pm, the dentist called."))
        m.chat = Script({"role": "assistant", "content": "ok"})
        m.act("Move my dentist appointment to Monday", self.world)
        user = m.chat.requests[0][0][1]["content"]
        self.assertIn("STANDING RULE", user)
        self.assertIn("never message Nikhil after 9 pm", user)
        self.assertIn("(CURRENT;", user)
        self.assertIn("OUTDATED - replaced", user)
        self.assertIn("REQUEST: Move my dentist appointment to Monday", user)
        self.assertIn("2026-11-02", m.chat.requests[0][0][0]["content"])  # "now" is the latest message time

    def test_unknown_tool_and_bad_arguments_are_reported_to_the_model_not_executed(self):
        m = self.system({"role": "assistant", "content": None, "tool_calls": [tc(1, "format_disk", {}), tc(2, "contacts_lookup", "{not json")]},
                        {"role": "assistant", "content": "I could not do that."})
        m.act("do something", self.world)
        self.assertEqual(World.calls, [])
        results = [x["content"] for x in m.chat.requests[1][0] if x["role"] == "tool"]
        self.assertTrue(any("unknown tool" in r for r in results) and any("JSON object" in r for r in results))

    def test_unreachable_world_changes_nothing_and_says_so_without_calling_the_model(self):
        m = self.system()
        out = m.act("Tell Kavya hi", {"tools_url": "http://127.0.0.1:9/tools"})
        self.assertEqual(m.chat.requests, [])
        self.assertIn("haven't changed anything", out["reply"])

    def test_endless_tool_calling_stops_within_the_model_call_budget_with_an_honest_reply(self):
        loop = {"role": "assistant", "content": None, "tool_calls": [tc(1, "contacts_lookup", {"query": "x"})]}
        m = self.system(*[loop] * 40)
        out = m.act("loop forever", self.world)
        self.assertLessEqual(len(m.chat.requests), 12)
        self.assertEqual(m.chat.requests[-1][1], "none")  # the last allowed call cannot ask for more tools
        self.assertIn("ran out", out["reply"])

    def test_every_call_stays_under_the_per_call_cap_even_with_huge_tool_results(self):
        World.result = {"rows": "x" * 5000}
        loop = {"role": "assistant", "content": None, "tool_calls": [tc(1, "contacts_lookup", {"query": "x"})]}
        m = self.system(*[loop] * 8, {"role": "assistant", "content": "done"})
        m.act("search lots", self.world)
        costs = [tool_loop._cost(msgs, TOOLS) for msgs, _ in m.chat.requests]
        self.assertTrue(all(c <= tool_loop.PER_CALL_CAP for c in costs), costs)
        self.assertLessEqual(sum(costs), tool_loop.TOTAL_CAP)

    def test_model_failure_gives_an_honest_reply_and_no_exception(self):
        m = self.system({"role": "assistant", "content": None, "tool_calls": [tc(1, "reminders_create", {"text": "gift"})]},
                        TimeoutError("slow"))
        out = m.act("Remind me about the gift", self.world)
        self.assertEqual([c[0] for c in World.calls], ["reminders_create"])
        self.assertIn("please check", out["reply"])

    def test_upcoming_commitments_reach_the_prompt_even_with_no_word_overlap(self):
        flight = dict(fact("user", "flight booking", "IndiGo 6E 2134 from BLR", "Booked my flight, IndiGo 6E 2134 from BLR", "EVENT"), when="2026-11-13")
        past = dict(fact("user", "old trip", "Goa", "I went on the Goa trip", "EVENT"), when="2026-10-01")
        far = dict(fact("user", "conference", "Berlin", "The Berlin conference is booked", "EVENT"), when="2027-03-01")
        m = self.mem({"facts": [flight, past, far]})
        m.add_message(chat("m1", "Booked my flight, IndiGo 6E 2134 from BLR. I went on the Goa trip. The Berlin conference is booked.", ts="2026-11-06T17:59:00+05:30"))
        m.add_message({"id": "m2", "ts": "2026-11-06T18:00:00+05:30", "type": "event", "source": "assistant", "op": "add_event",
                       "data": {"id": "evt_x", "title": "Dentist", "start": "2026-11-09T18:00:00+05:30"}})
        m.chat = Script({"role": "assistant", "content": "ok"})
        m.act("Block my gym sessions for next week", self.world)
        user = m.chat.requests[0][0][1]["content"]
        self.assertIn("UPCOMING - may not be on the calendar", user)
        self.assertIn("IndiGo 6E 2134", user)
        self.assertNotIn("Goa", user)          # already past
        self.assertNotIn("Berlin", user)       # beyond the 30-day horizon
        self.assertNotIn("UPCOMING - may not be on the calendar) user | Dentist", user)  # calendar entities are not repeated

    def test_act_before_any_message_has_no_upcoming_section_and_does_not_crash(self):
        m = self.system({"role": "assistant", "content": "ok"})
        m.act("Remind me about the gift", self.world)
        self.assertNotIn("UPCOMING", m.chat.requests[0][0][1]["content"])

    def test_budget_nudge_is_sent_once_when_calls_run_low(self):
        loop = {"role": "assistant", "content": None, "tool_calls": [tc(1, "contacts_lookup", {"query": "x"})]}
        m = self.system(*[loop] * 40)
        m.act("keep browsing", self.world)
        nudges = [x for msgs, _ in m.chat.requests[-1:] for x in msgs if x["role"] == "user" and x["content"].startswith("Budget:")]
        self.assertEqual(len(nudges), 1)
        first = next(i for i, (msgs, _) in enumerate(m.chat.requests) if any(x["content"].startswith("Budget:") for x in msgs if x["role"] == "user"))
        self.assertGreaterEqual(first, 7)  # not before the last few calls

    def test_act_never_writes_to_memory(self):
        m = self.system({"role": "assistant", "content": "nothing to do"})
        m.add_message(chat("m1", "hello there, nice weather today"))  # no model reply queued: extraction fails, source kept
        before = (self.n(m, "facts"), self.n(m, "sources"), self.n(m, "evidence"))
        m.act("What a day", self.world)
        self.assertEqual(before, (self.n(m, "facts"), self.n(m, "sources"), self.n(m, "evidence")))


if __name__ == "__main__":
    unittest.main()
