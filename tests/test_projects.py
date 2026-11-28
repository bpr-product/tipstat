"""Product workspace tests: genuine project isolation (separate databases), persistent conversations, receipts built from
real before/after state, Forget reaching saved chats, and separation from the evaluation database."""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_stage02 import FakeLLM, chat, fact  # noqa: E402
from test_stage05 import everything  # noqa: E402
from test_core import FakeEmbedder  # noqa: E402
from backend.source.api.memory_system import MemorySystem  # noqa: E402
from backend.source.service.projects.workspace import Workspace, WorkspaceError  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("ui_serve", os.path.join(ROOT, "frontend", "serve.py"))
serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve)

JOB = fact("user", "job offer", "from Zentrix", "looking at a job offer from Zentrix", "STATE")


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def ws(self, *replies, embedder=None):
        self.llm = FakeLLM(*replies)
        return Workspace(os.path.join(self.dir, "product"), llm_fn=self.llm, embedder=embedder)

    def pids(self, w):
        return [p["id"] for p in w.list_projects()]


class ProjectTests(Base):
    def test_a_default_project_exists_and_names_are_validated_and_unique(self):
        w = self.ws()
        self.assertEqual([p["name"] for p in w.list_projects()], ["General"])
        p = w.create_project("Work")
        for bad in ["", "   ", "x" * 61, "work", "General"]:
            with self.assertRaises(WorkspaceError):
                w.create_project(bad)
        self.assertEqual(w.rename_project(p["id"], "Work 2")["name"], "Work 2")
        with self.assertRaises(WorkspaceError):
            w.rename_project(p["id"], "general")

    def test_unknown_or_hostile_project_ids_reach_nothing(self):
        w = self.ws()
        for pid in ["nope", "../registry", "p_zzzzzzzz", "p_00000000", "", "p_12345678/../../x"]:
            with self.assertRaises(WorkspaceError):
                w.list_chats(pid)
        self.assertEqual(serve.route(w, "GET", "/p/p_00000000/memory", {})[0], 404)

    def test_each_project_is_its_own_database_file(self):
        w = self.ws()
        a, b = self.pids(w)[0], w.create_project("Work")["id"]
        self.assertNotEqual(w.system(a).database.path, w.system(b).database.path)
        self.assertTrue(os.path.exists(w.system(b).database.path))

    def test_deleting_a_project_removes_its_file_conversations_and_memory_and_keeps_others(self):
        w = self.ws({"facts": [JOB]})
        a, b = self.pids(w)[0], w.create_project("Work")["id"]
        c = w.new_chat(b)["id"]
        w.send(b, c, "tell", "I have been looking at a job offer from Zentrix, keep it between us.")
        path = w.system(b).database.path
        w.delete_project(b)
        self.assertFalse(os.path.exists(path))
        self.assertEqual(self.pids(w), [a])
        with self.assertRaises(WorkspaceError):
            w.list_chats(b)

    def test_deleting_the_last_project_leaves_a_fresh_general(self):
        w = self.ws()
        w.delete_project(self.pids(w)[0])
        self.assertEqual([p["name"] for p in w.list_projects()], ["General"])


class IsolationTests(Base):
    def test_a_fact_told_in_one_project_is_invisible_to_ask_in_another_even_semantically(self):
        w = self.ws({"facts": [fact("user", "dentist appointment", "Friday 6 pm", "dentist moved to Friday 6 pm", "EVENT")]}, embedder=FakeEmbedder())
        a, b = self.pids(w)[0], w.create_project("Work")["id"]
        ca, cb = w.new_chat(a)["id"], w.new_chat(b)["id"]
        w.send(a, ca, "tell", "Clinic called, dentist moved to Friday 6 pm.")
        w.system(b).llm = self.llm = FakeLLM({"answer": "x", "know": False, "used": []})
        out = w.send(b, cb, "ask", "When is my dentist appointment, the teeth doctor?")["assistant"]
        self.assertEqual((out["payload"]["evidence"], len(self.llm.prompts)), ([], 0))   # nothing to retrieve, no model answer
        self.assertEqual(w.memory(b)["facts"], [])
        self.assertEqual(w.system(b).repo.embeddings(""), [])
        self.assertEqual(len(w.memory(a)["facts"]), 1)

    def test_the_same_message_ids_and_facts_in_two_projects_never_merge(self):
        w = self.ws({"facts": [fact("user", "gym locker number", "214", "my locker number is 214")]},
                    {"facts": [fact("user", "gym locker number", "999", "my locker number is 999")]})
        a, b = self.pids(w)[0], w.create_project("Work")["id"]
        w.system(a).add_message(chat("m1", "Btw my locker number is 214, I keep forgetting."))
        w.system(b).add_message(chat("m1", "Btw my locker number is 999, I keep forgetting."))     # same id, other project
        self.assertEqual([f["value"] for f in w.memory(a)["facts"]], ["214"])
        self.assertEqual([f["value"] for f in w.memory(b)["facts"]], ["999"])

    def test_forgetting_in_one_project_leaves_the_other_untouched_including_its_chats_and_hashes(self):
        w = self.ws({"facts": [JOB]}, {"facts": [JOB]}, {"facts": []}, {"forget": True, "fact_ids": [1], "source_ids": ["MID"]})
        a, b = self.pids(w)[0], w.create_project("Work")["id"]
        ca, cb = w.new_chat(a)["id"], w.new_chat(b)["id"]
        w.send(a, ca, "tell", "I have been looking at a job offer from Zentrix, keep it between us.")
        w.send(b, cb, "tell", "I have been looking at a job offer from Zentrix, keep it between us.")
        source = w.system(a).repo.db.execute("SELECT message_id FROM sources").fetchone()[0]
        self.llm.replies[-1] = {"forget": True, "fact_ids": [1], "source_ids": [source]}
        w.send(a, ca, "tell", "Please forget the job offer, delete it.")
        self.assertEqual(w.memory(a)["facts"], [])
        self.assertEqual(len(w.memory(b)["facts"]), 1)
        self.assertIn("Zentrix", w.open_chat(b, cb)["messages"][0]["content"])
        self.assertEqual(w.system(b).repo.tombstones(), set())
        self.assertNotEqual(w.system(a).repo.tombstones(), set())


class ConversationTests(Base):
    def test_conversations_persist_across_a_restart_and_are_titled_from_the_first_message(self):
        w = self.ws({"facts": [fact("user", "gym locker number", "214", "my locker number is 214")]})
        pid = self.pids(w)[0]
        cid = w.new_chat(pid)["id"]
        w.send(pid, cid, "tell", "Btw my locker number is 214, I keep forgetting it.")
        again = Workspace(os.path.join(self.dir, "product"), llm_fn=FakeLLM())
        chats = again.list_chats(pid)
        self.assertEqual((len(chats), chats[0]["title"], chats[0]["messages"]), (1, "Btw my locker number is 214, I keep forg", 2))
        self.assertEqual([m["role"] for m in again.open_chat(pid, cid)["messages"]], ["user", "assistant"])

    def test_memories_are_shared_across_chats_in_the_same_project(self):
        w = self.ws({"facts": [fact("user", "gym locker number", "214", "my locker number is 214")]})
        pid = self.pids(w)[0]
        c1, c2 = w.new_chat(pid)["id"], w.new_chat(pid)["id"]
        w.send(pid, c1, "tell", "Btw my locker number is 214, I keep forgetting it.")
        w.system(pid).llm = FakeLLM({"answer": "214", "know": True, "used": ["F1"]})
        out = w.send(pid, c2, "ask", "What is my gym locker number?")["assistant"]
        self.assertEqual([e["message_id"] for e in out["payload"]["evidence"]][0][0], "u")
        self.assertEqual(out["content"], "214")

    def test_rename_delete_and_clear_never_touch_memories(self):
        w = self.ws({"facts": [fact("user", "gym locker number", "214", "my locker number is 214")]})
        pid = self.pids(w)[0]
        c1, c2 = w.new_chat(pid)["id"], w.new_chat(pid)["id"]
        w.send(pid, c1, "tell", "Btw my locker number is 214, I keep forgetting it.")
        w.rename_chat(pid, c1, "Gym stuff")
        self.assertIn("Gym stuff", [c["title"] for c in w.list_chats(pid)])
        w.delete_chat(pid, c1)
        w.clear_chats(pid)
        self.assertEqual((w.list_chats(pid), len(w.memory(pid)["facts"])), ([], 1))
        with self.assertRaises(WorkspaceError):
            w.open_chat(pid, c2)

    def test_bad_input_is_rejected_before_anything_runs(self):
        w = self.ws()
        pid = self.pids(w)[0]
        cid = w.new_chat(pid)["id"]
        for mode, text, world in [("fly", "hi", ""), ("tell", "   ", ""), ("tell", "x" * 4001, ""), ("do", "send it", ""), ("do", "send it", "file:///etc")]:
            with self.assertRaises(WorkspaceError):
                w.send(pid, cid, mode, text, world)
        with self.assertRaises(WorkspaceError):
            w.send(pid, 999, "tell", "hello there my friend")
        self.assertEqual(w.open_chat(pid, cid)["messages"], [])


class ReceiptTests(Base):
    def test_a_tell_reports_what_was_really_saved_updated_and_already_known(self):
        w = self.ws({"facts": [fact("user", "preferred editor", "Emacs", "my preferred editor is Emacs", "PREFERENCE")]},
                    {"facts": [dict(fact("user", "preferred editor", "VS Code", "my preferred editor is now VS Code", "PREFERENCE"), replaces=1)]},
                    {"facts": [fact("user", "preferred editor", "VS Code", "my preferred editor is now VS Code", "PREFERENCE")]},
                    {"facts": []})
        pid = self.pids(w)[0]
        cid = w.new_chat(pid)["id"]
        first = w.send(pid, cid, "tell", "My preferred editor is Emacs, always has been.")["assistant"]["payload"]
        second = w.send(pid, cid, "tell", "Update: my preferred editor is now VS Code.")["assistant"]["payload"]
        third = w.send(pid, cid, "tell", "Yes my preferred editor is now VS Code, really.")["assistant"]["payload"]
        fourth = w.send(pid, cid, "tell", "Nice weather in the city today, honestly lovely.")["assistant"]["payload"]
        self.assertEqual(len(first["saved"]), 1)
        self.assertEqual((len(second["updated"]), second["updated"][0]["was"].endswith("Emacs"), second["updated"][0]["now"].endswith("VS Code")), (1, True, True))
        self.assertEqual((third["saved"], len(third["already_known"])), ([], 1))
        self.assertIn("No structured facts were extracted", fourth["headline"])

    def test_a_failed_model_call_is_reported_honestly_not_as_success(self):
        w = self.ws(RuntimeError("model down"))
        pid = self.pids(w)[0]
        out = w.send(pid, w.new_chat(pid)["id"], "tell", "My preferred editor is Emacs, always has been.")["assistant"]
        self.assertEqual(out["payload"]["extraction"], "failed")
        self.assertIn("no facts were saved", out["content"])

    def test_forget_removes_the_fact_redacts_every_saved_chat_that_carried_it_and_leaves_no_trace_in_the_file(self):
        w = self.ws({"facts": [JOB]}, {"facts": []}, {"forget": True, "fact_ids": [1], "source_ids": []})
        pid = self.pids(w)[0]
        cid = w.new_chat(pid)["id"]
        w.send(pid, cid, "tell", "I have been looking at a job offer from Zentrix, keep it between us.")
        w.system(pid).llm = FakeLLM({"answer": "You have a job offer from Zentrix.", "know": True, "used": ["F1"]})
        w.send(pid, cid, "ask", "Who made me a job offer?")
        w.system(pid).llm = FakeLLM({"facts": []}, {"forget": True, "fact_ids": [1], "source_ids": []})
        out = w.send(pid, cid, "tell", "Please forget the job offer, delete it.")["assistant"]
        self.assertIn("Forgot 1 fact", out["content"])
        msgs = w.open_chat(pid, cid)["messages"]
        self.assertTrue(all("Zentrix" not in m["content"] and "offer" not in str(m["payload"]) for m in msgs))
        self.assertEqual(sum("removed: this message contained" in m["content"] for m in msgs), 4)   # tell, its receipt, the question, the answer
        self.assertTrue(msgs[-2]["content"].startswith("(You asked me to forget"))                     # the request itself is not kept
        self.assertFalse(everything(w.system(pid), "zentrix"))
        self.assertEqual(w.system(pid).repo.dangling(), 0)


class MixedMessageForgetTests(Base):
    def test_forgetting_one_thing_from_a_message_that_told_two_keeps_the_independent_fact(self):
        # found by the real smoke test: one Tell held a job offer and a locker number; forgetting the offer erased both
        locker = fact("user", "gym locker number", "214", "my gym locker number is 214")
        w = self.ws({"facts": [JOB, locker]}, {"facts": []}, {"forget": True, "fact_ids": [1], "source_ids": ["SRC"]})
        pid = self.pids(w)[0]
        cid = w.new_chat(pid)["id"]
        w.send(pid, cid, "tell", "I have been looking at a job offer from Zentrix, keep it between us. Also my gym locker number is 214.")
        source = w.system(pid).repo.db.execute("SELECT message_id FROM sources").fetchone()[0]
        self.llm.replies[-1] = {"forget": True, "fact_ids": [1], "source_ids": [source]}
        out = w.send(pid, cid, "tell", "Please forget the job offer from Zentrix, delete it.")["assistant"]
        self.assertIn("Forgot 1 fact", out["content"])
        self.assertEqual([f["value"] for f in w.memory(pid)["facts"]], ["214"])
        self.assertFalse(everything(w.system(pid), "zentrix"))             # the raw message text is gone
        self.assertEqual(w.system(pid).repo.dangling(), 0)

    def test_everything_learned_from_a_chosen_source_still_goes_when_it_is_about_the_request(self):
        w = self.ws({"facts": [JOB]}, {"facts": []}, {"forget": True, "fact_ids": [], "source_ids": ["SRC"]})
        pid = self.pids(w)[0]
        cid = w.new_chat(pid)["id"]
        w.send(pid, cid, "tell", "I have been looking at a job offer from Zentrix, keep it between us.")
        source = w.system(pid).repo.db.execute("SELECT message_id FROM sources").fetchone()[0]
        self.llm.replies[-1] = {"forget": True, "fact_ids": [], "source_ids": [source]}
        w.send(pid, cid, "tell", "Please forget the job offer, delete it.")
        self.assertEqual(w.memory(pid)["facts"], [])


class EvaluationSeparationTests(Base):
    def test_the_official_reset_cannot_touch_the_product_workspace_and_cannot_see_it(self):
        w = self.ws({"facts": [fact("user", "gym locker number", "214", "my locker number is 214")]})
        pid = self.pids(w)[0]
        w.send(pid, w.new_chat(pid)["id"], "tell", "Btw my locker number is 214, I keep forgetting it.")
        evaluation = MemorySystem(llm_fn=FakeLLM({"facts": [fact("user", "gym locker number", "777", "my locker number is 777")]}),
                                  db_path=os.path.join(self.dir, "evaluation.db"))
        evaluation.add_message(chat("m1", "Btw my locker number is 777, I keep forgetting it."))
        self.assertEqual([f["value"] for f in w.memory(pid)["facts"]], ["214"])       # neither sees the other
        evaluation.reset()
        self.assertEqual(len(w.memory(pid)["facts"]), 1)                              # the official reset leaves the product alone
        self.assertEqual(evaluation.repo.db.execute("SELECT COUNT(*) FROM facts").fetchone()[0], 0)
        self.assertNotEqual(w.system(pid).database.path, evaluation.database.path)

    def test_reset_all_wipes_the_workspace_only(self):
        w = self.ws({"facts": [fact("user", "gym locker number", "214", "my locker number is 214")]})
        pid = self.pids(w)[0]
        w.send(pid, w.new_chat(pid)["id"], "tell", "Btw my locker number is 214, I keep forgetting it.")
        w.reset_all()
        new = self.pids(w)[0]
        self.assertEqual((w.memory(new)["facts"], w.list_chats(new)), ([], []))


class ApiTests(Base):
    def test_routes_return_clear_statuses_and_surface_model_failures_as_errors(self):
        w = self.ws(RuntimeError("down"))
        pid = self.pids(w)[0]
        code, chat_ = serve.route(w, "POST", f"/p/{pid}/chats", {})
        self.assertEqual(code, 200)
        self.assertEqual(serve.route(w, "POST", f"/p/{pid}/chats/{chat_['id']}/send", {"mode": "x", "text": "hi"})[0], 400)
        self.assertEqual(serve.route(w, "POST", f"/p/{pid}/chats/99/send", {"mode": "ask", "text": "hello there"})[0], 404)
        self.assertEqual(serve.route(w, "GET", "/p/nothing", {})[0], 404)
        w.system(pid).llm = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))
        w.system(pid).chat = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("tools down"))
        code, body = serve.route(w, "POST", f"/p/{pid}/chats/{chat_['id']}/send", {"mode": "do", "text": "send it", "world": "http://127.0.0.1:9/tools"})
        self.assertEqual(code, 200)       # unreachable tools: the job reports it did nothing, it does not crash
        self.assertIn("haven't changed anything", body["assistant"]["content"])

    def test_http_serves_the_page_and_the_json_api_on_loopback(self):
        w = self.ws()
        srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(w))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        base = f"http://127.0.0.1:{srv.server_port}"
        html = urllib.request.urlopen(base + "/", timeout=10).read().decode()
        self.assertIn("HUSH", html)
        projects = json.loads(urllib.request.urlopen(base + "/p/projects", timeout=10).read())["projects"]
        self.assertEqual([p["name"] for p in projects], ["General"])
        for bad in ["/serve.py", "/../memory.py", "/%2e%2e/.env"]:
            try:
                urllib.request.urlopen(base + bad, timeout=10)
                self.fail(bad)
            except urllib.error.HTTPError as e:
                self.assertEqual(e.code, 404)

    def test_page_script_has_valid_syntax(self):
        import subprocess
        r = subprocess.run(["node", "--check", os.path.join(ROOT, "frontend", "app.js")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
