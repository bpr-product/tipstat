"""Product experience tests: measured usage, idempotent sends, receipts, Ask details, memory inspection, index status,
schema compatibility, short messages, and prompts free of evaluation examples."""
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_stage02 import FakeLLM, chat, fact  # noqa: E402
from test_core import FakeEmbedder  # noqa: E402
from backend.source.api.memory_system import MemorySystem  # noqa: E402
from backend.source.dal.db_session import Database  # noqa: E402
from backend.source.service.projects.workspace import Workspace, WorkspaceError  # noqa: E402
from backend.source.utils import metrics  # noqa: E402

LOCKER = fact("user", "gym locker number", "214", "my gym locker number is 214")


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def ws(self, *replies, embedder=None, llm=None):
        self.llm = llm or FakeLLM(*replies)
        w = Workspace(os.path.join(self.dir, "product"), llm_fn=self.llm, embedder=embedder)
        self.pid = w.list_projects()[0]["id"]
        self.cid = w.new_chat(self.pid)["id"]
        return w

    def send(self, w, mode, text, **kw):
        return w.send(self.pid, self.cid, mode, text, **kw)["assistant"]


class UsageTests(Base):
    def test_usage_counts_the_real_model_calls_and_never_adds_one(self):
        w = self.ws({"facts": [LOCKER]}, {"answer": "214", "know": True, "used": ["F1"]})
        tell = self.send(w, "tell", "Btw my gym locker number is 214, I keep forgetting it.")["payload"]["usage"]
        ask = self.send(w, "ask", "What is my gym locker number?")["payload"]["usage"]
        self.assertEqual((tell["model_calls"], ask["model_calls"]), (1, 2))                 # extraction; keywords + answer
        self.assertEqual(len(self.llm.prompts) + len(self.llm.expansions), 3)               # exactly the calls the engine made
        for u in (tell, ask):
            self.assertEqual((u["basis"], u["total_tokens"] == u["input_tokens"] + u["output_tokens"], u["total_tokens"] > 0, u["latency_ms"] >= 0),
                             ("estimated", True, True, True))

    def test_provider_reported_numbers_win_over_estimates_and_a_mix_is_labelled(self):
        def llm(messages, **kw):
            if kw.get("max_tokens") == 900:                      # extraction call: the provider reports its usage
                metrics.current().report({"prompt_tokens": 123, "completion_tokens": 45})
                return '{"facts": []}'
            return '{"answer": "x", "know": false, "used": []}'  # other calls report nothing
        w = self.ws(llm=llm)
        usage = self.send(w, "tell", "Btw my gym locker number is 214, I keep forgetting it.")["payload"]["usage"]
        self.assertEqual((usage["basis"], usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]), ("provider", 123, 45, 168))
        self.send(w, "tell", "Another note about something worth keeping for later.")
        meter = metrics.Meter()
        meter.report({"prompt_tokens": 10, "completion_tokens": 2})
        meter.record(99, 99, 0.1)
        meter.record(50, 5, 0.1)
        self.assertEqual((meter.summary()["basis"], meter.summary()["input_tokens"]), ("mixed", 60))

    def test_usage_is_stored_with_the_conversation_turn_and_a_failed_call_is_counted(self):
        w = self.ws(RuntimeError("model down"))
        out = self.send(w, "tell", "Btw my gym locker number is 214, I keep forgetting it.")
        self.assertEqual((out["payload"]["usage"]["model_calls"], out["payload"]["usage"]["failed_calls"]), (1, 1))
        stored = w.open_chat(self.pid, self.cid)["messages"][-1]["payload"]["usage"]
        self.assertEqual(stored["model_calls"], 1)

    def test_the_official_path_is_not_metered(self):
        m = MemorySystem(llm_fn=FakeLLM({"facts": []}), db_path=os.path.join(self.dir, "e.db"))
        m.add_message(chat("m1", "Btw my gym locker number is 214, I keep forgetting it."))
        self.assertIsNone(metrics.current())


class IdempotencyAndGuidanceTests(Base):
    def test_the_same_request_id_runs_once_and_returns_the_stored_turn(self):
        w = self.ws({"facts": [LOCKER]})
        first = w.send(self.pid, self.cid, "tell", "Btw my gym locker number is 214, I keep forgetting it.", request_id="req-1")
        again = w.send(self.pid, self.cid, "tell", "Btw my gym locker number is 214, I keep forgetting it.", request_id="req-1")
        self.assertTrue(again["replayed"])
        self.assertEqual((again["assistant"]["id"], len(self.llm.prompts), len(w.open_chat(self.pid, self.cid)["messages"])), (first["assistant"]["id"], 1, 2))
        self.assertEqual(len(w.memory(self.pid)["facts"]), 1)

    def test_a_question_typed_into_tell_is_guided_to_ask_without_storing_or_calling_the_model(self):
        w = self.ws()
        out = self.send(w, "tell", "What is my gym locker number?")
        self.assertEqual((out["payload"]["guidance"], len(self.llm.prompts), out["payload"]["usage"]["model_calls"]), ("ask", 0, 0))
        self.assertIn("use Ask", out["content"])
        self.assertEqual(w.memory(self.pid)["facts"], [])

    def test_a_statement_with_a_question_mark_inside_is_still_remembered(self):
        w = self.ws({"facts": [LOCKER]})
        out = self.send(w, "tell", "Remember? my gym locker number is 214 and that is that.")
        self.assertEqual((out["payload"].get("guidance"), len(out["payload"]["saved"])), (None, 1))

    def test_short_meaningful_messages_are_extracted_and_empty_acknowledgements_are_not(self):
        w = self.ws({"facts": [fact("user", "diet", "vegan", "I'm vegan", "STATE")]})
        saved = self.send(w, "tell", "I'm vegan.")["payload"]
        self.assertEqual((len(saved["saved"]), len(self.llm.prompts)), (1, 1))
        for ack in ["ok thanks", "Thanks!", "hi", "ok"]:
            self.send(w, "tell", ack)
        self.assertEqual(len(self.llm.prompts), 1)                                          # no model call for them


class ReceiptTests(Base):
    def test_an_update_appears_only_under_updated_with_the_real_before_and_after(self):
        old1 = fact("user", "preferred editor", "Emacs", "my preferred editor is Emacs", "PREFERENCE")
        old2 = fact("user", "gym locker number", "214", "my gym locker number is 214")
        new1 = dict(fact("user", "preferred editor", "VS Code", "my preferred editor is now VS Code", "PREFERENCE"), replaces=1)
        new2 = dict(fact("user", "gym locker number", "315", "my gym locker number is now 315"), replaces=2)
        w = self.ws({"facts": [old1, old2]}, {"facts": [new2, new1]})        # reply order differs from the old facts' order
        self.send(w, "tell", "My preferred editor is Emacs and my gym locker number is 214.")
        r = self.send(w, "tell", "Update: my preferred editor is now VS Code and my gym locker number is now 315.")["payload"]
        pairs = {(u["was"].split(": ")[1], u["now"].split(": ")[1]) for u in r["updated"]}
        self.assertEqual((r["saved"], pairs), ([], {("Emacs", "VS Code"), ("214", "315")}))     # not saved twice, not paired by position

    def test_the_forget_receipt_reports_the_actual_verification_result(self):
        job = fact("user", "job offer", "from Zentrix", "looking at a job offer from Zentrix")
        w = self.ws({"facts": [job]}, {"facts": []}, {"forget": True, "fact_ids": [1], "source_ids": []},
                    {"facts": []}, {"forget": True, "fact_ids": [1], "source_ids": []})
        self.send(w, "tell", "I have been looking at a job offer from Zentrix, keep it between us.")
        ok = self.send(w, "tell", "Please forget the job offer, delete it.")["payload"]
        self.assertEqual((ok["verification"], ok["forgotten"]), ({"verified": True, "problems": 0}, 1))
        self.assertIn("found nothing left", ok["headline"])

    def test_a_failed_verification_is_never_reported_as_success(self):
        job = fact("user", "job offer", "from Zentrix", "looking at a job offer from Zentrix")
        w = self.ws({"facts": [job]}, {"facts": []}, {"forget": True, "fact_ids": [1], "source_ids": []})
        self.send(w, "tell", "I have been looking at a job offer from Zentrix, keep it between us.")
        w.system(self.pid).repo.dangling = lambda: 2
        out = self.send(w, "tell", "Please forget the job offer, delete it.")
        self.assertEqual(out["payload"]["verification"], {"verified": False, "problems": 2})
        self.assertIn("2 problem(s)", out["content"])
        self.assertIn("not fully forgotten", out["content"])


class TitleRedactionTests(Base):
    def test_a_conversation_title_made_from_the_forgotten_message_is_redacted_too(self):
        job = fact("user", "job offer", "from Zentrix", "looking at a job offer from Zentrix")
        w = self.ws({"facts": [job]}, {"facts": []}, {"forget": True, "fact_ids": [1], "source_ids": []})
        self.send(w, "tell", "I have been looking at a job offer from Zentrix, keep it between us.")
        self.assertIn("job offer", w.list_chats(self.pid)[0]["title"])          # the auto-title carries the message
        self.send(w, "tell", "Please forget the job offer, delete it.")
        titles = " ".join(c["title"] for c in w.list_chats(self.pid))
        self.assertTrue("job offer" not in titles.lower() and "zentrix" not in titles.lower(), titles)
        self.assertEqual(w.system(self.pid).repo.chat_leftovers({"looking at a job offer from zentrix"}), 0)


class AskDetailsTests(Base):
    def dentist(self, embedder=None):
        w = self.ws({"facts": [dict(fact("user", "dentist appointment", "Dr Shah Friday 6 pm", "dentist moved to Friday 6 pm", "EVENT"), when="2026-11-06")]},
                    {"facts": [LOCKER]}, embedder=embedder)
        self.send(w, "tell", "Clinic called, dentist moved to Friday 6 pm.")
        self.send(w, "tell", "Btw my gym locker number is 214, I keep forgetting it.")
        return w

    def test_the_answer_details_show_how_each_candidate_was_found_what_was_supplied_and_what_was_cited(self):
        w = self.dentist(FakeEmbedder())
        w.system(self.pid).llm = self.llm = FakeLLM({"answer": "Friday 6 pm.", "know": True, "used": ["F1"]}, expand={"terms": ["dentist"]})
        out = self.send(w, "ask", "When is my teeth doctor visit?")["payload"]
        flow, cited = out["flow"], out["cited"]
        self.assertEqual(flow["expansion"]["terms"], ["dentist"])
        self.assertTrue(flow["semantic_used"] and flow["semantic_hits"] >= 1 and flow["lexical"] >= 1 and flow["fused"] >= 1)
        first = next(c for c in flow["candidates"] if c["ref"] == "F1")
        self.assertEqual((first["via"], first["status"], "dentist" in first["label"]), ("both", "active", True))
        self.assertTrue(any(s["ref"] == "F1" for s in flow["supplied"]))
        self.assertEqual((cited[0]["quote"], cited[0]["kind"], cited[0]["status"], cited[0]["message_id"][0], cited[0]["source_kind"]),
                         ("dentist moved to Friday 6 pm", "EVENT", "active", "u", "chat"))

    def test_a_follow_up_uses_the_previous_question_only_as_a_hint(self):
        w = self.dentist()
        w.system(self.pid).llm = self.llm = FakeLLM({"answer": "Dr Shah", "know": True, "used": ["F1"]}, {"answer": "x", "know": False, "used": []})
        self.send(w, "ask", "When is my dentist appointment?")
        self.send(w, "ask", "and with whom?")
        hinted = self.llm.prompts[1][1]["content"]
        self.assertIn("When is my dentist appointment?", hinted)
        self.assertIn("not evidence", hinted)
        self.assertNotIn("Dr Shah", self.llm.prompts[1][1]["content"].split("(This question follows")[1])   # the earlier answer is not passed on
        self.llm.replies.append({"answer": "x", "know": False, "used": []})
        self.send(w, "ask", "What is the current status of my gym locker number assignment at the club today?")
        self.assertNotIn("follows an earlier one", self.llm.prompts[-1][1]["content"])


class MemoryInspectionTests(Base):
    def test_one_memory_shows_its_evidence_source_versions_and_dependencies(self):
        old = fact("user", "dentist appointment", "Thu 5 pm", "dentist appointment Thursday 5 pm", "EVENT")
        derived = dict(fact("user", "dentist reminder", "4 pm", "reminder an hour before the dentist", "DERIVED"), derived_from=[1])
        new = dict(fact("user", "dentist appointment", "Fri 6 pm", "dentist appointment moved to Friday 6 pm", "EVENT"), replaces=1)
        w = self.ws({"facts": [old]}, {"facts": [derived]}, {"facts": [new]})
        self.send(w, "tell", "Booked the dentist appointment Thursday 5 pm.")
        self.send(w, "tell", "Set a reminder an hour before the dentist please.")
        self.send(w, "tell", "The dentist appointment moved to Friday 6 pm, clinic called.")
        head = w.memory_item(self.pid, 3)
        self.assertEqual([v["id"] for v in head["earlier_versions"]], [1])
        self.assertEqual((head["fact"]["status"], head["evidence"][0]["source_kind"], head["evidence"][0]["source_text"].startswith("USER: The dentist")), ("active", "chat", True))
        old_item = w.memory_item(self.pid, 1)
        self.assertEqual((old_item["replaced_by"], [d["id"] for d in old_item["derived_facts"]], old_item["fact"]["status"]), ([3], [2], "superseded"))
        self.assertEqual(w.memory_item(self.pid, 2)["derived_from"][0]["id"], 1)
        with self.assertRaises(WorkspaceError):
            w.memory_item(self.pid, 999)

    def test_the_timeline_snapshot_orders_versions_newest_first_and_marks_the_head(self):
        a = fact("user", "preferred editor", "Emacs", "my preferred editor is Emacs", "PREFERENCE")
        b = dict(fact("user", "preferred editor", "VS Code", "my preferred editor is now VS Code", "PREFERENCE"), replaces=1)
        c = dict(fact("user", "preferred editor", "Vim", "my preferred editor is now Vim", "PREFERENCE"), replaces=2)
        w = self.ws({"facts": [a]}, {"facts": [b]}, {"facts": [c]})
        for t in ["My preferred editor is Emacs.", "My preferred editor is now VS Code.", "My preferred editor is now Vim."]:
            self.send(w, "tell", t)
        facts = {f["id"]: f for f in w.memory(self.pid)["facts"]}
        self.assertEqual((facts[3]["versions"], facts[3]["is_head"], facts[1]["is_head"], facts[3]["status"]), ([2, 1], True, False, "active"))


class IndexStatusTests(Base):
    def test_system_info_reports_semantic_readiness_and_how_much_is_indexed_and_reindex_backfills(self):
        w = self.ws({"facts": [LOCKER]}, embedder=FakeEmbedder())
        self.send(w, "tell", "Btw my gym locker number is 214, I keep forgetting it.")
        info = w.system_info()
        self.assertTrue(info["semantic"]["ready"])
        self.assertEqual((info["index"][0]["facts"], info["index"][0]["facts_indexed"], info["index"][0]["sources_indexed"]), (1, 1, 1))
        w.system(self.pid).repo.db.execute("DELETE FROM embeddings")
        w.system(self.pid).repo.db.commit()
        self.assertEqual(w.system_info()["index"][0]["facts_indexed"], 0)
        done = w.reindex(self.pid)
        self.assertEqual((done["indexed_now"], done["still_missing"], done["facts_indexed"], done["sources_indexed"]), (2, 0, 1, 1))

    def test_when_the_model_is_missing_the_reason_is_shown_and_reindex_says_so(self):
        class Missing:
            available, error = False, "ModuleNotFoundError: No module named 'fastembed'"
            embed = embed_one = lambda self, *a: None
        w = self.ws({"facts": [LOCKER]}, embedder=Missing())
        info = w.system_info()
        self.assertEqual((info["semantic"]["ready"], "fastembed" in info["semantic"]["reason"]), (False, True))
        with self.assertRaises(WorkspaceError):
            w.reindex(self.pid)


class CompatibilityAndPromptTests(Base):
    def test_a_database_from_an_earlier_version_opens_and_keeps_working(self):
        path = os.path.join(self.dir, "old.db")
        con = sqlite3.connect(path)
        con.executescript("CREATE TABLE facts(id INTEGER PRIMARY KEY, kind TEXT, subject TEXT, relation TEXT, value TEXT, attribution TEXT,"
                          " when_date TEXT, quote TEXT, key TEXT, status TEXT NOT NULL DEFAULT 'active', predecessor_id INTEGER,"
                          " first_message_id TEXT NOT NULL, created_ts TEXT);"
                          "CREATE TABLE chat_messages(id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL, role TEXT NOT NULL,"
                          " mode TEXT, content TEXT NOT NULL, payload TEXT, ts TEXT);")
        con.commit()
        con.close()
        db = Database(path)
        for table, col in [("facts", "when_ts"), ("facts", "sensitive"), ("facts", "skey"), ("chat_messages", "request_id")]:
            self.assertIn(col, {r[1] for r in db.db.execute(f"PRAGMA table_info({table})")})
        m = MemorySystem(llm_fn=FakeLLM({"facts": [LOCKER]}), db_path=path)
        m.add_message(chat("m1", "Btw my gym locker number is 214, I keep forgetting it."))
        self.assertEqual(m.repo.db.execute("SELECT COUNT(*) FROM facts").fetchone()[0], 1)

    def test_runtime_prompts_contain_no_evaluation_examples(self):
        from backend.source.service.memory import acting, extraction_pipeline, forget_scope_search, hybrid_retrieval, retrieval
        prompts = " ".join([extraction_pipeline.EXTRACT_SYS, retrieval.ANSWER_SYS, acting.AGENT_SYS, hybrid_retrieval.EXPAND_SYS,
                            forget_scope_search.SELECT_SYS]).lower()
        for word in ["kavya", "nikhil", "zentrix", "dentist", "northline", "jaipur", "gym", "priya", "rohan", "arjun", "m006", "9 pm", "dinner", "locker"]:
            self.assertNotIn(word, prompts, word)


if __name__ == "__main__":
    unittest.main()
