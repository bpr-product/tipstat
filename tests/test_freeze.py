"""Pre-freeze correctness tests: source-only forget, dedupe vs privacy/attribution, a forget request is never a source,
assistant text is not evidence, receipts name retained sources, retrieval diagnostics tell the truth."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_stage02 import Base, FakeLLM, chat, fact  # noqa: E402
from test_stage05 import everything  # noqa: E402
from test_core import FakeEmbedder  # noqa: E402
from test_experience import Base as WsBase  # noqa: E402


def facts(m):
    return [dict(r) for r in m.repo.db.execute("SELECT * FROM facts ORDER BY id")]


class SourceOnlyForgetTests(WsBase):
    def test_forgetting_a_source_that_produced_no_fact_still_clears_chats_titles_and_arms_the_guard(self):
        w = self.ws({"facts": []}, {"facts": []}, {"forget": True, "fact_ids": [], "source_ids": ["SRC"]})
        self.send(w, "tell", "I have been looking at a job offer from Zentrix, keep it between us.")
        system = w.system(self.pid)
        src = system.repo.db.execute("SELECT message_id FROM sources").fetchone()[0]
        self.llm.replies[-1] = {"forget": True, "fact_ids": [], "source_ids": [src]}
        self.assertEqual(w.memory(self.pid)["facts"], [])                       # nothing was extracted: only the raw source exists
        self.assertIn("Zentrix", w.open_chat(self.pid, self.cid)["messages"][0]["content"])
        out = self.send(w, "tell", "Please forget the job offer, delete it.")
        self.assertIn("Forgot 0 fact(s) and erased 1 source(s)", out["content"])
        chat_ = w.open_chat(self.pid, self.cid)["messages"]
        self.assertFalse(any("zentrix" in (m["content"] + str(m["payload"])).lower() for m in chat_))
        self.assertNotIn("zentrix", " ".join(c["title"] for c in w.list_chats(self.pid)).lower())
        self.assertFalse(everything(system, "zentrix"))
        self.assertIn(system.repo.hash_term("u:zentrix"), system.repo.tombstones())     # later writes cannot mention it
        self.assertEqual(system.repo.dangling(), 0)


class DedupePrivacyTests(Base):
    def test_restating_a_fact_with_keep_it_private_makes_the_stored_fact_private_and_never_the_reverse(self):
        plain = fact("user", "salary", "50k", "my salary is 50k")
        private = dict(plain, sensitive=True, quote="my salary is 50k, keep it private")
        m = self.mem({"facts": [plain]}, {"facts": [private]}, {"facts": [plain]})
        m.add_message(chat("m1", "FYI my salary is 50k, for the record."))
        self.assertEqual(facts(m)[0]["sensitive"], 0)
        m.add_message(chat("m2", "Again my salary is 50k, keep it private please."))
        m.add_message(chat("m3", "Just noting my salary is 50k once more."))
        rows = facts(m)
        self.assertEqual((len(rows), rows[0]["sensitive"]), (1, 1))                        # one fact, now private, and it stays private

    def test_a_paraphrase_that_adds_a_privacy_request_also_makes_the_stored_fact_private(self):
        m = self.mem({"facts": [fact("user", "gym locker number", "214", "my gym locker number is 214")]},
                     {"facts": [dict(fact("user", "locker number", "214", "locker number is 214, between us"), sensitive=True)]})
        m.add_message(chat("m1", "Btw my gym locker number is 214, I keep forgetting it."))
        m.add_message(chat("m2", "Honestly the locker number is 214, between us only."))
        rows = facts(m)
        self.assertEqual((len(rows), rows[0]["sensitive"], self.n(m, "evidence")), (1, 1, 2))

    def test_the_same_statement_from_the_user_and_from_someone_else_stays_two_facts_with_their_own_attribution(self):
        about = fact("Kavya", "allergy", "peanuts", "Kavya is allergic to peanuts")
        m = self.mem({"facts": [dict(about, attribution="other", quote="Kavya is allergic to peanuts")]}, {"facts": [dict(about, attribution="user")]})
        m.add_message({"id": "e1", "ts": "2026-11-02T10:00:00+05:30", "type": "event", "source": "inbound", "op": "add_message",
                       "data": {"from": "@rohan", "text": "FYI Kavya is allergic to peanuts so be careful"}})
        m.add_message(chat("m2", "Kavya is allergic to peanuts, remember that when cooking."))
        self.assertEqual(sorted(r["attribution"] for r in facts(m)), ["other", "user"])


class ForgetRequestIsNotASourceTests(Base):
    def test_when_nothing_stored_matches_the_fact_the_request_itself_produced_is_removed(self):
        echo = fact("user", "salary", "50k", "my salary is 50k")
        m = self.mem({"facts": [echo]}, {"forget": True, "fact_ids": [], "source_ids": []})
        m.add_message(chat("m1", "Please forget that my salary is 50k, delete it."))
        self.assertEqual((facts(m), m.repo.source("m1")["text"]), ([], ""))
        self.assertFalse(any("50k" in str(r) for r in m.repo.db.execute("SELECT * FROM fts")))
        m.add_message(chat("m1", "Please forget that my salary is 50k, delete it."))        # replaying the id is still a no-op
        self.assertEqual(len(self.llm.prompts), 2)

    def test_other_sentences_of_the_same_request_message_are_kept(self):
        echo = fact("user", "salary", "50k", "my salary is 50k")
        keep = fact("user", "gym", "7 pm", "my gym is at 7 pm now")
        m = self.mem({"facts": [echo, keep]}, {"forget": True, "fact_ids": [], "source_ids": []})
        m.add_message(chat("m1", "Please forget that my salary is 50k. Also my gym is at 7 pm now."))
        self.assertEqual([r["value"] for r in facts(m)], ["7 pm"])

    def test_when_the_decision_fails_a_plain_forget_sentence_still_does_not_leave_its_content_behind(self):
        echo = fact("user", "salary", "50k", "my salary is 50k")
        m = self.mem({"facts": [echo]}, RuntimeError("down"))
        m.add_message(chat("m1", "Forget that my salary is 50k."))
        self.assertEqual((facts(m), m.repo.source("m1")["text"]), ([], ""))
        self.assertEqual(m.repo.db.execute("SELECT MIN(verified) FROM forget_log").fetchone()[0], 0)   # the receipt never claims success: the decision was unresolved

    def test_a_false_positive_wording_with_a_failed_decision_keeps_ordinary_facts(self):
        keep = fact("user", "dentist", "Thursday", "booked the dentist Thursday")
        m = self.mem({"facts": [keep]}, RuntimeError("down"))
        m.add_message(chat("m1", "I forgot my password and booked the dentist Thursday."))
        self.assertEqual([r["value"] for r in facts(m)], ["Thursday"])


class AssistantTextTests(Base):
    def test_assistant_lines_are_never_shown_to_ask_or_the_agent_as_evidence(self):
        m = self.mem({"facts": []})
        m.add_message(chat("m1", "Booked my train for Monday morning, early start.", assistant="Zebra: your flight is on Friday"))
        m.llm = llm = FakeLLM({"answer": "x", "know": False, "used": []})
        m.answer("What about my train on Monday?")
        prompt = llm.prompts[0][1]["content"]
        self.assertTrue("train for Monday" in prompt and "Zebra" not in prompt and "ASSISTANT" not in prompt)
        from backend.source.service.memory import act_context
        text, _, _ = act_context.build(m.repo, m.search, FakeLLM(), None, "Plan my Monday train")
        self.assertTrue("train for Monday" in text and "Zebra" not in text)

    def test_a_fact_whose_only_support_is_an_assistant_line_is_rejected(self):
        m = self.mem({"facts": [fact("user", "flight", "Friday", "your flight is on Friday")]})
        m.add_message(chat("m1", "What do you think about my week?", assistant="your flight is on Friday"))
        self.assertEqual(facts(m), [])


class ReceiptTests(WsBase):
    def test_receipts_say_whether_facts_were_extracted_and_that_the_raw_message_is_kept_searchable(self):
        w = self.ws({"facts": [fact("user", "gym locker number", "214", "my gym locker number is 214")]}, {"facts": []}, RuntimeError("down"))
        saved = self.send(w, "tell", "Btw my gym locker number is 214, I keep forgetting it.")
        none = self.send(w, "tell", "Nice weather in the city today, honestly lovely.")
        failed = self.send(w, "tell", "Another note about something worth keeping for later.")
        ack = self.send(w, "tell", "ok thanks")
        asked = self.send(w, "tell", "What is my gym locker number?")
        self.assertIn("also kept as a searchable source", saved["content"])
        self.assertIn("No structured facts were extracted", none["content"])
        self.assertIn("kept as a searchable source", none["content"])
        self.assertIn("no facts were saved. The message itself is kept as a searchable source", failed["content"])
        self.assertIn("kept as a searchable source, but no memory was saved", ack["content"])
        self.assertEqual([x["payload"]["source_kept"] for x in (saved, none, failed, ack, asked)], [True, True, True, True, False])

    def test_a_forget_request_reports_that_its_own_text_was_not_kept(self):
        job = fact("user", "job offer", "from Zentrix", "looking at a job offer from Zentrix")
        w = self.ws({"facts": [job]}, {"facts": []}, {"forget": True, "fact_ids": [1], "source_ids": []})
        self.send(w, "tell", "I have been looking at a job offer from Zentrix, keep it between us.")
        out = self.send(w, "tell", "Please forget the job offer, delete it.")
        self.assertFalse(out["payload"]["source_kept"])


class DiagnosticsTests(Base):
    def setup_memory(self, embedder):
        from backend.source.api.memory_system import MemorySystem
        self.llm = FakeLLM({"facts": [dict(fact("user", "dentist appointment", "Friday 6 pm", "dentist moved to Friday 6 pm", "EVENT"))]},
                           {"facts": [fact("user", "gym locker number", "214", "my gym locker number is 214")]})
        m = MemorySystem(llm_fn=self.llm, db_path=self.path, embedder=embedder)
        m.add_message(chat("m1", "Clinic called, dentist moved to Friday 6 pm."))
        m.add_message(chat("m2", "Btw my gym locker number is 214, I keep forgetting it."))
        return m

    def retrieve_step(self, m, question, **expand):
        m.llm = FakeLLM({"answer": "x", "know": False, "used": []}, **expand)
        return next(t for t in m.answer(question)["trace"] if t["step"] == "retrieve")

    def test_counts_cover_facts_and_sources_for_each_method_and_rrf_reports_its_own_ranking(self):
        step = self.retrieve_step(self.setup_memory(FakeEmbedder()), "When is my dentist appointment?")
        self.assertEqual((step["lexical"], step["semantic"], step["fused"], step["rrf_applied"]), (2, 2, 2, True))   # 1 fact + 1 source each
        self.assertEqual({c["ref"]: c["via"] for c in step["candidates"]}["F1"], "both")

    def test_a_semantic_only_candidate_is_labelled_semantic_and_rrf_is_applied(self):
        step = self.retrieve_step(self.setup_memory(FakeEmbedder()), "When is my teeth doctor visit?")
        self.assertEqual(({c["ref"]: c["via"] for c in step["candidates"]}["F1"], step["lexical"], step["rrf_applied"]), ("semantic", 0, True))

    def test_without_a_model_nothing_semantic_is_claimed(self):
        none = type("No", (), {"available": False, "embed": lambda s, t: None, "embed_one": lambda s, t: None})()
        step = self.retrieve_step(self.setup_memory(none), "When is my dentist appointment?")
        self.assertEqual((step["semantic"], step["rrf_applied"], step["fused"], step["lexical"]), (0, False, 2, 2))   # fused = lexical ranking alone

    def test_semantic_search_that_returns_nothing_is_not_reported_as_fusion(self):
        step = self.retrieve_step(self.setup_memory(FakeEmbedder()), "Which airline and plane am I taking?")   # nothing stored about flights
        self.assertEqual((step["semantic"], step["rrf_applied"]), (0, False))

    def test_candidates_found_by_a_part_search_keep_their_real_method(self):
        step = self.retrieve_step(self.setup_memory(FakeEmbedder()), "When is my dentist appointment, and what is my gym locker number?")
        via = {c["ref"]: c["via"] for c in step["candidates"]}
        self.assertTrue(via["F2"] in ("lexical", "both") and via["F1"] in ("lexical", "both"), via)      # not mislabelled "related"


if __name__ == "__main__":
    unittest.main()
