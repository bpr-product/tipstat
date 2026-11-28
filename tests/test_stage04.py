"""Stage 04 deterministic tests: candidate discovery, evidence package, citations, budgets. Fake model, real SQLite."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_stage02 import Base, FakeLLM, chat, fact  # noqa: E402
from backend.source.utils.text import est_tokens  # noqa: E402


def ask_llm(m, *replies, expand=None):
    m.llm = FakeLLM(*replies, expand=expand)
    return m.llm


class RetrievalTests(Base):
    def dentist(self):
        m = self.mem({"facts": [fact("user", "dentist appointment", "Dr Shah, Friday 6 pm", "dentist got moved to Friday the 6th, 6 pm", "EVENT")]})
        m.add_message(chat("m027", "Clinic called, dentist got moved to Friday the 6th, 6 pm."))
        return m

    def test_paraphrase_is_found_through_model_suggested_keywords(self):
        m = self.dentist()
        llm = ask_llm(m, {"answer": "Friday 6 pm.", "know": True, "used": ["F1"]}, expand={"terms": ["dentist", "appointment"]})
        out = m.answer("When do I see the teeth doctor?")
        self.assertEqual(llm.expansions[0][1]["content"], "When do I see the teeth doctor?")
        self.assertEqual([e["message_id"] for e in out["evidence"]], ["m027"])
        self.assertEqual(out["trace"][0], {"step": "expand", "terms": ["dentist", "appointment"]})

    def test_without_keywords_the_paraphrase_abstains_without_an_answer_call(self):
        m = self.dentist()
        llm = ask_llm(m)
        out = m.answer("When do I see the teeth doctor?")
        self.assertEqual((out["evidence"], len(llm.prompts)), ([], 0))
        self.assertIn("don't know", out["answer"].lower())

    def test_keyword_call_failure_falls_back_to_lexical_search(self):
        m = self.dentist()
        ask_llm(m, {"answer": "Friday 6 pm.", "know": True, "used": ["F1"]}, expand=TimeoutError("slow"))
        out = m.answer("When is my dentist appointment?")
        self.assertEqual(len(out["evidence"]), 1)
        self.assertEqual(out["trace"][0], {"step": "expand", "error": "TimeoutError"})

    def test_unusable_keywords_are_filtered_and_never_break_the_query(self):
        m = self.dentist()
        ask_llm(m, {"answer": "Friday 6 pm.", "know": True, "used": ["F1"]},
                expand={"terms": ['bad"term OR x', 123, "dentist", "a", "x" * 50]})
        out = m.answer("When do I see the teeth doctor?")
        self.assertEqual(out["trace"][0]["terms"], ["dentist"])
        self.assertEqual(len(out["evidence"]), 1)

    def test_no_stored_data_means_no_model_calls_at_all(self):
        m = self.mem()
        llm = ask_llm(m)
        out = m.answer("Anything coming up?")
        self.assertEqual((len(llm.expansions), len(llm.prompts), out["evidence"]), (0, 0, []))

    def test_old_fact_is_found_among_hundreds_of_distractors(self):
        m = self.mem()
        m.repo.add_fact(fact("user", "passport number", "K1234567", "my passport number is K1234567"), "m001", "2026-11-01")
        for i in range(300):
            m.repo.add_fact(fact("user", f"note {i}", f"gym session {i} leg day", f"went to gym session {i}"), f"m{i + 2:03d}", "2026-11-10")
        llm = ask_llm(m, {"answer": "K1234567", "know": True, "used": ["F1"]})
        out = m.answer("What is my passport number?")
        self.assertIn("K1234567", llm.prompts[0][1]["content"])
        self.assertEqual(out["evidence"][0]["message_id"], "m001")

    def test_both_calls_together_stay_under_the_4000_token_answer_budget(self):
        m = self.mem()
        m.repo.put_source("m000", "t", "chat", "user", "seed")  # real facts always have a source; this enables expansion
        for i in range(80):
            m.repo.add_fact(fact("user", f"locker note {i}", "214 " + "x" * 300, "locker " + "y" * 300), f"m{i:03d}", "t")
        llm = ask_llm(m, {"answer": "214", "know": True, "used": ["F1"]}, expand={"terms": ["locker"]})
        m.answer("What is my locker number?")
        total = sum(est_tokens(x["content"]) for call in llm.expansions + llm.prompts for x in call)
        self.assertLess(total, 4000)
        self.assertEqual(len(llm.expansions) + len(llm.prompts), 2)

    def test_unrelated_question_with_no_candidates_abstains_without_an_answer_call(self):
        m = self.dentist()
        llm = ask_llm(m, expand={"terms": ["sibling"]})
        out = m.answer("What is my sister's name?")
        self.assertEqual((out["evidence"], len(llm.prompts)), ([], 0))


class EvidencePackageTests(Base):
    def test_structured_event_source_is_not_sent_twice(self):
        m = self.mem()
        m.add_message({"id": "m19", "ts": "2026-11-03T09:01:00+05:30", "type": "event", "source": "assistant", "op": "add_event",
                       "data": {"id": "evt_dentist", "title": "Dentist - Dr. Shah", "start": "2026-11-05T17:00:00+05:30"}})
        llm = ask_llm(m, {"answer": "Thursday 5 pm", "know": True, "used": ["F1"]})
        m.answer("When is my dentist appointment?")
        prompt = llm.prompts[0][1]["content"]
        self.assertIn("[F1]", prompt)
        self.assertNotIn("[S:m19]", prompt)

    def test_chat_source_is_still_offered_next_to_its_fact(self):
        m = self.mem({"facts": [fact("user", "dinner guest allergy", "Kavya peanuts", "Kavya is allergic to peanuts")]})
        m.add_message(chat("m014", "Hosting dinner Saturday. Kavya is allergic to peanuts, hosting Rohan too."))
        llm = ask_llm(m, {"answer": "x", "know": False, "used": []})
        m.answer("Who is coming to dinner Saturday?")
        self.assertIn("[S:m014]", llm.prompts[0][1]["content"])

    def test_others_statements_are_labelled_so_they_are_not_the_users(self):
        m = self.mem({"facts": [fact("migration", "deadline", "Nov 24", "deadline moves to Nov 24", "PLAN")]})
        m.add_message({"id": "m26", "ts": "2026-11-03T16:40:00+05:30", "type": "event", "source": "inbound", "op": "add_message",
                       "data": {"from": "@nikhil", "text": "Heads up: the migration deadline moves to Nov 24 for the audit."}})
        llm = ask_llm(m, {"answer": "x", "know": False, "used": []})
        m.answer("What is the migration deadline?")
        self.assertIn("(CURRENT; other,", llm.prompts[0][1]["content"])
        self.assertIn("never present it as the user's own", llm.prompts[0][0]["content"].lower().replace("an item marked (other, ...) is what someone else said or wrote. ", ""))

    def test_a_fact_from_someone_else_shows_who_and_where_it_came_from(self):
        m = self.mem({"facts": [fact("@kavya.m", "asked user to", "review the ledger-sync PR by Thursday", "can you review my ledger-sync PR by Thursday", "TODO", "other")]})
        m.add_message({"id": "m21", "ts": "2026-11-03T11:20:00+05:30", "type": "event", "source": "inbound", "op": "add_message",
                       "data": {"from": "@kavya.m", "to": "@arjun", "text": "hey Arjun, can you review my ledger-sync PR by Thursday? no rush before that"}})
        llm = ask_llm(m, {"answer": "x", "know": False, "used": []})
        m.answer("Did anyone at work ask me for something?")
        sent = llm.prompts[0]
        self.assertIn("from: @kavya.m", sent[1]["content"])
        self.assertIn("the handle, e.g. @name.surname", sent[0]["content"])

    def test_evidence_is_only_what_was_cited_not_everything_supplied(self):
        m = self.mem({"facts": [fact("user", "hobby", "ceramics", "I enjoy ceramics", "PREFERENCE"),
                                fact("user", "hobby", "cycling", "I enjoy cycling", "PREFERENCE")]})
        m.add_message(chat("m001", "I enjoy ceramics and I enjoy cycling on weekends."))
        llm = ask_llm(m, {"answer": "Ceramics.", "know": True, "used": ["F1", "F77"]})
        out = m.answer("What hobbies do I enjoy?")
        self.assertIn("[F2]", llm.prompts[0][1]["content"])  # supplied...
        self.assertEqual(len(out["evidence"]), 1)             # ...but only the valid citation is returned
        steps = [t["step"] for t in out["trace"]]
        self.assertEqual(steps, ["expand", "semantic", "retrieve", "supply", "answer"])  # candidates, supplied and cited kept distinct


if __name__ == "__main__":
    unittest.main()
