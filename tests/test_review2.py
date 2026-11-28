"""Second manual-testing review: reproductions for each reported finding (written first, run on the old code, then fixed)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_stage02 import Base, FakeLLM, chat, fact  # noqa: E402
from test_stage05 import everything  # noqa: E402
from test_experience import Base as WsBase  # noqa: E402


def facts(m):
    return [dict(r) for r in m.repo.db.execute("SELECT * FROM facts ORDER BY id")]


def line(prompt, needle):
    return next(ln for ln in prompt.splitlines() if needle in ln)


class ReceiptsAndRedactionTests(WsBase):
    def test_1_a_historical_receipt_that_names_a_forgotten_fact_is_redacted_even_when_its_words_live_on_elsewhere(self):
        a = fact("user", "allergy", "shellfish", "I'm allergic to shellfish", "STATE")
        b = fact("user", "restaurant", "shellfish platter", "shellfish platter at the restaurant tonight", "EVENT")
        w = self.ws({"facts": [a]}, {"facts": [b]}, {"facts": []}, {"forget": True, "fact_ids": [1], "source_ids": []})
        self.send(w, "tell", "I'm allergic to shellfish, keep it between us.")
        self.send(w, "tell", "We booked a shellfish platter at the restaurant tonight.")
        self.send(w, "tell", "Please forget my allergy, delete it.")
        msgs = w.open_chat(self.pid, self.cid)["messages"]
        text = " | ".join(m["content"] + str(m["payload"]) for m in msgs)
        self.assertNotIn("allergy", text.lower())                      # the receipt that listed it
        self.assertNotIn("allergic", text.lower())                     # and the message that told it
        self.assertIn("restaurant", text)                              # the unrelated message about the same word survives

    def test_3_unrelated_responses_are_not_redacted_because_of_dates_or_numbers(self):
        passport = fact("user", "passport", "K1234567", "my passport expires in 2026", "STATE")
        locker = fact("user", "gym locker number", "214", "my gym locker number is 214")
        w = self.ws({"facts": [passport]}, {"facts": [locker]}, {"answer": "214", "know": True, "used": ["F2"]},
                    {"facts": []}, {"forget": True, "fact_ids": [1], "source_ids": []})
        self.send(w, "tell", "My passport expires in 2026, keep it between us.")
        self.send(w, "tell", "Btw my gym locker number is 214, I keep forgetting it.")
        self.send(w, "ask", "What is my gym locker number?")
        self.send(w, "tell", "Please forget my passport details, delete them.")
        msgs = w.open_chat(self.pid, self.cid)["messages"]
        survivors = [m for m in msgs if "locker" in m["content"].lower() or m["content"] == "214"]
        self.assertTrue(len(survivors) >= 3 and not any("removed" in m["content"] for m in survivors),
                        [m["content"][:40] for m in msgs])           # the locker turns and the answer "214" are untouched
        self.assertFalse(any("passport" in (m["content"] + str(m["payload"])).lower() for m in msgs))


class RedactionPrecisionTests(WsBase):
    def test_3b_an_erased_fact_that_was_only_a_candidate_is_dropped_from_the_list_not_the_whole_answer(self):
        import json
        w = self.ws()
        repo = w.system(self.pid).repo
        flow = {"candidates": [{"ref": "F1", "label": "passport"}, {"ref": "F2", "label": "locker"}], "supplied": [{"ref": "F1", "label": "passport"}]}
        for rid, role, mode, content, payload in [("r1", "assistant", "ask", "Your locker is 214.", {"fact_refs": [2], "flow": flow}),
                                                  ("r2", "assistant", "tell", "Done: saved 1.", {"fact_refs": [1]})]:
            repo.db.execute("INSERT INTO chat_messages(conversation_id, role, mode, content, payload, ts, request_id) VALUES(?,?,?,?,?,?,?)",
                            (self.cid, role, mode, content, json.dumps(payload), "t", rid))
        repo.db.commit()
        n = repo.redact_chats(set(), set(), set(), {1})
        rows = {r["request_id"]: r for r in repo.db.execute("SELECT * FROM chat_messages")}
        kept = json.loads(rows["r1"]["payload"])
        self.assertEqual((n, rows["r1"]["content"], [c["ref"] for c in kept["flow"]["candidates"]], kept["flow"]["supplied"]),
                         (1, "Your locker is 214.", ["F2"], []))
        self.assertIn("removed", rows["r2"]["content"])                 # the receipt that referenced the erased fact id is redacted


class BroadForgetTests(Base):
    def test_2_a_broad_request_removes_every_fact_that_names_the_thing_not_only_the_one_the_model_picked(self):
        replies = [{"facts": [fact("user", "employer", "Acme", "I work at Acme")]},
                   {"facts": [fact("user", "role", "senior engineer", "I am a senior engineer at Acme")]},
                   {"facts": [fact("user", "salary", "90k", "my Acme salary is 90k")]},
                   {"facts": [fact("Dana", "manager", "manages me", "Dana manages me at Acme")]},
                   {"facts": [fact("user", "gym", "7 pm", "my gym is at 7 pm")]},
                   {"facts": []}, {"forget": True, "fact_ids": [1], "source_ids": []}]
        m = self.mem(*replies)
        for i, t in enumerate(["I work at Acme these days.", "I am a senior engineer at Acme now.", "Also my Acme salary is 90k per month.",
                               "Dana manages me at Acme since spring.", "And my gym is at 7 pm on weekdays."]):
            m.add_message(chat(f"m{i}", t))
        m.add_message(chat("m9", "Please forget everything about Acme."))
        self.assertEqual([r["value"] for r in facts(m)], ["7 pm"])
        self.assertFalse(everything(m, "acme"))

    def test_2b_a_narrow_request_does_not_sweep_everything_that_shares_a_name(self):
        replies = [{"facts": [fact("user", "offer", "from Acme", "I got an offer from Acme")]},
                   {"facts": [fact("user", "office", "Acme office is in Whitefield", "the Acme office is in Whitefield")]},
                   {"facts": []}, {"forget": True, "fact_ids": [1], "source_ids": []}]
        m = self.mem(*replies)
        m.add_message(chat("m1", "I got an offer from Acme, keep it between us."))
        m.add_message(chat("m2", "Fyi the Acme office is in Whitefield, long commute."))
        m.add_message(chat("m3", "Please forget the offer, delete it."))
        self.assertEqual([r["relation"] for r in facts(m)], ["office"])


class RetrievalLabelTests(Base):
    def chain(self):
        m = self.mem({"facts": [fact("user", "employer", "Initech", "I work at Initech")]},
                     {"facts": [dict(fact("user", "employer", "Globex", "Now I work at Globex"), replaces=1)]},
                     {"facts": [dict(fact("user", "employer", "Acme", "Now I work at Acme"), replaces=2)]})
        for i, t in enumerate(["I work at Initech.", "Now I work at Globex.", "Now I work at Acme."]):
            m.add_message(chat(f"m{i}", t, ts=f"2026-11-0{i + 1}T10:00:00+05:30"))
        return m

    def test_4_the_version_immediately_before_the_current_one_is_marked_as_such(self):
        m = self.chain()
        m.llm = llm = FakeLLM({"answer": "Globex", "know": True, "used": ["F2"]})
        m.answer("Where did I work before Acme?")
        prompt = llm.prompts[0][1]["content"]
        self.assertIn("immediately before the current value", line(prompt, "Globex"))
        self.assertIn("2 versions before the current value", line(prompt, "Initech"))
        self.assertIn("immediately before the current value", llm.prompts[0][0]["content"])   # and the rule that uses it

    def test_5_and_8_past_events_are_not_labelled_current_and_state_since_dates_are_not_called_past(self):
        m = self.mem()
        m.repo.put_source("m0", "2026-11-10T08:59:00+05:30", "chat", "user", "seed")
        m.repo.add_fact(dict(fact("user", "dentist appointment", "Fri 6 Nov", "dentist appointment on Friday 6 Nov", "EVENT"), when="2026-11-06"), "m0", "t")
        m.repo.add_fact(dict(fact("user", "dentist habit", "evenings", "dentist visits are in the evenings since 1 Nov", "STATE"), when="2026-11-01"), "m0", "t")
        m.llm = llm = FakeLLM({"answer": "x", "know": False, "used": []})
        m.answer("Tell me about my dentist")
        prompt = llm.prompts[0][1]["content"]
        past, state = line(prompt, "dentist appointment"), line(prompt, "dentist habit")
        self.assertTrue("PAST" in past and "(CURRENT;" not in past, past)
        self.assertTrue("(CURRENT;" in state and "PAST" not in state, state)

    def test_5_a_to_do_whose_date_has_passed_is_not_offered_as_pending_and_extraction_knows_about_completion(self):
        from backend.source.service.memory.extraction_pipeline import EXTRACT_SYS
        self.assertIn("completed", EXTRACT_SYS)
        m = self.mem()
        m.repo.put_source("m0", "2026-11-10T08:59:00+05:30", "chat", "user", "seed")
        m.repo.add_fact(dict(fact("user", "to-do", "return the library book", "return the library book", "TODO"), when="2026-11-02"), "m0", "t")
        m.repo.add_fact(fact("user", "to-do", "renew the passport", "renew the passport", "TODO"), "m0", "t")
        m.llm = llm = FakeLLM({"answer": "x", "know": False, "used": []})
        m.answer("Anything pending for me?")
        prompt = llm.prompts[0][1]["content"]
        self.assertTrue("renew the passport" in prompt and "library book" not in prompt)


class FollowUpTests(WsBase):
    def hint_for(self, w, question):
        self.send(w, "ask", "When is my dentist appointment?")
        self.llm.replies.append({"answer": "x", "know": False, "used": []})
        out = self.send(w, "ask", question)
        return out["payload"].get("follow_up_of")

    def test_6_standalone_questions_are_not_treated_as_follow_ups_but_thin_ones_are(self):
        w = self.ws({"facts": []})
        w.system(self.pid).repo.put_source("m0", "2026-11-10T08:59:00+05:30", "chat", "user", "seed")
        self.llm.replies = [{"answer": "x", "know": False, "used": []}] * 12
        for q in ["What is my gym locker number?", "Who is Priya?", "Do I have a flight on Friday?", "How many episodes did I watch?"]:
            self.assertIsNone(self.hint_for(w, q), q)
        for q in ["and with whom?", "When is it?", "Where?", "What about the next one?"]:
            self.assertIsNotNone(self.hint_for(w, q), q)


class ConflictAndHistoryTests(Base):
    def base(self, *more):
        return self.mem({"facts": [dict(fact("user", "work location", "remote", "I work remotely", "STATE"), cardinality="EXCLUSIVE")]}, *more)

    def test_7_a_temporary_or_date_specific_exception_is_not_a_conflict_but_a_plain_second_value_still_is(self):
        exception = dict(fact("user", "work location", "office", "this Friday I will be in the office only", "STATE"), cardinality="EXCLUSIVE", when="2026-11-06")
        temp = dict(fact("user", "work location", "cafe", "just for today I am working from the cafe", "STATE"), cardinality="EXCLUSIVE")
        plain = dict(fact("user", "work location", "hybrid", "I work hybrid", "STATE"), cardinality="EXCLUSIVE")
        m = self.base({"facts": [exception]}, {"facts": [temp]}, {"facts": [plain]})
        for i, t in enumerate(["I work remotely.", "This Friday I will be in the office only.", "Just for today I am working from the cafe.", "Actually I work hybrid."]):
            m.add_message(chat(f"m{i}", t))
        self.assertEqual([r["conflict_with"] is not None for r in facts(m)], [False, False, False, True])

    def test_8_a_statement_that_something_is_over_is_stored_as_former_not_current(self):
        former = dict(fact("user", "employer", "Initech", "I used to work at Initech until 2021", "STATE"), historical=True)
        claim = dict(fact("user", "employer", "Acme", "I work at Acme", "STATE"), historical=True)    # no marker in the words: flag ignored
        m = self.mem({"facts": [former]}, {"facts": [claim]})
        m.add_message(chat("m1", "I used to work at Initech until 2021, long ago."))
        m.add_message(chat("m2", "I work at Acme these days."))
        rows = facts(m)
        self.assertEqual([r["status"] for r in rows], ["superseded", "active"])
        m.llm = llm = FakeLLM({"answer": "x", "know": False, "used": []})
        m.answer("Where did I work at Initech?")
        self.assertIn("FORMER value", line(llm.prompts[0][1]["content"], "Initech"))

    def test_8b_restating_an_active_fact_as_former_closes_it(self):
        now = fact("user", "employer", "Initech", "I work at Initech")
        old = dict(now, quote="I used to work at Initech, no longer", historical=True)
        m = self.mem({"facts": [now]}, {"facts": [old]})
        m.add_message(chat("m1", "I work at Initech."))
        m.add_message(chat("m2", "I used to work at Initech, no longer though."))
        self.assertEqual([r["status"] for r in facts(m)], ["superseded"])


if __name__ == "__main__":
    unittest.main()
