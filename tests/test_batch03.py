"""Batch 03 tests: Ask is read-only and honest, shared request deadline, multi-part retrieval, current-state consistency."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_stage02 import Base, FakeLLM, chat, fact  # noqa: E402
from backend.source.service.memory.intent import is_change_request  # noqa: E402
from backend.source.service.memory.retrieval import READ_ONLY_REPLY  # noqa: E402


class AskIsReadOnlyTests(Base):
    def seeded(self):
        m = self.mem({"facts": [fact("user", "job offer", "from Zentrix", "looking at a job offer from Zentrix")]})
        m.add_message(chat("m1", "I've been looking at a job offer from Zentrix, keep it between us."))
        return m

    def test_a_forget_or_remember_command_in_ask_gets_an_honest_reply_and_changes_nothing(self):
        m = self.seeded()
        m.llm = llm = FakeLLM()
        for q in ["Forget the job offer", "Please delete everything about Zentrix.", "Remember that my locker is 214",
                  "Don't keep that job offer anywhere"]:
            out = m.answer(q)
            self.assertEqual((out["answer"], out["evidence"]), (READ_ONLY_REPLY, []), q)
            self.assertNotIn("forgot", out["answer"].lower().replace("forgotten or changed", ""))
        self.assertEqual((len(llm.prompts), len(llm.expansions)), (0, 0))   # no model call, so nothing could be promised
        self.assertEqual(self.n(m, "facts"), 1)                             # and nothing was deleted or added

    def test_real_questions_that_mention_forgetting_still_get_answered(self):
        m = self.seeded()
        m.llm = FakeLLM({"answer": "You have a job offer from Zentrix.", "know": True, "used": ["F1"]})
        out = m.answer("Did I forget to tell anyone about my job offer?")
        self.assertEqual([e["message_id"] for e in out["evidence"]], ["m1"])
        for q in ["Who should I not forget to thank", "Tell me what you know about forgetting my job offer"]:
            self.assertFalse(is_change_request(q), q)

    def test_the_answer_and_job_prompts_forbid_claiming_to_change_memory(self):
        m = self.seeded()
        m.llm = FakeLLM({"answer": "x", "know": False, "used": []})
        m.answer("What do I know about the Zentrix job offer?")
        self.assertIn("You can only read memory here", m.llm.prompts[0][0]["content"])
        from backend.source.service.memory.acting import AGENT_SYS
        self.assertIn("cannot save, correct or delete memories", AGENT_SYS)


class DeadlineTests(Base):
    def test_every_call_gets_what_is_left_of_the_shared_deadline_and_none_after_it(self):
        m = self.mem()
        seen = []
        m.llm = lambda messages, **kw: seen.append(kw["timeout"]) or "{}"
        bounded = m._bounded(30)
        bounded([], timeout=20)
        bounded([], timeout=100)
        self.assertEqual((seen[0], seen[1] <= 30), (20, True))      # the caller cap holds and nothing exceeds the time left
        with self.assertRaises(TimeoutError):
            m._bounded(1)([])                                       # under 2 s left: fail fast, callers fall back

    def test_an_exhausted_deadline_degrades_to_lexical_search_and_an_honest_abstention_not_a_crash(self):
        m = self.mem({"facts": [fact("user", "gym locker number", "214", "my locker number is 214")]})
        m.add_message(chat("m1", "Btw my locker number is 214, I keep forgetting it."))
        m._bounded = lambda seconds: (lambda messages, **kw: (_ for _ in ()).throw(TimeoutError("request deadline reached")))
        out = m.answer("What is my gym locker number?")             # keyword call and answer call both time out
        self.assertEqual(out["evidence"], [])
        self.assertIn("don't know", out["answer"].lower())
        m.add_message(chat("m2", "My badge number is 8812, please note this down."))   # extraction times out: source kept
        self.assertEqual(m.repo.source("m2")["extraction"], "failed")


class RetrievalCompletenessTests(Base):
    def test_each_part_of_a_two_part_question_is_searched_so_neither_is_crowded_out(self):
        m = self.mem()
        m.repo.put_source("m0", "2026-11-02T09:00:00+05:30", "chat", "user", "seed")
        for i in range(14):   # fourteen facts that match the first part on every word
            m.repo.add_fact(fact("user", f"dentist appointment reminder clinic {i}", f"note {i}", f"dentist appointment reminder clinic note {i}"), "m0", "t")
        m.repo.add_fact(fact("user", "gym locker number", "214", "my gym locker number is 214"), "m0", "t")
        m.llm = llm = FakeLLM({"answer": "x", "know": False, "used": []})
        out = m.answer("Which dentist appointment reminder clinic is it, and what is my gym locker number?")
        self.assertIn("locker number is 214", llm.prompts[0][1]["content"])
        self.assertIn({"step": "parts", "n": 2}, out["trace"])

    def test_a_single_question_is_not_split(self):
        m = self.mem()
        m.repo.put_source("m0", "t", "chat", "user", "seed")
        m.repo.add_fact(fact("user", "gym locker number", "214", "my gym locker number is 214"), "m0", "t")
        m.llm = FakeLLM({"answer": "x", "know": False, "used": []})
        out = m.answer("What is my gym locker number?")
        self.assertFalse(any(t["step"] == "parts" for t in out["trace"]))

    def test_one_very_long_item_does_not_crowd_out_the_others(self):
        m = self.mem()
        m.repo.put_source("m0", "t", "chat", "user", "seed")
        m.repo.add_fact(fact("user", "gym locker notes", "x " * 6000, "locker " + "long " * 3000), "m0", "t")
        m.repo.add_fact(fact("user", "gym locker number", "214", "my gym locker number is 214"), "m0", "t")
        m.llm = llm = FakeLLM({"answer": "214", "know": True, "used": ["F2"]})
        out = m.answer("What is my gym locker number?")
        self.assertIn("locker number is 214", llm.prompts[0][1]["content"])
        self.assertEqual([e["message_id"] for e in out["evidence"]], ["m0"])


class CurrentStateConsistencyTests(Base):
    def edited(self):
        new = dict(fact("user", "preferred editor", "VS Code", "my preferred editor is now VS Code", "PREFERENCE"), replaces=1)
        m = self.mem({"facts": [fact("user", "preferred editor", "Emacs", "my preferred editor is Emacs", "PREFERENCE")]}, {"facts": [new]})
        m.add_message(chat("m1", "My preferred editor is Emacs, always has been.", ts="2026-11-01T10:00:00+05:30"))
        m.add_message(chat("m2", "Update: my preferred editor is now VS Code.", ts="2026-11-05T10:00:00+05:30"))
        return m

    def test_a_current_state_answer_that_cites_only_an_outdated_version_is_not_allowed_to_stand(self):
        m = self.edited()
        m.llm = FakeLLM({"answer": "You prefer Emacs.", "know": True, "used": ["F1"]})
        out = m.answer("Which editor do I prefer?")
        self.assertEqual(out["evidence"], [])
        self.assertNotIn("Emacs", out["answer"])
        self.assertIn({"step": "abstain", "reason": "answer rests only on outdated versions"}, out["trace"])

    def test_the_same_citation_is_fine_for_a_question_about_the_past(self):
        m = self.edited()
        m.llm = FakeLLM({"answer": "You used Emacs before.", "know": True, "used": ["F1"]})
        out = m.answer("Which editor did I prefer before?")
        self.assertEqual([e["message_id"] for e in out["evidence"]], ["m1"])

    def test_citing_the_current_version_alongside_history_is_fine(self):
        m = self.edited()
        m.llm = FakeLLM({"answer": "VS Code now (Emacs before).", "know": True, "used": ["F2", "F1"]})
        out = m.answer("Which editor do I prefer?")
        self.assertEqual({e["message_id"] for e in out["evidence"]}, {"m1", "m2"})


class Live4Tests(Base):
    def test_a_no_but_answer_still_returns_the_facts_it_relied_on(self):
        m = self.mem({"facts": [fact("Kavya", "allergy", "peanuts", "Kavya is allergic to peanuts", "STATE", "other")]})
        m.add_message(chat("m1", "Kavya is allergic to peanuts so nothing with peanuts in it."))
        m.llm = FakeLLM({"answer": "I don't know of an allergy of yours; the peanut allergy is Kavya's.", "know": False, "used": ["F1"]})
        out = m.answer("Am I allergic to anything?")
        self.assertEqual([e["message_id"] for e in out["evidence"]], ["m1"])
        m.llm = FakeLLM({"answer": "I don't know.", "know": False, "used": []})
        self.assertEqual(m.answer("Am I allergic to anything?")["evidence"], [])

    def agenda_memory(self):
        m = self.mem()
        m.repo.put_source("m0", "2026-11-04T08:59:00+05:30", "chat", "user", "seed")
        m.repo.add_fact(dict(fact("user", "flight booking", "IndiGo 6E 2134 from BLR", "Booked my flight, IndiGo 6E 2134", "EVENT"),
                             when="2026-11-13", when_ts="2026-11-13T18:15"), "m0", "t")
        m.repo.add_fact(fact("user", "gift for Priya", "silver photo frame", "still need to get Priya's gift", "TODO"), "m0", "t")
        m.repo.add_fact(dict(fact("user", "conference", "Berlin", "The Berlin conference is booked", "EVENT"), when="2027-03-01"), "m0", "t")
        return m

    def test_an_agenda_question_gets_what_is_due_soon_and_still_open_even_with_no_shared_words(self):
        m = self.agenda_memory()
        m.llm = llm = FakeLLM({"answer": "x", "know": False, "used": []})
        out = m.answer("Anything important I should keep in mind for this week?")
        prompt = llm.prompts[0][1]["content"]
        self.assertTrue("6E 2134" in prompt and "silver photo frame" in prompt)
        self.assertNotIn("Berlin", prompt)                       # beyond the 14-day horizon
        self.assertIn({"step": "agenda", "n": 2}, out["trace"])

    def test_an_ordinary_question_does_not_pull_in_the_agenda(self):
        m = self.agenda_memory()
        m.llm = FakeLLM({"answer": "x", "know": False, "used": []})
        out = m.answer("What is the gift idea for Priya?")
        self.assertFalse(any(t["step"] == "agenda" for t in out["trace"]))

    def test_an_outdated_date_cannot_go_into_a_message_to_someone_else_but_the_current_one_can(self):
        from backend.source.service.memory.act_guard import ActGuard
        m = self.mem()
        m.repo.put_source("m0", "2026-11-10T22:29:00+05:30", "chat", "user", "seed")
        old = dict(fact("migration", "deadline", "Nov 20", "deadline is Nov 20", "PLAN"), when="2026-11-20")
        new = dict(fact("migration", "deadline", "Nov 24", "deadline moves to Nov 24", "PLAN"), when="2026-11-24")
        m.repo.add_fact(old, "m0", "t")
        m.repo.add_fact(new, "m0", "t", replaces=1)
        g = ActGuard(m.repo, "x", None, request="tell nikhil", memory="nikhil")
        for text in ["on track for the new deadline (moved from Nov 20)", "moved from 20th November", "was 2026-11-20", "Nov 20th was old"]:
            self.assertIn("out of date", g("slack_send_message", {"to": "nikhil", "text": text}) or "", text)
        self.assertIsNone(g("slack_send_message", {"to": "nikhil", "text": "on track for the new Nov 24 deadline"}))
        self.assertIsNone(g("reminders_create", {"text": "the old deadline was Nov 20", "due": "2026-11-12"}))   # to oneself: allowed

    def test_a_date_that_is_still_current_for_something_else_is_not_treated_as_outdated(self):
        m = self.mem()
        m.repo.put_source("m0", "t", "chat", "user", "seed")
        m.repo.add_fact(dict(fact("a", "deadline", "Nov 20", "a deadline is Nov 20", "PLAN"), when="2026-11-20"), "m0", "t")
        m.repo.add_fact(dict(fact("a", "deadline", "Nov 24", "a deadline moves to Nov 24", "PLAN"), when="2026-11-24"), "m0", "t", replaces=1)
        m.repo.add_fact(dict(fact("b", "party", "Nov 20", "party on Nov 20", "EVENT"), when="2026-11-20"), "m0", "t")
        self.assertEqual(m.repo.outdated_dates(), set())


class IdentityTests(Base):
    def ev(self, m, mid, source, data):
        m.add_message({"id": mid, "ts": "2026-11-03T11:20:00+05:30", "type": "event", "source": source, "op": "add_message", "data": data})

    def test_identity_comes_only_from_stored_events_and_never_invented(self):
        m = self.mem({"facts": []}, {"facts": []}, {"facts": []})
        self.assertEqual(m.repo.people_note(), "")                       # nothing stored: nothing claimed
        self.ev(m, "m1", "inbound", {"from": "@kavya.m", "to": "@arjun", "text": "hey Arjun, can you review my PR by Thursday?"})
        self.ev(m, "m2", "inbound", {"from": "@kavya", "to": "@arjun", "text": "gg today, rematch thursday 7:30 same court"})
        self.ev(m, "m3", "inbound", {"from": "@deepa", "channel": "#eng", "text": "standing by in the channel for the release"})
        note = m.repo.people_note()
        self.assertIn("own handle is @arjun", note)
        self.assertIn("@kavya/@kavya.m", note)
        self.assertNotIn("deepa", note)

    def test_a_handle_the_user_wrote_from_is_the_users_own_handle(self):
        m = self.mem({"facts": []}, {"facts": []})
        self.ev(m, "m1", "inbound", {"from": "@nikhil", "to": "@someoneelse", "text": "a direct message to somebody else"})
        self.ev(m, "m2", "user", {"from": "@arjun", "to": "@nikhil", "text": "on my way to the standup room now"})
        self.assertIn("own handle is @arjun", m.repo.people_note())

    def test_the_answer_and_job_prompts_carry_the_identity_note(self):
        m = self.mem({"facts": []}, {"facts": [fact("@kavya.m", "asked user to", "review the PR by Thursday", "can you review my PR by Thursday", "TODO", "other")]})
        self.ev(m, "m1", "inbound", {"from": "@kavya", "to": "@arjun", "text": "see you at the badminton court tonight"})
        self.ev(m, "m2", "inbound", {"from": "@kavya.m", "to": "@arjun", "text": "hey Arjun, can you review my PR by Thursday?"})
        m.llm = llm = FakeLLM({"answer": "x", "know": False, "used": []})
        m.answer("Did anyone at work ask me for something to review?")
        system = llm.prompts[0][0]["content"]
        self.assertTrue("own handle is @arjun" in system and "never merge them" in system)


class Live6Tests(Base):
    def memory_with_many_matches(self):
        m = self.mem()
        m.repo.put_source("m0", "2026-11-10T08:59:00+05:30", "chat", "user", "seed")
        for i in range(20):   # many facts that match an "important this week" question lexically
            m.repo.add_fact(fact("user", f"important note {i}", f"keep in mind item {i}", f"important note {i} to keep in mind this week"), "m0", "t")
        m.repo.add_fact(dict(fact("user", "flight booking", "IndiGo 6E 2134", "Booked my flight, IndiGo 6E 2134", "EVENT"), when="2026-11-13"), "m0", "t")
        m.repo.add_fact(dict(fact("Priya", "engagement", "Saturday 14 Nov", "Priya's engagement is on Saturday 14 Nov", "EVENT"), when="2026-11-14"), "m0", "t")
        m.repo.add_fact(dict(fact("user", "dentist appointment", "Fri 6 Nov", "dentist appointment on Friday 6 Nov", "EVENT"), when="2026-11-06"), "m0", "t")
        return m

    def test_due_soon_items_survive_a_full_candidate_list_for_an_agenda_question(self):
        m = self.memory_with_many_matches()
        m.llm = llm = FakeLLM({"answer": "x", "know": False, "used": []})
        out = m.answer("Anything important I should keep in mind for this week?")
        prompt = llm.prompts[0][1]["content"]
        self.assertTrue("6E 2134" in prompt and "engagement" in prompt)       # they used to be cut by the cap
        self.assertLess(prompt.index("6E 2134"), prompt.index("important note 0"))   # and they come first
        self.assertTrue(any(t["step"] == "agenda" for t in out["trace"]))

    def test_past_dated_items_are_marked_and_the_answer_prompt_says_to_leave_them_out(self):
        m = self.memory_with_many_matches()
        m.llm = llm = FakeLLM({"answer": "x", "know": False, "used": []})
        m.answer("Anything important I should keep in mind for this week? dentist")
        user, system = llm.prompts[0][1]["content"], llm.prompts[0][0]["content"]
        past = [ln for ln in user.splitlines() if "dentist appointment" in ln][0]
        future = [ln for ln in user.splitlines() if "6E 2134" in ln][0]
        self.assertTrue("PAST (before today)" in past and "PAST" not in future)
        self.assertIn("leave PAST items out", system)

    def test_the_per_message_prompts_stay_inside_the_official_input_budget(self):
        from backend.source.service.memory.extraction_pipeline import EXTRACT_SYS
        from backend.source.service.memory.forget_scope_search import SELECT_SYS
        from backend.source.utils.text import clip, est_tokens
        worst_extract = est_tokens(EXTRACT_SYS) + 4000 + 1000 + 60      # body clip + related facts + framing
        worst_forget = est_tokens(SELECT_SYS) + 2500 + 800 + 60         # candidate listing + request + framing
        self.assertLess(worst_extract, 8000)                            # per call
        self.assertLess(worst_extract + worst_forget, 10000)            # per message (extraction + forget decision)
        self.assertLessEqual(est_tokens(clip("word " * 20000, 4000)), 4000)


if __name__ == "__main__":
    unittest.main()
