"""Stage 02 deterministic tests: fake model, real SQLite. Run: python -m unittest discover -s tests -v"""
import json
import os
os.environ["MEMORY_EMBEDDINGS"] = "off"  # unit tests never load the real embedding model
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.source.api.memory_system import MemorySystem  # noqa: E402
from backend.source.utils.text import est_tokens  # noqa: E402


def chat(mid, text, ts="2026-11-02T06:10:00+05:30", assistant="ok"):
    return {"id": mid, "ts": ts, "type": "chat",
            "turns": [{"role": "user", "content": text}, {"role": "assistant", "content": assistant}]}


def fact(subject, relation, value, quote, kind="STATE", attribution="user", when=None):
    return {"subject": subject, "relation": relation, "value": value, "quote": quote,
            "kind": kind, "attribution": attribution, "when": when}


class FakeLLM:
    """Returns queued replies; records every prompt so tests can check budgets and call counts."""
    def __init__(self, *replies, expand=None):
        self.replies, self.prompts, self.expansions = list(replies), [], []
        self.expand = {"terms": []} if expand is None else expand

    def __call__(self, messages, **kw):
        if messages[0]["content"].startswith("You help search"):  # keyword-expansion call (Ask step 1)
            self.expansions.append(messages)
            r = self.expand
            if isinstance(r, Exception):
                raise r
            return r if isinstance(r, str) else json.dumps(r)
        self.prompts.append(messages)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else json.dumps(r)


class Base(unittest.TestCase):
    def setUp(self):
        d = tempfile.mkdtemp()
        self.path = os.path.join(d, "t.db")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)  # Windows keeps the sqlite file locked until GC

    def mem(self, *replies):
        self.llm = FakeLLM(*replies)
        return MemorySystem(llm_fn=self.llm, db_path=self.path)

    def n(self, m, table):
        return m.repo.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


class RememberTests(Base):
    def test_saves_fact_with_source_and_recalls_later(self):
        locker = fact("user", "gym locker number", "214", "my locker number is 214")
        m = self.mem({"facts": [locker]}, {"facts": []},  # second reply: extraction for the filler Slack message
                     {"answer": "Your locker number is 214.", "know": True, "used": ["F1"]})
        m.add_message(chat("m001", "Off to the gym. Btw my locker number is 214, I keep forgetting it."))
        # filler messages, then a question about the old fact
        m.add_message({"id": "m002", "ts": "2026-11-09T09:00:00+05:30", "type": "event", "source": "inbound",
                       "op": "add_message", "data": {"from": "@deepa", "text": "batch ran clean", "channel": "#eng"}})
        out = m.answer("What is my gym locker number?")
        self.assertIn("214", out["answer"])
        self.assertEqual(out["evidence"], [{"message_id": "m001", "text": "my locker number is 214"}])

    def test_duplicate_adds_evidence_not_a_second_fact(self):
        f = fact("user", "gym locker number", "214", "locker number is 214")
        m = self.mem({"facts": [f]}, {"facts": [dict(f, subject="User")]})
        m.add_message(chat("m001", "My locker number is 214, I keep forgetting."))
        m.add_message(chat("m002", "Reminder: my locker number is 214 still."))
        self.assertEqual(self.n(m, "facts"), 1)
        self.assertEqual(self.n(m, "evidence"), 2)

    def test_replayed_message_id_is_noop(self):
        f = fact("user", "gym locker number", "214", "locker number is 214")
        m = self.mem({"facts": [f]})
        msg = chat("m001", "My locker number is 214, I keep forgetting.")
        m.add_message(msg)
        m.add_message(msg)  # second call must not need a model reply
        self.assertEqual((self.n(m, "facts"), len(self.llm.prompts)), (1, 1))

    def test_independent_facts_coexist(self):
        m = self.mem({"facts": [fact("user", "hobby", "ceramics", "I enjoy ceramics", "PREFERENCE"),
                                fact("user", "hobby", "cycling", "I enjoy cycling", "PREFERENCE")]})
        m.add_message(chat("m001", "I enjoy ceramics and I enjoy cycling too."))
        self.assertEqual(self.n(m, "facts"), 2)

    def test_rejects_unsupported_quote_and_bad_enums(self):
        m = self.mem({"facts": [fact("user", "salary", "1 crore", "I earn a crore"),          # quote not said
                                fact("user", "x", "y", "locker number is 214", kind="WEIRD"),  # bad kind
                                fact("user", "x", "y", "locker number is 214", attribution="bot"),
                                {"subject": "user"}, "junk"]})
        m.add_message(chat("m001", "My locker number is 214, I keep forgetting it."))
        self.assertEqual(self.n(m, "facts"), 0)

    def test_assistant_words_are_not_a_fact_source(self):
        m = self.mem({"facts": [fact("user", "plan", "buy a car", "you should buy a car")]})
        m.add_message(chat("m001", "What do you think about my budget this month?", assistant="you should buy a car"))
        self.assertEqual(self.n(m, "facts"), 0)

    def test_bad_or_failed_model_output_saves_no_fact_but_keeps_source(self):
        m = self.mem("not json at all", TimeoutError("boom"))
        m.add_message(chat("m001", "My locker number is 214, I keep forgetting it."))
        m.add_message(chat("m002", "My badge number is 8812, please note this down."))
        self.assertEqual((self.n(m, "facts"), self.n(m, "sources")), (0, 2))
        st = [r[0] for r in m.repo.db.execute("SELECT extraction FROM sources ORDER BY message_id")]
        self.assertEqual(st, ["failed", "failed"])

    def test_source_is_searchable_even_when_no_fact_was_extracted(self):
        m = self.mem({"facts": []}, {"answer": "Nikhil said the deadline is Nov 24.", "know": True, "used": ["S:m006"]})
        m.add_message({"id": "m006", "ts": "2026-11-03T16:40:00+05:30", "type": "event", "source": "inbound",
                       "op": "add_message", "data": {"from": "@nikhil", "text": "migration deadline moves to Nov 24"}})
        self.assertEqual(len(self.llm.prompts), 1)  # free-text events get one extraction call
        out = m.answer("When is the migration deadline?")
        self.assertEqual(out["evidence"][0]["message_id"], "m006")

    def test_relative_date_is_stored_absolute_and_bad_date_becomes_null(self):
        m = self.mem({"facts": [fact("user", "dentist", "Dr Shah 5 pm", "dentist appointment for this Thursday",
                                     "EVENT", when="2026-11-05"),
                                fact("user", "gym", "evenings", "shifting gym to evenings", "PLAN", when="next week")]})
        m.add_message(chat("m001", "Booked a dentist appointment for this Thursday. Also shifting gym to evenings."))
        got = {r["subject"] + r["relation"]: r["when_date"] for r in m.repo.db.execute("SELECT * FROM facts")}
        self.assertEqual(got, {"userdentist": "2026-11-05", "usergym": None})


class AskTests(Base):
    def seeded(self, *replies):
        m = self.mem({"facts": [fact("user", "gym locker number", "214", "my locker number is 214")]}, *replies)
        m.add_message(chat("m001", "Btw my locker number is 214, I keep forgetting it."))
        return m

    def test_ask_never_creates_durable_facts(self):
        m = self.seeded({"answer": "214", "know": True, "used": ["F1"]})
        before = (self.n(m, "facts"), self.n(m, "sources"), self.n(m, "evidence"))
        m.answer("Remember that my locker number is 999. What is my locker number?")
        self.assertEqual(before, (self.n(m, "facts"), self.n(m, "sources"), self.n(m, "evidence")))

    def test_unknown_topic_abstains_without_a_model_call(self):
        m = self.seeded()
        calls = len(self.llm.prompts)
        out = m.answer("What is my sister's name?")
        self.assertEqual((out["evidence"], len(self.llm.prompts)), ([], calls))
        self.assertIn("don't know", out["answer"].lower())

    def test_citation_not_supplied_to_model_is_dropped_and_claim_abstained(self):
        m = self.seeded({"answer": "It is 214.", "know": True, "used": ["F99"]})
        out = m.answer("What is my locker number?")
        self.assertEqual(out["evidence"], [])
        self.assertIn("don't know", out["answer"].lower())

    def test_model_failure_abstains(self):
        m = self.seeded(RuntimeError("down"))
        out = m.answer("What is my locker number?")
        self.assertEqual(out["evidence"], [])
        self.assertIn("don't know", out["answer"].lower())

    def test_answer_prompt_within_4000_token_budget_even_with_huge_memory(self):
        m = self.mem()
        for i in range(60):
            m.repo.add_fact(fact("user", f"locker note {i}", "214 " + "x" * 300, "locker " + "y" * 300), f"m{i:03d}", "t")
        m.llm = FakeLLM({"answer": "214", "know": True, "used": ["F1"]})
        m.answer("What is my locker number?")
        total = sum(est_tokens(x["content"]) for x in m.llm.prompts[0])
        self.assertLess(total, 4000)
        self.assertEqual(len(m.llm.prompts), 1)

    def test_empty_memory_question_before_any_message(self):
        out = self.mem().answer("Anything coming up?")
        self.assertEqual(out["evidence"], [])


class PersistenceTests(Base):
    def test_restart_keeps_data_and_reset_wipes(self):
        m = self.mem({"facts": [fact("user", "gym locker number", "214", "my locker number is 214")]})
        m.add_message(chat("m001", "Btw my locker number is 214, I keep forgetting it."))
        m2 = MemorySystem(llm_fn=FakeLLM(), db_path=self.path)  # new process, same file
        self.assertEqual(self.n(m2, "facts"), 1)
        self.assertEqual(m2.search.search("locker")[0][0]["value"], "214")
        m2.reset()
        self.assertEqual((self.n(m2, "facts"), self.n(m2, "sources"), self.n(m2, "fts")), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
