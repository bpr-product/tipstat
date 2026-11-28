"""Core memory tests: reconciliation rules, versions and effective dates, derived facts and dependencies, semantic
retrieval with Reciprocal Rank Fusion (fake embedder: the real model cannot be downloaded in this environment)."""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_stage02 import Base, FakeLLM, chat, fact  # noqa: E402
from test_stage05 import everything  # noqa: E402
from backend.source.dal.repositories.search_repository import fuse  # noqa: E402

GROUPS = [{"dentist", "teeth", "doctor", "dental", "appointment"}, {"locker", "gym"}, {"flight", "plane", "airline"}]


class FakeEmbedder:
    """Concept vectors: words of a group share an axis, so 'teeth doctor' is close to 'dentist' with no shared word."""
    available = True

    def embed(self, texts):
        out = []
        for t in texts:
            ws = set(re.findall(r"\w+", t.lower()))
            v = [float(len(ws & g)) for g in GROUPS] + [0.0]
            if not any(v):
                v[-1] = 1.0
            n = sum(x * x for x in v) ** 0.5
            out.append([x / n for x in v])
        return out

    def embed_one(self, text):
        return self.embed([text])[0]


def rows(m, where="1=1"):
    return m.repo.db.execute(f"SELECT * FROM facts WHERE {where} ORDER BY id").fetchall()


class ReconciliationTests(Base):
    def test_a_paraphrase_of_a_stored_fact_adds_evidence_not_a_second_fact(self):
        m = self.mem({"facts": [fact("user", "gym locker number", "214", "my locker number is 214")]},
                     {"facts": [fact("user", "locker number", "214", "locker is number 214 still")]})
        m.add_message(chat("m1", "Btw my locker number is 214, I keep forgetting it."))
        m.add_message(chat("m2", "Reminder that the locker is number 214 still."))
        self.assertEqual((len(rows(m)), self.n(m, "evidence")), (1, 2))

    def test_same_words_but_a_different_value_is_not_a_duplicate(self):
        m = self.mem({"facts": [fact("user", "gym locker number", "214", "my locker number is 214")]},
                     {"facts": [fact("user", "gym locker number", "315", "locker number is now 315")]})
        m.add_message(chat("m1", "Btw my locker number is 214, I keep forgetting it."))
        m.add_message(chat("m2", "The gym locker number is now 315, they moved me."))
        self.assertEqual(len(rows(m)), 2)

    def test_exclusive_value_without_a_valid_replace_is_flagged_as_conflicting_and_both_stay_current(self):
        a = dict(fact("user", "preferred editor", "Emacs", "my preferred editor is Emacs", "PREFERENCE"), cardinality="EXCLUSIVE")
        b = dict(fact("user", "preferred editor", "VS Code", "my preferred editor is VS Code", "PREFERENCE"), cardinality="EXCLUSIVE")
        m = self.mem({"facts": [a]}, {"facts": [b]})
        m.add_message(chat("m1", "My preferred editor is Emacs, always has been.", ts="2026-11-01T10:00:00+05:30"))
        m.add_message(chat("m2", "Honestly my preferred editor is VS Code lately.", ts="2026-11-05T10:00:00+05:30"))
        old, new = rows(m)
        self.assertEqual((old["status"], new["status"], new["conflict_with"]), ("active", "active", old["id"]))
        m.llm = FakeLLM({"answer": "VS Code (noted later).", "know": True, "used": ["F2"]})
        m.answer("Which editor do I prefer?")
        self.assertIn("CONFLICTS with F1", m.llm.prompts[0][1]["content"])

    def test_members_of_a_set_never_conflict(self):
        a = dict(fact("user", "hobby", "ceramics", "I enjoy ceramics", "PREFERENCE"), cardinality="MULTI")
        b = dict(fact("user", "hobby", "cycling", "I enjoy cycling", "PREFERENCE"), cardinality="MULTI")
        m = self.mem({"facts": [a]}, {"facts": [b]})
        m.add_message(chat("m1", "I enjoy ceramics, it relaxes me."))
        m.add_message(chat("m2", "I enjoy cycling on weekends too."))
        self.assertEqual([r["conflict_with"] for r in rows(m)], [None, None])

    def test_a_replace_target_that_shares_nothing_with_the_new_fact_is_ignored(self):
        m = self.mem({"facts": [fact("sister", "city", "Pune", "my sister lives in Pune", "STATE")]},
                     {"facts": [dict(fact("manager", "birthday", "May 3", "my manager birthday is May 3", "STATE"), replaces=1)]})
        m.add_message(chat("m1", "My sister lives in Pune, near the old fort."))
        m.add_message(chat("m2", "My manager birthday is May 3, near the Pune trip date."))
        self.assertEqual([r["status"] for r in rows(m)], ["active", "active"])

    def test_a_backdated_statement_is_history_and_the_newer_effective_fact_stays_current(self):
        m = self.mem({"facts": [dict(fact("user", "city", "Pune", "I live in Pune now", "STATE"), when="2026-06-01")]},
                     {"facts": [dict(fact("user", "city", "Mumbai", "I lived in Mumbai before that", "STATE"), when="2026-03-01", replaces=1)]})
        m.add_message(chat("m1", "I live in Pune now, moved in June."))
        m.add_message(chat("m2", "Correction on my city: I lived in Mumbai before that, until June."))
        pune, mumbai = rows(m)
        self.assertEqual((pune["status"], mumbai["status"], mumbai["ended_ts"], pune["predecessor_id"]),
                         ("active", "superseded", "2026-06-01", mumbai["id"]))

    def test_a_time_of_day_is_kept_as_a_wall_clock_time_and_a_bad_one_is_dropped(self):
        m = self.mem({"facts": [dict(fact("user", "flight", "6E 2134", "my flight is at six", "EVENT"), when="2026-11-13T18:15"),
                                dict(fact("user", "gym", "evening", "gym in the evening", "PLAN"), when="2026-11-13T25:99")]})
        m.add_message(chat("m1", "Booked: my flight is at six, and gym in the evening."))
        got = {r["relation"]: (r["when_date"], r["when_ts"]) for r in rows(m)}
        self.assertEqual(got, {"flight": ("2026-11-13", "2026-11-13T18:15"), "gym": ("2026-11-13", None)})


class DerivedAndDependencyTests(Base):
    def setup_dentist(self, *more):
        m = self.mem({"facts": [dict(fact("user", "dentist appointment", "Thu 5 pm", "dentist appointment Thursday 5 pm", "EVENT"), when="2026-11-05")]},
                     {"facts": [dict(fact("user", "dentist reminder", "4 pm Thursday", "reminder an hour before the dentist", "DERIVED"), derived_from=[1])]},
                     *more)
        m.add_message(chat("m1", "Booked the dentist appointment Thursday 5 pm."))
        m.add_message(chat("m2", "Set a reminder an hour before the dentist please."))
        return m

    def test_a_derived_fact_records_the_facts_it_rests_on(self):
        m = self.setup_dentist()
        self.assertEqual([(r["kind"], r["status"]) for r in rows(m)], [("EVENT", "active"), ("DERIVED", "active")])
        self.assertEqual(m.repo.db.execute("SELECT child_id, parent_id FROM dependencies").fetchall()[0][:], (2, 1))

    def test_an_inference_with_no_valid_basis_is_not_stored(self):
        m = self.mem({"facts": [dict(fact("user", "mood", "stressed", "I feel stressed about work", "DERIVED"), derived_from=[])]})
        m.add_message(chat("m1", "I feel stressed about work lately, honestly."))
        self.assertEqual(len(rows(m)), 0)

    def test_changing_a_fact_makes_what_was_derived_from_it_stale_not_current(self):
        moved = dict(fact("user", "dentist appointment", "Fri 6 pm", "dentist appointment moved to Friday 6 pm", "EVENT"), when="2026-11-06", replaces=1)
        m = self.setup_dentist({"facts": [moved]})
        m.add_message(chat("m3", "The dentist appointment moved to Friday 6 pm, clinic called."))
        self.assertEqual([r["status"] for r in rows(m)], ["superseded", "stale", "active"])
        m.llm = FakeLLM({"answer": "Friday 6 pm; the old reminder no longer applies.", "know": True, "used": ["F3"]})
        m.answer("When is the dentist appointment and my reminder?")
        prompt = m.llm.prompts[0][1]["content"]
        self.assertIn("no longer valid since", prompt)
        self.assertNotIn("(CURRENT; user, derived", prompt)

    def test_forgetting_a_fact_also_removes_what_was_derived_from_it(self):
        m = self.setup_dentist({"facts": []}, {"forget": True, "fact_ids": [1], "source_ids": []})
        m.add_message(chat("m3", "Please forget the dentist appointment, delete it."))
        self.assertEqual(len(rows(m)), 0)
        self.assertEqual((m.repo.dangling(), self.n(m, "dependencies")), (0, 0))
        self.assertFalse(everything(m, "reminder an hour"))


class SemanticRetrievalTests(Base):
    def mem_sem(self, *replies, embedder=None):
        self.llm = FakeLLM(*replies)
        from backend.source.api.memory_system import MemorySystem
        return MemorySystem(llm_fn=self.llm, db_path=self.path, embedder=embedder or FakeEmbedder())

    def dentist(self, embedder=None):
        m = self.mem_sem({"facts": [fact("user", "dentist appointment", "Dr Shah Friday 6 pm", "dentist moved to Friday 6 pm", "EVENT")]},
                         {"facts": [fact("user", "gym locker number", "214", "my gym locker number is 214")]}, embedder=embedder)
        m.add_message(chat("m1", "Clinic called, dentist moved to Friday 6 pm."))
        m.add_message(chat("m2", "My gym locker number is 214, I keep forgetting."))
        return m

    def test_vectors_are_indexed_for_facts_and_sources_and_removed_on_forget(self):
        m = self.dentist()
        self.assertEqual(sorted(r[0] for r in m.repo.embeddings("")), ["f:1", "f:2", "s:m1", "s:m2"])
        m.llm = FakeLLM({"facts": []}, {"forget": True, "fact_ids": [1], "source_ids": ["m1"]})
        m.add_message(chat("m3", "Please forget the dentist appointment, delete it."))
        self.assertEqual(sorted(r[0] for r in m.repo.embeddings("")), ["f:2", "s:m2"])
        self.assertEqual(m.repo.dangling(), 0)

    def test_a_paraphrase_with_no_shared_word_and_no_keywords_is_found_semantically(self):
        m = self.dentist()
        m.llm = FakeLLM({"answer": "Friday 6 pm.", "know": True, "used": ["F1"]})   # keyword expansion returns no terms
        out = m.answer("When is my teeth doctor visit?")
        self.assertEqual([e["message_id"] for e in out["evidence"]], ["m1"])
        self.assertIn({"step": "semantic", "used": True}, out["trace"])

    def test_without_an_embedding_model_the_same_question_finds_nothing_lexically_and_abstains(self):
        m = self.dentist(embedder=type("NoModel", (), {"available": False, "embed": lambda s, t: None, "embed_one": lambda s, t: None})())
        m.llm = FakeLLM()
        out = m.answer("When is my teeth doctor visit?")
        self.assertEqual((out["evidence"], len(m.llm.prompts)), ([], 0))
        self.assertIn({"step": "semantic", "used": False}, out["trace"])

    def test_unrelated_semantic_neighbours_below_the_floor_are_not_candidates(self):
        m = self.dentist()
        m.llm = FakeLLM({"answer": "x", "know": False, "used": []})
        m.answer("Which airline and plane am I taking?")   # nothing stored about flights
        self.assertEqual(len(m.llm.prompts), 0)

    def test_reciprocal_rank_fusion_prefers_items_found_by_both_methods(self):
        self.assertEqual(fuse([1, 2, 3], [3, 1, 4]), [1, 3, 2, 4])
        self.assertEqual(fuse([1, 2], []), [1, 2])      # one method alone keeps its order (no semantic model)

    def test_the_other_members_of_a_set_come_with_the_one_that_matched(self):
        a = dict(fact("user", "hobby", "ceramics", "I enjoy ceramics", "PREFERENCE"), cardinality="MULTI")
        b = dict(fact("user", "hobby", "cycling", "I enjoy cycling", "PREFERENCE"), cardinality="MULTI")
        m = self.mem({"facts": [a]}, {"facts": [b]})
        m.add_message(chat("m1", "I enjoy ceramics, it relaxes me."))
        m.add_message(chat("m2", "I enjoy cycling on weekends too."))
        m.llm = FakeLLM({"answer": "Ceramics and cycling.", "know": True, "used": ["F1", "F2"]})
        m.answer("Do I enjoy ceramics?")   # cycling shares no word with the question
        self.assertIn("cycling", m.llm.prompts[0][1]["content"])

    def test_embedding_failure_never_loses_the_fact(self):
        m = self.dentist(embedder=type("Broken", (), {"available": True, "embed": lambda s, t: None, "embed_one": lambda s, t: None})())
        self.assertEqual(self.n(m, "facts"), 2)
        self.assertEqual(self.n(m, "embeddings"), 0)


if __name__ == "__main__":
    unittest.main()
