"""
Tests for the Kelvin6K assistant: database, machine-state services, RAG and
tool routing.

Run with:   python -m pytest tests -v
       or:  python tests/test_assistant.py     (no pytest needed)

Everything runs offline: a temporary database and RAG index are used, and the
assistant is forced onto its deterministic router so no API key is required.
"""

import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import config, db  # noqa: E402
from agent import query_router as qr  # noqa: E402
from agent.assistant import Assistant  # noqa: E402
from agent.rag.embeddings import _STOPWORDS, LexicalEmbedder, tokenize  # noqa: E402
from agent.rag.ingest import chunk_text, ingest_path  # noqa: E402
from agent.rag.retriever import get_store, search_documentation  # noqa: E402
from agent.services import machine_state as ms  # noqa: E402


class TempEnvironment(unittest.TestCase):
    """Isolates each test class in its own database + RAG index."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_db = config.DATABASE_PATH
        self._old_index = config.RAG_INDEX_DIR
        config.DATABASE_PATH = os.path.join(self._tmp.name, "test.db")
        config.RAG_INDEX_DIR = os.path.join(self._tmp.name, "index")
        db.init_db()
        ms.bind_live_state(None)
        get_store(refresh=True)

    def tearDown(self):
        config.DATABASE_PATH = self._old_db
        config.RAG_INDEX_DIR = self._old_index
        get_store(refresh=True)
        self._tmp.cleanup()


class TestPrintProgress(TempEnvironment):
    """1. print progress   2. remaining length calculation"""

    def test_progress_and_remaining(self):
        ms.start_print_job("Wall A", total_length_mm=1000.0, total_layers=60)
        ms.update_progress(completed_length_mm=650.0, current_layer=42)

        progress = ms.get_print_progress()
        self.assertTrue(progress["available"])
        self.assertEqual(progress["total_mm"], 1000.0)
        self.assertEqual(progress["completed_mm"], 650.0)
        self.assertEqual(progress["remaining_mm"], 350.0)  # computed, not stored
        self.assertEqual(progress["percentage"], 65.0)

    def test_remaining_never_negative(self):
        ms.start_print_job("Overrun", total_length_mm=100.0)
        ms.update_progress(completed_length_mm=140.0)
        self.assertEqual(ms.get_print_progress()["remaining_mm"], 0.0)

    def test_remaining_recomputed_after_update(self):
        ms.start_print_job("Wall B", total_length_mm=500.0)
        ms.update_progress(completed_length_mm=100.0)
        self.assertEqual(ms.get_print_progress()["remaining_mm"], 400.0)
        ms.update_progress(completed_length_mm=375.0)
        self.assertEqual(ms.get_print_progress()["remaining_mm"], 125.0)

    def test_no_job_reports_unavailable(self):
        progress = ms.get_print_progress()
        self.assertFalse(progress["available"])
        self.assertIn("No active print job", progress["reason"])


class TestMachineState(TempEnvironment):
    """3. current layer  4. layer width  5. machine status  9. empty state"""

    def test_current_layer(self):
        ms.start_print_job("Wall", total_length_mm=100.0)
        ms.update_progress(current_layer=42)
        layer = ms.get_current_layer()
        self.assertTrue(layer["available"])
        self.assertEqual(layer["current_layer"], 42)

    def test_layer_width(self):
        db.record_machine_state(printing=True, status="PRINTING",
                                layer_width_mm=18.5, layer_width_px=462.5, confidence=0.94)
        width = ms.get_current_layer_width()
        self.assertTrue(width["available"])
        self.assertEqual(width["layer_width_mm"], 18.5)

    def test_layer_width_without_calibration_is_not_reported_as_mm(self):
        db.record_machine_state(printing=True, status="PRINTING",
                                layer_width_mm=None, layer_width_px=462.5, confidence=0.9)
        width = ms.get_current_layer_width()
        self.assertFalse(width["available"])
        self.assertEqual(width["width_px"], 462.5)
        self.assertIn("calibration", width["reason"].lower())

    def test_low_confidence_is_flagged(self):
        db.record_machine_state(printing=True, status="PRINTING",
                                layer_width_mm=18.5, confidence=0.2)
        width = ms.get_current_layer_width()
        self.assertIsNotNone(width["note"])
        self.assertIn("uncertain", width["note"].lower())

    def test_print_status(self):
        db.record_machine_state(printing=True, status="PRINTING", confidence=0.9)
        status = ms.get_current_print_status()
        self.assertTrue(status["available"])
        self.assertTrue(status["printing"])

    def test_machine_status(self):
        db.record_machine_state(printing=False, status="NO_PRINT",
                                confidence=0.8, camera_connected=True)
        self.assertTrue(ms.get_machine_status()["available"])

    def test_empty_state_is_unavailable_not_fabricated(self):
        for result in (ms.get_current_print_status(), ms.get_current_layer_width(),
                       ms.get_machine_status()):
            self.assertFalse(result["available"])
            self.assertIn("reason", result)

    def test_stale_state_is_rejected(self):
        old = time.time() - (config.MACHINE_STATE_MAX_AGE_SECONDS + 60)
        db.record_machine_state(printing=True, status="PRINTING",
                                layer_width_mm=18.5, confidence=0.9, timestamp=old)
        self.assertFalse(ms.get_current_print_status()["available"])


class TestPersistence(TempEnvironment):
    """6. database persistence"""

    def test_state_and_measurements_persist(self):
        job_id = db.create_print_job("Persist", total_length_mm=200.0)
        db.record_machine_state(printing=True, status="PRINTING",
                                layer_width_mm=12.0, confidence=0.8, job_id=job_id)
        db.record_layer_measurement(layer_number=7, width_px=300.0,
                                    width_mm=12.0, confidence=0.8, job_id=job_id)
        db.record_event("alert", "layer width high", job_id)

        self.assertIsNotNone(db.get_latest_machine_state())
        self.assertEqual(len(db.get_recent_layer_measurements()), 1)
        self.assertEqual(db.get_recent_events()[0]["event_type"], "alert")

    def test_generated_remaining_column(self):
        job_id = db.create_print_job("Gen", total_length_mm=800.0)
        db.update_job_progress(job_id, completed_length_mm=250.0)
        self.assertEqual(db.get_job(job_id)["remaining_length_mm"], 550.0)

    def test_survives_reconnect(self):
        db.create_print_job("Restart", total_length_mm=300.0)
        # New connection (simulates a different process reading the same file).
        self.assertEqual(db.get_active_job()["job_name"], "Restart")


class TestRag(TempEnvironment):
    """7. RAG retrieval   8. missing information"""

    def _write_doc(self, name: str, text: str) -> str:
        path = os.path.join(self._tmp.name, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def test_ingest_and_retrieve_with_source(self):
        self._write_doc(
            "kelvin6k_manual.md",
            "# Nozzle Maintenance\n"
            "To clear a nozzle clog, stop the pump, release pressure, then soak "
            "the nozzle in solvent for ten minutes and clear it with the cleaning rod.\n\n"
            "# Calibration\n"
            "Set PIXELS_PER_MM using a reference block of known width.\n",
        )
        result = ingest_path(self._tmp.name)
        self.assertTrue(result["ok"])
        self.assertGreater(result["chunks"], 0)

        get_store(refresh=True)
        found = search_documentation("how do I clear a nozzle clog")
        self.assertTrue(found["found"])
        self.assertIn("nozzle", found["results"][0]["text"].lower())
        self.assertIn("kelvin6k_manual.md", found["results"][0]["source"])
        self.assertEqual(found["results"][0]["doc_type"], "manual")

    def test_missing_information_is_reported_not_invented(self):
        self._write_doc("sop_startup.md", "# Startup\nPower on the controller first.\n")
        ingest_path(self._tmp.name)
        get_store(refresh=True)

        found = search_documentation("quantum entanglement warp drive calibration")
        self.assertFalse(found["found"])
        self.assertEqual(found["results"], [])
        self.assertIn("not found", found["message"].lower())

    def test_weak_word_overlap_is_rejected_not_scored_perfect(self):
        """
        A chunk that shares one incidental word with the question must NOT be
        returned as an answer.

        This is the regression: scores used to be normalised against the best
        hit, so the top result was 1.0 by construction and RAG_MIN_SCORE could
        never reject it. Any question sharing a single word with any document
        got a confident, wrong passage instead of "not in the documentation".
        """
        self._write_doc(
            "kelvin6k_manual.md",
            "# Calibration\n"
            "Calibrate the camera with a reference block of known width.\n",
        )
        ingest_path(self._tmp.name)
        get_store(refresh=True)

        # Shares only "calibration" with the document; the subject is unrelated.
        found = search_documentation("what is the warp drive calibration procedure")
        self.assertFalse(
            found["found"],
            "a single shared word must not be reported as a documented answer",
        )
        self.assertEqual(found["results"], [])

        # The genuine question still retrieves the same document.
        answered = search_documentation("how do I calibrate with a reference block")
        self.assertTrue(answered["found"])
        self.assertIn("calibrate", answered["results"][0]["text"].lower())

    def test_scores_are_absolute_not_relative_to_the_best_hit(self):
        """The top hit must be scored on merit, never pinned to 1.0."""
        scorer = LexicalEmbedder()
        chunks = [
            "Stop the pump and release pressure before clearing a nozzle clog.",
            "Level the sand surface and place polythene before printing.",
        ]
        strong = scorer.score("nozzle clog", chunks)
        weak = scorer.score("clog", ["Printing requires a level surface."])

        self.assertLess(max(strong), 1.0, "the top hit must not be normalised to 1.0")
        self.assertGreater(max(strong), config.RAG_MIN_SCORE)
        self.assertEqual(max(weak), 0.0, "no shared term must score above zero")
        # Padding a question with stopwords must not change its score: the
        # subject decides relevance, not how politely it was asked.
        verbose = scorer.score("what should I do about the nozzle clog", chunks)
        self.assertAlmostEqual(max(strong), max(verbose), delta=0.01)

    def test_empty_index_reports_no_documents(self):
        get_store(refresh=True)
        found = search_documentation("anything")
        self.assertFalse(found["found"])
        self.assertIn("no documentation", found["message"].lower())

    def test_chunking_overlaps_and_covers(self):
        text = "\n\n".join(f"Paragraph {i} with some content about printing." for i in range(60))
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c[0]) <= 400 for c in chunks))


class TestAssistantToolCalling(TempEnvironment):
    """10. assistant tool calling (offline deterministic router)"""

    def setUp(self):
        super().setUp()
        ms.start_print_job("Wall", total_length_mm=1000.0, total_layers=60)
        ms.update_progress(completed_length_mm=650.0, current_layer=42)
        db.record_machine_state(printing=True, status="PRINTING", current_layer=42,
                                layer_width_mm=18.5, layer_width_px=462.5,
                                confidence=0.94, camera_connected=True)
        self.assistant = Assistant(use_llm=False)

    def test_routes_to_correct_tools(self):
        cases = [
            ("kitni wall print ho gayi hai?", "get_print_progress"),
            ("abhi kitni baki hai?", "get_print_progress"),
            ("printing chal rahi hai?", "get_current_print_status"),
            ("current layer kya hai?", "get_current_layer"),
            ("current layer width kitni hai?", "get_current_layer_width"),
            ("how much is printed?", "get_print_progress"),
            ("is printing running?", "get_current_print_status"),
        ]
        for question, expected in cases:
            with self.subTest(question=question):
                self.assertEqual(Assistant.route(question), expected)

    def test_answers_contain_real_values(self):
        self.assertIn("650", self.assistant.ask("kitni wall print ho gayi hai?"))
        self.assertIn("350", self.assistant.ask("abhi kitni baki hai?"))
        self.assertIn("42", self.assistant.ask("current layer kya hai?"))
        self.assertIn("18.5", self.assistant.ask("layer width kitni hai?"))
        self.assertIn("running", self.assistant.ask("printing chal rahi hai?").lower())

    def test_no_fabrication_when_state_missing(self):
        empty = Assistant(use_llm=False)
        config.DATABASE_PATH = os.path.join(self._tmp.name, "empty.db")
        db.init_db()
        answer = empty.ask("layer width kitni hai?")
        self.assertNotIn("18.5", answer)
        self.assertTrue(
            "don't have" in answer.lower() or "not" in answer.lower(),
            f"Expected an unavailability answer, got: {answer}",
        )


class TestRouterKnowledgeRouting(unittest.TestCase):
    """
    11. router: printing-knowledge questions must reach RAG, live questions
        must stay live.

    These are classification tests, so they need no database or index -- they
    assert which intent a question is given, which is where the bug was: the
    knowledge existed in the index but the router never asked for it.
    """

    RAG_INTENTS = {qr.Intent.KNOWLEDGE_QUERY, qr.Intent.COMBINED_QUERY}

    def _intent(self, question: str) -> "qr.Intent":
        return qr.classify(question)[0]

    def test_printing_problem_questions_reach_rag(self):
        """A troubleshooting or how-to question must be answered from the docs."""
        questions = [
            "Why is there a gap in my printed layer?",
            "How often should I clean the accelerator hole?",
            "My hose is repeatedly choking, what should I do?",
            "Is 40 mm or 50 mm correct?",
            "40 mm or 50 mm layer thickness?",
            "Z-axis servo E08.5 error aa raha hai",
            "How do I clear a nozzle clog?",
            "Layer thickness is uneven, what should I check?",
        ]
        for question in questions:
            with self.subTest(question=question):
                intent = self._intent(question)
                self.assertIn(
                    intent, self.RAG_INTENTS,
                    f"{question!r} routed to {intent.value}, so RAG is never consulted",
                )

    def test_defect_question_never_returns_a_production_number(self):
        """
        The specific dangerous regression.

        "Why is there a gap in my printed layer?" used to reach the fallback
        keyword table, match the bare word "layer", and be answered with
        "Current layer is 42 out of 60" -- a confident, plausible, unrelated
        answer to a defect report.
        """
        intent = self._intent("Why is there a gap in my printed layer?")
        self.assertNotEqual(intent, qr.Intent.GET_CURRENT_LAYER)
        self.assertIn(intent, self.RAG_INTENTS)

        # Same trap via other bare fallback keywords.
        for question in (
            "there is a void in the printed wall",
            "the printed wall has a crack, what should I do?",
        ):
            with self.subTest(question=question):
                self.assertIn(self._intent(question), self.RAG_INTENTS)

    def test_live_queries_are_unchanged(self):
        """
        Every live question that worked before the knowledge-routing change
        must still resolve to the same live intent.
        """
        expected = {
            "What is the current layer?": qr.Intent.GET_CURRENT_LAYER,
            "current layer kya hai?": qr.Intent.GET_CURRENT_LAYER,
            "abhi kaunsi layer chal rahi hai": qr.Intent.GET_CURRENT_LAYER,
            "How much printing is completed?": qr.Intent.GET_COMPLETED_PRODUCTION,
            "kitni wall print ho gayi hai?": qr.Intent.GET_COMPLETED_PRODUCTION,
            "how much is printed?": qr.Intent.GET_COMPLETED_PRODUCTION,
            "aaj kitna print hua": qr.Intent.GET_24_HOUR_PRODUCTION,
            "kitni layers ho gayi hain": qr.Intent.GET_CURRENT_LAYER,
            "abhi kitni baki hai?": qr.Intent.GET_REMAINING_PRODUCTION,
            "how much is remaining": qr.Intent.GET_REMAINING_PRODUCTION,
            "printing chal rahi hai?": qr.Intent.GET_MACHINE_STATUS,
            "is printing running?": qr.Intent.GET_MACHINE_STATUS,
            "what is the machine status": qr.Intent.GET_MACHINE_STATUS,
            "machine ka status kya hai": qr.Intent.GET_MACHINE_STATUS,
            # layer width is deliberately NOT a technical topic: it names a live
            # measurement the router already resolves.
            "current layer width kitni hai?": qr.Intent.GET_LAYER_WIDTH,
            "layer width kitni hai?": qr.Intent.GET_LAYER_WIDTH,
            "what is the current layer width": qr.Intent.GET_LAYER_WIDTH,
            "what is the layer width": qr.Intent.GET_LAYER_WIDTH,
            "camera ka status kya hai": qr.Intent.GET_CAMERA_STATUS,
            "robot ka status kya hai": qr.Intent.GET_ROBOT_STATUS,
            "what is the total target": qr.Intent.GET_TOTAL_TARGET,
            "total kitna target hai": qr.Intent.GET_TOTAL_TARGET,
            "who is the operator": qr.Intent.GET_OPERATOR,
            "kitna material use hua hai": qr.Intent.GET_MATERIAL_USAGE,
            "eta kya hai": qr.Intent.GET_ETA,
            "how long has it been printing": qr.Intent.GET_ELAPSED_TIME,
            "what is the print quality": qr.Intent.GET_PRINT_QUALITY,
            "print rate kya hai": qr.Intent.GET_PRINT_RATE,
            # Bare "problem"/"issue" are excluded from _PROBLEM_TERMS precisely
            # so these two keep returning the live problem list.
            "kya koi problem hai": qr.Intent.GET_RECENT_PROBLEMS,
            "recent problems kya hain": qr.Intent.GET_RECENT_PROBLEMS,
        }
        for question, want in expected.items():
            with self.subTest(question=question):
                self.assertEqual(self._intent(question), want)

    def test_hinglish_defect_questions_reach_rag(self):
        """
        The operator utterances documented in the troubleshooting dataset.
        These carry the defect meaning in Hinglish words ("patli", "chipak
        nahi rahe") that no English keyword list would catch.
        """
        for question in (
            "layer patli aa rahi hai",
            "material zyada spread ho raha hai",
            "layers chipak nahi rahe beech mein gap aa raha hai",
            "ek layer dusri layer se shift ho gayi",
            "nozzle se material kam aa raha hai shayad nozzle choke hai",
            "material flow drop ho gaya hai",
            "pipe ko last kab clean kiya tha",
        ):
            with self.subTest(question=question):
                intent = self._intent(question)
                self.assertIn(
                    intent, self.RAG_INTENTS,
                    f"{question!r} routed to {intent.value}",
                )

    def test_unrelated_question_is_still_refused(self):
        """Widening the vocabulary must not make the router accept anything."""
        for question in (
            "Tell me something about quantum entanglement.",
            "who won the cricket match",
        ):
            with self.subTest(question=question):
                self.assertEqual(self._intent(question), qr.Intent.UNKNOWN_QUERY)


@unittest.skipUnless(
    os.path.isfile(os.path.join("rag_index", "index.json")),
    "needs the ingested knowledge index (run: python ingest_docs.py docs/)",
)
class TestKnowledgeRetrievalEndToEnd(unittest.TestCase):
    """
    12. router + RAG together: a routed question must retrieve the record that
        actually answers it, from the real ingested index.
    """

    def setUp(self):
        # The RAG test class points the store at a temp directory; make sure we
        # are reading the real ingested index.
        config.RAG_INDEX_DIR = "rag_index"
        get_store(refresh=True)

    def _top_source(self, question: str) -> str:
        result = search_documentation(question)
        self.assertTrue(result["found"], f"nothing retrieved for {question!r}")
        return result["results"][0]["source"]

    def test_conflicting_layer_thickness_reaches_the_conflict_record(self):
        """
        The 40 mm / 50 mm question must reach the record that FLAGS the
        conflict, not a live layer intent and not one side of the conflict.
        """
        for question in ("40 mm or 50 mm layer thickness?", "Is 40 mm or 50 mm correct?"):
            with self.subTest(question=question):
                self.assertIn(qr.classify(question)[0], TestRouterKnowledgeRouting.RAG_INTENTS)
                self.assertIn("conflicts.md", self._top_source(question))

    def test_servo_fault_reaches_the_servo_fault_record(self):
        source = self._top_source("Z-axis servo E08.5 error aa raha hai")
        self.assertIn("E08.5", source)

    def test_accelerator_hole_reaches_a_cleaning_record(self):
        source = self._top_source("How often should I clean the accelerator hole?")
        self.assertIn("ccelerator", source)

    def test_hose_choking_reaches_the_hose_record(self):
        source = self._top_source("My hose is repeatedly choking, what should I do?")
        self.assertIn("Hose", source)


@unittest.skipUnless(
    os.path.isfile(os.path.join("rag_index", "index.json")),
    "needs the ingested knowledge index (run: python ingest_docs.py docs/)",
)
class TestSpokenAnswerStyle(unittest.TestCase):
    """
    13. spoken style: a retrieved record must reach the operator as speech, not
        as a Markdown chunk.

    The router used to return the raw chunk, so the assistant said "From
    operational_rules.md, Issue: ...: ## Issue: ... **Problem:** Four different
    parts are cleaned on four differ" -- truncated mid-word, with the actual
    answer discarded below the cut.
    """

    # Everything here is either Markdown syntax or an internal retrieval detail.
    FORBIDDEN = ("##", "**", "```", ".md", "From ", "Issue:", "Problem:",
                 "Symptoms", "Cause:", "Verification Status", "Source:",
                 "English Questions", "Hinglish Questions")

    KNOWLEDGE_QUESTIONS = (
        "How do I clear a nozzle clog?",
        "How often should I clean the accelerator hole?",
        "My hose is repeatedly choking, what should I do?",
        "Why is there a gap in my printed layer?",
        "40 mm or 50 mm layer thickness?",
        "Is 40 mm or 50 mm correct?",
        "Z-axis servo E08.5 error aa raha hai",
        "Layer thickness is uneven, what should I check?",
        "What do I use if the dosing pump is not available?",
        "How should I prepare the base before printing?",
        "layer patli aa rahi hai",
        "material flow drop ho gaya hai",
        "there is a crack in the wall, what should I do?",
        "pipe ko last kab clean kiya tha",
        "how do I fix the E-axis mixer?",
    )

    def setUp(self):
        config.RAG_INDEX_DIR = "rag_index"
        get_store(refresh=True)

    def _answer(self, question: str) -> str:
        return qr.summarise(qr.route(question))

    def test_no_markdown_or_internal_details_in_any_answer(self):
        for question in self.KNOWLEDGE_QUESTIONS:
            with self.subTest(question=question):
                answer = self._answer(question)
                for token in self.FORBIDDEN:
                    self.assertNotIn(
                        token, answer,
                        f"{token!r} leaked into the answer for {question!r}: {answer[:120]!r}",
                    )

    def test_no_answer_ends_mid_word(self):
        """The old code sliced at 320 characters, which cut words in half."""
        for question in self.KNOWLEDGE_QUESTIONS:
            with self.subTest(question=question):
                answer = self._answer(question).strip()
                self.assertTrue(
                    answer.endswith((".", "!", "?")),
                    f"answer for {question!r} does not end on a sentence: ...{answer[-40:]!r}",
                )

    def test_accelerator_answer_gives_the_interval_tool_and_warning(self):
        answer = self._answer("How often should I clean the accelerator hole?")
        lower = answer.lower()
        self.assertIn("5 layers", lower)
        self.assertIn("allen key", lower)
        self.assertIn("wire", lower, "the 'do not use wire' warning must survive")

    def test_nozzle_clog_answer_keeps_the_pressure_warning(self):
        """Safety instruction from a label-free prose document."""
        answer = self._answer("How do I clear a nozzle clog?")
        lower = answer.lower()
        self.assertIn("pressuris", lower)
        self.assertIn("pump", lower)

    def test_layer_gap_answer_preserves_uncertainty(self):
        answer = self._answer("Why is there a gap in my printed layer?").lower()
        self.assertTrue(
            "possible" in answer or "not confirmed" in answer,
            "an unconfirmed cause must not be spoken as a confirmed one",
        )

    def test_conflicting_thickness_answer_states_the_conflict(self):
        for question in ("40 mm or 50 mm layer thickness?", "Is 40 mm or 50 mm correct?"):
            with self.subTest(question=question):
                answer = self._answer(question)
                lower = answer.lower()
                self.assertIn("40", answer)
                self.assertIn("50", answer)
                self.assertTrue(
                    "conflict" in lower or "both" in lower,
                    f"the conflict must be stated, not resolved: {answer[:140]!r}",
                )

    def test_machine_specific_procedure_says_authorization(self):
        answer = self._answer("Z-axis servo E08.5 error aa raha hai").lower()
        self.assertIn("authoriz", answer)

    def test_temporary_workaround_is_labelled_temporary(self):
        answer = self._answer("What do I use if the dosing pump is not available?").lower()
        self.assertIn("temporar", answer)

    def test_label_free_document_still_answers(self):
        """
        The hand-written manual and troubleshooting guide carry no field
        labels, so they take the cleaned-prose fallback path.
        """
        answer = self._answer("How do I clear a nozzle clog?")
        self.assertGreater(len(answer), 60)
        self.assertNotIn("Nozzle Clog Procedure:", answer)

    def test_live_answers_are_untouched(self):
        """The formatter must not reach any live-data branch of summarise()."""
        expected = {
            "What is the current layer?": "Current layer is 42 out of 60.",
            "How much printing is completed?":
                "650.0 millimeter printed out of 1000.0 millimeter (65.0 percent).",
            # The seeded database holds a job_started event and no alerts, so
            # this is the live wording, byte-for-byte, before and after.
            "kya koi problem hai":
                "Most recent event: Started job 'Demo Wall' (1000.0 mm)",
        }
        for question, want in expected.items():
            with self.subTest(question=question):
                self.assertEqual(self._answer(question), want)


@unittest.skipUnless(
    os.path.isfile(os.path.join("rag_index", "index.json")),
    "needs the ingested knowledge index (run: python ingest_docs.py docs/)",
)
class TestDocumentationBeforeFallback(unittest.TestCase):
    """
    14. router: ask the documentation before the fallback keyword table.

    `_fallback_keywords` maps one bare word to a live intent, so a documented
    question was answered with a production number -- "How should the wall be
    lifted?" matched "wall" and returned "650.0 millimeter printed out of
    1000.0 millimeter". Retrieval is now consulted first, but only when it has
    a confident answer.
    """

    def setUp(self):
        config.RAG_INDEX_DIR = "rag_index"
        get_store(refresh=True)

    def _intent(self, question: str) -> "qr.Intent":
        return qr.classify(question)[0]

    def test_single_informative_word_live_query_stays_live(self):
        """
        "who is the operator" retrieves an unrelated record at 0.734 -- high,
        because a one-word query scores near its own ceiling against any
        document containing that word. The term requirement is what screens it
        out; no score threshold could.
        """
        self.assertEqual(self._intent("who is the operator"), qr.Intent.GET_OPERATOR)

        informative = [
            t for t in tokenize(qr._normalise("who is the operator"))
            if t not in _STOPWORDS
        ]
        self.assertLess(
            len(informative), qr.FALLBACK_RAG_MIN_TERMS,
            "this query is protected by the term count; if that changes, the "
            "gate no longer protects it",
        )

    def test_low_scoring_live_query_stays_live(self):
        """
        "kitna material use hua hai" retrieves an irrelevant record at ~0.219.
        RAG_MIN_SCORE (0.15) would let that through, which is exactly why this
        gate uses its own, higher threshold.
        """
        self.assertEqual(
            self._intent("kitna material use hua hai"), qr.Intent.GET_MATERIAL_USAGE
        )
        result = search_documentation("kitna material use hua hai")
        if result.get("found"):
            self.assertLess(
                result["results"][0]["score"], qr.FALLBACK_RAG_MIN_SCORE,
                "this query is protected by the score threshold",
            )

    def test_the_gate_is_not_rag_min_score(self):
        """The two thresholds are separate decisions and must not be conflated."""
        self.assertGreater(qr.FALLBACK_RAG_MIN_SCORE, config.RAG_MIN_SCORE)
        self.assertEqual(config.RAG_MIN_SCORE, 0.15, "RAG_MIN_SCORE must not change")

    def test_documented_questions_no_longer_get_a_production_number(self):
        """
        Representative sample of the questions this change fixes. Each one
        previously matched a bare fallback keyword and was answered with live
        production or status data.
        """
        previously_hijacked = {
            "How should the wall be lifted?": "wall",
            "Wall collapse ka kya reason hai?": "wall",
            "What planning is needed before a print?": "print",
            "How should I prepare the base before printing?": "printing",
            "How long does one layer take?": "layer",
            "Purana material use kar sakte hain?": "material",
            # NOTE: "The camera image is poor." is NOT here. It matches a
            # _PATTERNS phrase, so it exits on the best_intent path and never
            # reaches this gate -- one of the cases this change deliberately
            # does not address.
            "How do I view the camera remotely?": "camera",
            "Is the open wire safe?": None,
            "Which G92 command should I use?": None,
            "What must I do at shift handover?": None,
            "What feedrate should I use?": None,
            "Is 150 percent the maximum?": None,
        }
        for question in previously_hijacked:
            with self.subTest(question=question):
                intent = self._intent(question)
                self.assertIn(
                    intent,
                    (qr.Intent.KNOWLEDGE_QUERY, qr.Intent.COMBINED_QUERY),
                    f"{question!r} routed to {intent.value}",
                )
                answer = qr.summarise(qr.route(question))
                self.assertNotIn(
                    "millimeter printed out of", answer,
                    "a documented question must not be answered with production output",
                )

    def test_all_live_regression_queries_keep_their_intent(self):
        """The 30-query live set, unchanged."""
        expected = {
            "What is the current layer?": qr.Intent.GET_CURRENT_LAYER,
            "current layer kya hai?": qr.Intent.GET_CURRENT_LAYER,
            "abhi kaunsi layer chal rahi hai": qr.Intent.GET_CURRENT_LAYER,
            "kitni layers ho gayi hain": qr.Intent.GET_CURRENT_LAYER,
            "How much printing is completed?": qr.Intent.GET_COMPLETED_PRODUCTION,
            "kitni wall print ho gayi hai?": qr.Intent.GET_COMPLETED_PRODUCTION,
            "how much is printed?": qr.Intent.GET_COMPLETED_PRODUCTION,
            "aaj kitna print hua": qr.Intent.GET_24_HOUR_PRODUCTION,
            "abhi kitni baki hai?": qr.Intent.GET_REMAINING_PRODUCTION,
            "how much is remaining": qr.Intent.GET_REMAINING_PRODUCTION,
            "printing chal rahi hai?": qr.Intent.GET_MACHINE_STATUS,
            "is printing running?": qr.Intent.GET_MACHINE_STATUS,
            "what is the machine status": qr.Intent.GET_MACHINE_STATUS,
            "machine ka status kya hai": qr.Intent.GET_MACHINE_STATUS,
            "current layer width kitni hai?": qr.Intent.GET_LAYER_WIDTH,
            "layer width kitni hai?": qr.Intent.GET_LAYER_WIDTH,
            "what is the current layer width": qr.Intent.GET_LAYER_WIDTH,
            "what is the layer width": qr.Intent.GET_LAYER_WIDTH,
            "camera ka status kya hai": qr.Intent.GET_CAMERA_STATUS,
            "robot ka status kya hai": qr.Intent.GET_ROBOT_STATUS,
            "what is the total target": qr.Intent.GET_TOTAL_TARGET,
            "total kitna target hai": qr.Intent.GET_TOTAL_TARGET,
            "who is the operator": qr.Intent.GET_OPERATOR,
            "kitna material use hua hai": qr.Intent.GET_MATERIAL_USAGE,
            "eta kya hai": qr.Intent.GET_ETA,
            "how long has it been printing": qr.Intent.GET_ELAPSED_TIME,
            "what is the print quality": qr.Intent.GET_PRINT_QUALITY,
            "print rate kya hai": qr.Intent.GET_PRINT_RATE,
            "kya koi problem hai": qr.Intent.GET_RECENT_PROBLEMS,
            "recent problems kya hain": qr.Intent.GET_RECENT_PROBLEMS,
        }
        self.assertEqual(len(expected), 30)
        for question, want in expected.items():
            with self.subTest(question=question):
                self.assertEqual(self._intent(question), want)

    def test_retrieval_failure_does_not_change_live_routing(self):
        """
        The check is advisory. If retrieval raises, the fallback table must
        behave exactly as it did before.
        """
        import agent.query_router as module

        original = module.search_documentation
        try:
            def boom(*_a, **_k):
                raise RuntimeError("index unavailable")
            module.search_documentation = boom
            from unittest.mock import patch
            with patch("agent.config.GEMINI_API_KEY", ""):
                self.assertEqual(self._intent("who is the operator"), qr.Intent.GET_OPERATOR)
                # A documented question falls back to its old (wrong) live intent
                # rather than failing -- degraded, never broken.
                self.assertEqual(
                    self._intent("How should the wall be lifted?"),
                    qr.Intent.GET_COMPLETED_PRODUCTION,
                )
        finally:
            module.search_documentation = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
