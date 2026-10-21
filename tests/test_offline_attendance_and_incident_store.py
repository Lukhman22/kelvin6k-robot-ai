"""
tests/test_offline_attendance_and_incident_store.py
Comprehensive test suite verifying:
1. PART A: Deterministic local attendance commands that NEVER call Gemini.
2. PART B: Persistent storage and retrieval of uploaded machine-problem images
   and pre-Gemini historical incident search.
"""

import datetime
import io
import json
import os
import shutil
import time
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import agent
from agent import config, db, incident_store, shift_store
from agent.assistant import Assistant
from agent.troubleshooter import Attachment, extract_local_cv_observations, troubleshoot
from api.server import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Test Fixtures & Sample Data
# ---------------------------------------------------------------------------

@pytest.fixture
def test_env(tmp_path):
    """Sets up an isolated SQLite database and incident images storage directory."""
    db_file = tmp_path / "test_kelvin6k.db"
    db_path = str(db_file)
    db.init_db(db_path)

    img_dir = tmp_path / "test_incident_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_dir_path = str(img_dir)

    orig_db = config.DATABASE_PATH
    orig_img_dir = config.INCIDENT_IMAGES_DIR

    config.DATABASE_PATH = db_path
    config.INCIDENT_IMAGES_DIR = img_dir_path

    # Register standard test operators
    shift_store.register_operator(operator_id="OP_AROUND", name="Around", active=1, db_path=db_path)
    shift_store.register_operator(operator_id="OP_HAMADAN", name="Hamadan", active=1, db_path=db_path)
    shift_store.register_operator(operator_id="OP_ANAND", name="Anand Sharma", active=1, db_path=db_path)

    yield {
        "db_path": db_path,
        "img_dir": img_dir_path,
    }

    config.DATABASE_PATH = orig_db
    config.INCIDENT_IMAGES_DIR = orig_img_dir
    if db_file.exists():
        db_file.unlink()
    if img_dir.exists():
        shutil.rmtree(img_dir, ignore_errors=True)


def _make_dummy_image_bytes(width: int = 120, height: int = 100, color: int = 128) -> bytes:
    """Creates valid JPEG bytes using OpenCV."""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    # Draw simple concrete bead pattern
    cv2.rectangle(img, (10, 40), (110, 60), (200, 200, 200), -1)
    success, encoded = cv2.imencode(".jpg", img)
    assert success
    return encoded.tobytes()


# ===========================================================================
# PART A: Deterministic Local Attendance Tests (Zero Gemini Invocation)
# ===========================================================================

class TestOfflineAttendance:
    """Verifies attendance commands resolve deterministically without calling Gemini."""

    def test_shift_start_variants_offline(self, test_env):
        assistant = Assistant(use_llm=False)

        # Mock Gemini endpoints to guarantee ZERO external calls
        with patch("agent.nlu._call_semantic_model") as mock_semantic, \
             patch("agent.assistant.Assistant._ask_llm") as mock_ask_llm:

            mock_semantic.side_effect = AssertionError("Gemini semantic model must NOT be called for attendance!")
            mock_ask_llm.side_effect = AssertionError("Gemini LLM must NOT be called for attendance!")

            # 1. "Start Around's shift."
            ans1 = assistant.ask("Start Around's shift.", session_id="s1")
            assert "around's shift has started" in ans1.lower()
            rec1 = shift_store.get_active_attendance(operator_id="OP_AROUND", db_path=test_env["db_path"])
            assert rec1 is not None
            assert rec1["status"] == "ACTIVE"
            assert rec1["employee_name"] == "Around"

            # 2. "Start shift for Hamadan."
            ans2 = assistant.ask("Start shift for Hamadan.", session_id="s2")
            assert "hamadan's shift has started" in ans2.lower()
            rec2 = shift_store.get_active_attendance(operator_id="OP_HAMADAN", db_path=test_env["db_path"])
            assert rec2 is not None
            assert rec2["status"] == "ACTIVE"

            # 3. Prevent duplicate active shift
            dup_ans = assistant.ask("Start Around's shift.", session_id="s1")
            assert "already has an active shift" in dup_ans.lower()

            assert mock_semantic.call_count == 0
            assert mock_ask_llm.call_count == 0

    def test_clock_in_and_hinglish_start_offline(self, test_env):
        assistant = Assistant(use_llm=False)

        with patch("agent.nlu._call_semantic_model") as mock_semantic, \
             patch("agent.assistant.Assistant._ask_llm") as mock_ask_llm:

            # "Clock in Around."
            ans = assistant.ask("Clock in Around.", session_id="s1")
            assert "around's shift has started" in ans.lower()
            rec = shift_store.get_active_attendance(operator_id="OP_AROUND", db_path=test_env["db_path"])
            assert rec is not None

            # Hinglish: "Kelvin, Hamadan ka shift start karo"
            ans_h = assistant.ask("Kelvin, Hamadan ka shift start karo", session_id="s2")
            assert "hamadan's shift has started" in ans_h.lower()
            rec_h = shift_store.get_active_attendance(operator_id="OP_HAMADAN", db_path=test_env["db_path"])
            assert rec_h is not None

            assert mock_semantic.call_count == 0
            assert mock_ask_llm.call_count == 0

    def test_shift_end_variants_offline(self, test_env):
        assistant = Assistant(use_llm=False)

        # Start shifts first
        shift_store.create_attendance_record("OP_AROUND", "Around", shift_store.local_today(), db_path=test_env["db_path"])
        shift_store.create_attendance_record("OP_HAMADAN", "Hamadan", shift_store.local_today(), db_path=test_env["db_path"])

        with patch("agent.nlu._call_semantic_model") as mock_semantic, \
             patch("agent.assistant.Assistant._ask_llm") as mock_ask_llm:

            # 1. "End Around's shift."
            ans1 = assistant.ask("End Around's shift.", session_id="s1")
            assert "around's shift has ended" in ans1.lower()
            assert "duration" in ans1.lower()
            rec1 = shift_store.get_active_attendance(operator_id="OP_AROUND", db_path=test_env["db_path"])
            assert rec1 is None  # Ended

            # 2. "Clock out Hamadan."
            ans2 = assistant.ask("Clock out Hamadan.", session_id="s2")
            assert "hamadan's shift has ended" in ans2.lower()
            rec2 = shift_store.get_active_attendance(operator_id="OP_HAMADAN", db_path=test_env["db_path"])
            assert rec2 is None

            # Ending an already ended shift
            ans_again = assistant.ask("End Around's shift.", session_id="s1")
            assert "no active shift found for around" in ans_again.lower()

            assert mock_semantic.call_count == 0
            assert mock_ask_llm.call_count == 0

    def test_end_my_shift_ambiguity_never_guesses(self, test_env):
        """'End my shift' must NOT guess the first/only active employee if no session user exists."""
        assistant = Assistant(use_llm=False)

        # Start one active shift
        shift_store.create_attendance_record("OP_AROUND", "Around", shift_store.local_today(), db_path=test_env["db_path"])

        with patch("agent.nlu._call_semantic_model") as mock_semantic, \
             patch("agent.assistant.Assistant._ask_llm") as mock_ask_llm:

            # Unauthenticated / unassociated session asks "End my shift."
            ans = assistant.ask("End my shift.", session_id="unauthenticated_terminal")
            assert "please specify which employee" in ans.lower()

            # Ensure Around's shift was NOT ended
            rec = shift_store.get_active_attendance(operator_id="OP_AROUND", db_path=test_env["db_path"])
            assert rec is not None
            assert rec["status"] == "ACTIVE"

            assert mock_semantic.call_count == 0
            assert mock_ask_llm.call_count == 0

    def test_end_my_shift_with_session_context(self, test_env):
        """'End my shift' with authenticated operator in session correctly ends THAT employee."""
        assistant = Assistant(use_llm=False)

        shift_store.create_attendance_record("OP_AROUND", "Around", shift_store.local_today(), db_path=test_env["db_path"])
        shift_store.create_attendance_record("OP_HAMADAN", "Hamadan", shift_store.local_today(), db_path=test_env["db_path"])

        with patch("agent.nlu._call_semantic_model") as mock_semantic, \
             patch("agent.assistant.Assistant._ask_llm") as mock_ask_llm:

            # Operator Hamadan ends their shift passing operator_id in context
            ans = assistant.ask("End my shift.", session_id="sess_hamadan", operator_id="OP_HAMADAN")
            # If confirmation prompt is returned for self-voice, confirm it
            if "shall i" in ans.lower() or "confirm" in ans.lower():
                ans = assistant.ask("Yes", session_id="sess_hamadan")

            assert "hamadan" in ans.lower() and "shift has ended" in ans.lower()

            # Hamadan is ended, but Around is still active
            assert shift_store.get_active_attendance(operator_id="OP_HAMADAN", db_path=test_env["db_path"]) is None
            assert shift_store.get_active_attendance(operator_id="OP_AROUND", db_path=test_env["db_path"]) is not None

            assert mock_semantic.call_count == 0
            assert mock_ask_llm.call_count == 0

    def test_is_on_shift_queries_offline(self, test_env):
        """Verifies 'Is Around currently on shift?' and Hinglish queries check DB offline."""
        assistant = Assistant(use_llm=False)

        # Around is NOT clocked in yet
        with patch("agent.nlu._call_semantic_model") as mock_semantic, \
             patch("agent.assistant.Assistant._ask_llm") as mock_ask_llm:

            ans_no = assistant.ask("Is Around currently on shift?", session_id="s1")
            assert "no" in ans_no.lower()
            assert "around is not currently on shift" in ans_no.lower()

            # Now start Around's shift
            shift_store.create_attendance_record("OP_AROUND", "Around", shift_store.local_today(), db_path=test_env["db_path"])

            ans_yes = assistant.ask("Is Around currently on shift?", session_id="s1")
            assert "yes" in ans_yes.lower()
            assert "around is currently on shift" in ans_yes.lower()

            # Hinglish: "Kya Hamadan abhi shift par hai?" (Hamadan not on shift)
            ans_h_no = assistant.ask("Kya Hamadan abhi shift par hai?", session_id="s2")
            assert "no" in ans_h_no.lower()
            assert "hamadan is not currently on shift" in ans_h_no.lower()

            # General query: "Who is currently on shift?"
            ans_who = assistant.ask("Who is currently on shift?", session_id="s1")
            assert "currently clocked in: around" in ans_who.lower()

            assert mock_semantic.call_count == 0
            assert mock_ask_llm.call_count == 0

    def test_attendance_hours_and_confirmation_offline(self, test_env):
        assistant = Assistant(use_llm=False)
        shift_store.create_attendance_record("OP_AROUND", "Around", shift_store.local_today(), db_path=test_env["db_path"])

        with patch("agent.nlu._call_semantic_model") as mock_semantic, \
             patch("agent.assistant.Assistant._ask_llm") as mock_ask_llm:

            ans_hrs = assistant.ask("How many hours has Around worked?", session_id="s1")
            assert "around" in ans_hrs.lower() and ("active shift" in ans_hrs.lower() or "worked" in ans_hrs.lower())

            # Self shift start requires confirmation prompt
            prompt = assistant.ask("I want to start my shift", session_id="conf_sess", operator_id="OP_HAMADAN")
            assert "shall i" in prompt.lower() or "confirm" in prompt.lower() or "start" in prompt.lower()

            # Confirm with Yes
            confirm_ans = assistant.ask("Yes", session_id="conf_sess")
            assert "has started" in confirm_ans.lower()
            assert shift_store.get_active_attendance(operator_id="OP_HAMADAN", db_path=test_env["db_path"]) is not None

            assert mock_semantic.call_count == 0
            assert mock_ask_llm.call_count == 0

    def test_unknown_employee_rejected(self, test_env):
        assistant = Assistant(use_llm=False)

        with patch("agent.nlu._call_semantic_model") as mock_semantic, \
             patch("agent.assistant.Assistant._ask_llm") as mock_ask_llm:

            ans = assistant.ask("Start shift for GhostWorker.", session_id="s1")
            assert "could not find an employee" in ans.lower() or "not find an operator" in ans.lower()

            ans_end = assistant.ask("End GhostWorker's shift.", session_id="s1")
            assert "could not find an employee" in ans_end.lower()

            assert mock_semantic.call_count == 0
            assert mock_ask_llm.call_count == 0


# ===========================================================================
# PART B: Persistent Incident Storage & Pre-Gemini Historical Retrieval
# ===========================================================================

class TestPersistentIncidentStorage:
    """Verifies image persistence, SHA-256 deduplication, pre-Gemini search, and queries."""

    def test_save_image_persistence_and_deduplication(self, test_env):
        raw_bytes = _make_dummy_image_bytes(100, 80, 150)

        # 1. First upload
        rec1 = incident_store.save_incident_image(
            data=raw_bytes,
            original_name="nozzle_bead_defect.jpg",
            content_type="image/jpeg",
            db_path=test_env["db_path"],
            storage_dir=test_env["img_dir"],
        )
        assert rec1["image_id"].startswith("IMG-")
        assert os.path.exists(rec1["file_path"])
        assert rec1["width"] == 100
        assert rec1["height"] == 80

        # 2. Identical upload (content-hash deduplication)
        rec2 = incident_store.save_incident_image(
            data=raw_bytes,
            original_name="second_upload_same_image.jpg",
            content_type="image/jpeg",
            db_path=test_env["db_path"],
            storage_dir=test_env["img_dir"],
        )
        # Must reuse the same image_id and stored file path
        assert rec2["image_id"] == rec1["image_id"]
        assert rec2["file_path"] == rec1["file_path"]
        assert rec2["content_hash"] == rec1["content_hash"]

        # Only one file on disk
        files = os.listdir(test_env["img_dir"])
        assert len(files) == 1

    def test_reject_corrupt_and_non_image_files(self, test_env):
        # Empty bytes
        with pytest.raises(ValueError, match="empty"):
            incident_store.save_incident_image(b"", "empty.jpg", "image/jpeg", db_path=test_env["db_path"])

        # Corrupt / invalid magic bytes
        with pytest.raises(ValueError, match="Invalid image"):
            incident_store.save_incident_image(b"NOT_AN_IMAGE_DATA_12345", "test.jpg", "image/jpeg", db_path=test_env["db_path"])

    def test_image_retrieval_and_endpoint(self, test_env):
        raw_bytes = _make_dummy_image_bytes(100, 80, 180)
        saved = incident_store.save_incident_image(
            data=raw_bytes,
            original_name="test_defect.jpg",
            content_type="image/jpeg",
            db_path=test_env["db_path"],
            storage_dir=test_env["img_dir"],
        )
        image_id = saved["image_id"]

        # Direct function retrieval
        retrieved = incident_store.get_image_data(image_id, db_path=test_env["db_path"])
        assert retrieved is not None
        data, mime, fname = retrieved
        assert data == raw_bytes
        assert "image/jpeg" in mime
        assert fname == "test_defect.jpg"

        # HTTP Endpoint retrieval via FastAPI TestClient
        resp = client.get(f"/api/incidents/images/{image_id}")
        assert resp.status_code == 200
        assert resp.content == raw_bytes
        assert "image/jpeg" in resp.headers["content-type"]

        # Non-existent image ID returns 404
        resp404 = client.get("/api/incidents/images/IMG-NONEXISTENT")
        assert resp404.status_code == 404

    def test_incident_record_creation_and_search(self, test_env):
        raw_bytes = _make_dummy_image_bytes(100, 80, 90)
        saved = incident_store.save_incident_image(
            data=raw_bytes,
            original_name="layer_void.jpg",
            content_type="image/jpeg",
            db_path=test_env["db_path"],
            storage_dir=test_env["img_dir"],
        )

        inc = incident_store.create_incident(
            user_question="Visible concrete gap in layer 14",
            primary_image_id=saved["image_id"],
            primary_image_hash=saved["content_hash"],
            symptom_description="Layer void and discontinuous bead deposition",
            visual_observations="Clear 40mm horizontal gap between successive beads",
            defect_category="layer_discontinuity",
            possible_causes="Mortar delivery pause or aggregate obstruction in nozzle",
            confirmed_root_cause="Aggregate blockage in nozzle inlet reducer",
            recommended_action="Pause print, depressurize hose, clear nozzle reducer, verify flow rate",
            resolution_status="verified",
            engineering_review_status="approved",
            db_path=test_env["db_path"],
        )
        assert inc["incident_id"].startswith("INC-")
        assert inc["resolution_status"] == "verified"
        assert inc["primary_image_id"] == saved["image_id"]

        # Search historical incidents
        matches = incident_store.search_incidents("gap in layer void", db_path=test_env["db_path"])
        assert len(matches) >= 1
        assert matches[0]["incident_id"] == inc["incident_id"]
        assert matches[0]["match_score"] >= 0.50

    def test_search_historical_incidents_before_gemini(self, test_env):
        """When matching historical incident exists, return stored solution and cite incident without Gemini."""
        raw_bytes = _make_dummy_image_bytes(100, 80, 110)
        saved = incident_store.save_incident_image(
            data=raw_bytes,
            original_name="slump_collapse.jpg",
            content_type="image/jpeg",
            db_path=test_env["db_path"],
            storage_dir=test_env["img_dir"],
        )

        # Pre-seed a verified historical incident in the database
        hist_inc = incident_store.create_incident(
            user_question="Wall slumping and green strength collapse in corner",
            primary_image_id=saved["image_id"],
            primary_image_hash=saved["content_hash"],
            symptom_description="Wall slumping and bulging in corner",
            visual_observations="Excessive material spreading and layer collapse at 90-degree corner",
            defect_category="slump_collapse",
            confirmed_root_cause="Overly wet mix with high water-cement ratio and insufficient thixotropy",
            recommended_action="Reduce water ratio by 4%, increase resting pause at corners to 12s",
            resolution_status="verified",
            engineering_review_status="approved",
            db_path=test_env["db_path"],
        )

        # Operator asks about slumping defect with an uploaded image
        att = Attachment(
            filename="current_slump_defect.jpg",
            content_type="image/jpeg",
            size_bytes=len(raw_bytes),
            data=raw_bytes,
            kind="image",
        )

        with patch("agent.troubleshooter._call_gemini_fallback") as mock_gemini_fallback, \
             patch("agent.troubleshooter.extract_visual_observations", side_effect=agent.troubleshooter.extract_local_cv_observations):

            res = troubleshoot(
                question="Wall slumping and green strength collapse in corner",
                attachments=[att],
            )

            # Grounded in the historical incident
            assert res.rag_found is True
            assert not res.gemini_called
            assert hist_inc["incident_id"] in res.citations
            assert hist_inc["incident_id"] in res.answer
            assert "overly wet mix" in res.answer.lower()
            # Historical image accessible in response
            assert f"/api/incidents/images/{saved['image_id']}" in res.answer

            assert mock_gemini_fallback.call_count == 0

    def test_natural_language_historical_incident_queries(self, test_env):
        """Tests natural-language queries about past incidents answered directly from SQLite."""
        raw_bytes = _make_dummy_image_bytes(100, 80, 130)
        saved = incident_store.save_incident_image(
            data=raw_bytes,
            original_name="crack_sample.jpg",
            content_type="image/jpeg",
            db_path=test_env["db_path"],
            storage_dir=test_env["img_dir"],
        )

        inc = incident_store.create_incident(
            user_question="Concrete layer vertical tearing and crack",
            primary_image_id=saved["image_id"],
            primary_image_hash=saved["content_hash"],
            symptom_description="Vertical separation crack through concrete layers",
            defect_category="vertical_cracking",
            confirmed_root_cause="Differential shrinkage from rapid ambient draft",
            recommended_action="Erect draft shield and apply curing misting every 20 minutes",
            resolution_status="verified",
            engineering_review_status="approved",
            db_path=test_env["db_path"],
        )
        incident_store.verify_incident(
            incident_id=inc["incident_id"],
            confirmed_root_cause="Differential shrinkage from rapid ambient draft",
            recommended_action="Erect draft shield and apply curing misting every 20 minutes",
            verified_by="Staff Engineer Anand",
            db_path=test_env["db_path"],
        )

        with patch("agent.troubleshooter._call_gemini_fallback") as mock_fallback:
            mock_fallback.side_effect = AssertionError("Gemini must NOT be called for historical queries!")

            # 1. "Have we faced this concrete-layer defect before?"
            res1 = troubleshoot("Have we faced this concrete-layer defect before?")
            assert res1.gemini_called is False
            assert inc["incident_id"] in res1.answer
            assert "verified" in res1.answer.lower()

            # 2. "Show the image from the last similar incident"
            res2 = troubleshoot("Show the image from the last similar incident")
            assert res2.gemini_called is False
            assert f"/api/incidents/images/{saved['image_id']}" in res2.answer

            # 3. "What was the verified fix for incident INC-..."
            res3 = troubleshoot(f"What was the verified fix for incident {inc['incident_id']}?")
            assert res3.gemini_called is False
            assert "draft shield" in res3.answer.lower()
            assert "anand" in res3.answer.lower()

            # 4. "Was the previous fix verified by an engineer?"
            res4 = troubleshoot("Was the previous fix verified by an engineer?")
            assert res4.gemini_called is False
            assert "yes" in res4.answer.lower()
            assert "anand" in res4.answer.lower()

            assert mock_fallback.call_count == 0

    def test_gemini_fallback_saves_unverified_incident(self, test_env):
        """When neither stored incidents nor RAG match, Gemini is called and saved as unverified."""
        raw_bytes = _make_dummy_image_bytes(100, 80, 210)
        att = Attachment(
            filename="novel_exotic_defect.jpg",
            content_type="image/jpeg",
            size_bytes=len(raw_bytes),
            data=raw_bytes,
            kind="image",
        )

        fake_gemini_reply = (
            "### 1. General AI Observations\n"
            "Unusual helical curling along the top bead.\n\n"
            "### 2. Hypothesized Explanation\n"
            "Possible nozzle shear angle anomaly.\n\n"
            "### 3. Non-Invasive Checks\n"
            "Check nozzle mounting screws."
        )

        with patch("agent.troubleshooter._call_gemini_fallback") as mock_fallback, \
             patch("agent.troubleshooter.extract_visual_observations", side_effect=agent.troubleshooter.extract_local_cv_observations), \
             patch("agent.troubleshooter.search_documentation", return_value={"found": False, "results": []}):

            mock_fallback.return_value = (fake_gemini_reply, None)

            res = troubleshoot(
                question="We have novel exotic curling never seen before",
                attachments=[att],
            )

            assert res.gemini_called is True
            assert "general ai observations" in res.answer.lower()

            # Check that an incident was registered in SQLite with status unverified
            incidents = incident_store.list_incidents(db_path=test_env["db_path"])
            assert len(incidents) >= 1
            latest = incidents[0]
            assert latest["resolution_status"] == "unverified"
            assert latest["source_attribution"] == "gemini_vision"
            assert latest["primary_image_id"] is not None

    def test_gemini_failure_preserves_image_as_pending_analysis(self, test_env):
        """When Gemini is unavailable, image and incident are saved as pending_analysis."""
        raw_bytes = _make_dummy_image_bytes(100, 80, 220)
        att = Attachment(
            filename="camera_fault_print.jpg",
            content_type="image/jpeg",
            size_bytes=len(raw_bytes),
            data=raw_bytes,
            kind="image",
        )

        with patch("agent.troubleshooter._call_gemini_fallback") as mock_fallback, \
             patch("agent.troubleshooter.extract_visual_observations", side_effect=agent.troubleshooter.extract_local_cv_observations), \
             patch("agent.troubleshooter.search_documentation", return_value={"found": False, "results": []}):

            mock_fallback.return_value = (None, "GEMINI_API_KEY is not configured or network down")

            res = troubleshoot(
                question="Unknown printer glitch in cold weather",
                attachments=[att],
            )

            assert res.gemini_called is False
            assert "unavailable" in res.answer.lower() or "service" in res.answer.lower()

            # Image MUST be preserved under INCIDENT_IMAGES_DIR
            images = os.listdir(test_env["img_dir"])
            assert len(images) >= 1

            # Incident record MUST be marked as pending_analysis
            incidents = incident_store.list_incidents(db_path=test_env["db_path"])
            assert len(incidents) >= 1
            latest = incidents[0]
            assert latest["resolution_status"] == "pending_analysis"
            assert latest["primary_image_id"] is not None

    def test_engineer_verification_api(self, test_env):
        """Verifies engineer verification endpoint POST /api/incidents/{incident_id}/verify."""
        inc = incident_store.create_incident(
            user_question="Nozzle stutter during outer perimeter",
            symptom_description="Bead discontinuity and uneven layer",
            resolution_status="unverified",
            db_path=test_env["db_path"],
        )
        inc_id = inc["incident_id"]

        resp = client.post(
            f"/api/incidents/{inc_id}/verify",
            json={
                "confirmed_root_cause": "Loose mechanical belt coupling on extruder feeder axis",
                "recommended_action": "Tighten M4 set screws on feeder pulley to 3.2 Nm",
                "verified_by": "Senior Engineer Priya",
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["resolution_status"] == "verified"
        assert data["engineering_review_status"] == "approved"
        assert data["verified_by"] == "Senior Engineer Priya"
        assert "3.2 Nm" in data["recommended_action"]
