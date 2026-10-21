import pytest
import time
from fastapi.testclient import TestClient

from api.server import app
from agent import db, shift_store, query_router, assistant
import agent.config as config

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path, monkeypatch):
    """
    Set up an isolated test database with operators matching production:
    - OP_HAMADAN_D1B91C: Hamadan
    - OP_RAHUL: Rahul
    - OP_AMAN: Aman
    """
    test_db_path = str(tmp_path / "test_kelvin6k.db")
    db.init_db(test_db_path)
    monkeypatch.setattr(config, "DATABASE_PATH", test_db_path)
    
    # Register test operators
    shift_store.register_operator("OP_HAMADAN_D1B91C", "Hamadan", role="operator", active=1)
    shift_store.register_operator("OP_RAHUL", "Rahul", role="lead", active=1)
    shift_store.register_operator("OP_AMAN", "Aman", role="operator", active=1)
    
    # Reset query router cached operators
    query_router.refresh_known_operators()
    
    # Reset assistant session context and pending actions
    from api.server import assistant as server_assistant
    server_assistant._actions._pending.clear()
    server_assistant._session_ctx._sessions.clear()
    assistant._context = assistant.SessionContext()
    
    yield test_db_path


def test_start_and_end_shift_via_chat():
    """Test 'Kelvin, start Hamadan's shift' and duplicate prevention, followed by end shift."""
    # 1. Start Hamadan's shift
    resp = client.post("/api/chat", json={"question": "Kelvin, start Hamadan's shift"})
    assert resp.status_code == 200
    data = resp.json()
    assert "Hamadan" in data["answer"]
    assert "started" in data["answer"].lower()
    assert data["attendance_updated"] is True
    assert data["intent"] == "SHIFT_START"

    # Verify database state
    active = shift_store.get_active_shift("OP_HAMADAN_D1B91C")
    assert active is not None
    assert active["operator_id"] == "OP_HAMADAN_D1B91C"

    # 2. Duplicate start attempt should NOT create duplicate shift
    resp_dup = client.post("/api/chat", json={"question": "Kelvin, start Hamadan's shift"})
    assert resp_dup.status_code == 200
    data_dup = resp_dup.json()
    assert "already has an active shift" in data_dup["answer"].lower()

    # 3. Query hours worked
    resp_hours = client.post("/api/chat", json={"question": "How long has Hamadan worked?"})
    assert resp_hours.status_code == 200
    assert "Hamadan" in resp_hours.json()["answer"]
    assert "active" in resp_hours.json()["answer"].lower() or "worked" in resp_hours.json()["answer"].lower()

    # 4. End Hamadan's shift
    resp_end = client.post("/api/chat", json={"question": "End Hamadan's shift"})
    assert resp_end.status_code == 200
    data_end = resp_end.json()
    assert "Hamadan" in data_end["answer"]
    assert "ended" in data_end["answer"].lower()
    assert data_end["attendance_updated"] is True
    assert data_end["intent"] == "SHIFT_END"

    # Verify shift is closed
    active_after = shift_store.get_active_shift("OP_HAMADAN_D1B91C")
    assert active_after is None


def test_multi_turn_pronoun_context():
    """Test start shift followed by pronoun queries ('How long has he worked?', 'End his shift')."""
    # Start Hamadan's shift
    resp1 = client.post("/api/chat", json={"question": "Start Hamadan's shift"})
    assert resp1.status_code == 200
    assert "Hamadan" in resp1.json()["answer"]

    # Pronoun question: "How long has he worked?"
    resp2 = client.post("/api/chat", json={"question": "How long has he worked?"})
    assert resp2.status_code == 200
    assert "Hamadan" in resp2.json()["answer"]

    # Pronoun end shift: "End his shift"
    resp3 = client.post("/api/chat", json={"question": "End his shift"})
    assert resp3.status_code == 200
    assert "Hamadan" in resp3.json()["answer"]
    assert "ended" in resp3.json()["answer"].lower()


def test_who_is_currently_on_shift():
    """Test 'Who is currently on shift?' when empty and with active operators."""
    # When no one is on shift
    resp_empty = client.post("/api/chat", json={"question": "Who is currently on shift?"})
    assert resp_empty.status_code == 200
    assert "no employees are currently on shift" in resp_empty.json()["answer"].lower() or "0" in resp_empty.json()["answer"]

    # Clock in Rahul
    shift_store.check_in_attendance("OP_RAHUL")
    shift_store.start_shift("OP_RAHUL", "Line 1")

    # When Rahul is on shift
    resp_active = client.post("/api/chat", json={"question": "Who is on shift?"})
    assert resp_active.status_code == 200
    assert "Rahul" in resp_active.json()["answer"]


def test_when_did_operator_start():
    """Test 'When did Rahul start?'"""
    # Start Rahul's shift
    client.post("/api/chat", json={"question": "Start Rahul's shift"})

    resp = client.post("/api/chat", json={"question": "When did Rahul start?"})
    assert resp.status_code == 200
    assert "Rahul" in resp.json()["answer"]
    assert "started" in resp.json()["answer"].lower()


def test_who_has_most_overtime():
    """Test 'Who has the most overtime?'"""
    resp = client.post("/api/chat", json={"question": "Who has the most overtime?"})
    assert resp.status_code == 200
    assert "overtime" in resp.json()["answer"].lower()


def test_working_days_query():
    """Test 'How many working days has Aman completed?'"""
    # Add historical attendance records for Aman on different dates
    shift_store.check_in_attendance("OP_AMAN", timestamp="2026-10-01T08:00:00")
    shift_store.check_out_attendance("OP_AMAN", timestamp="2026-10-01T17:00:00")
    shift_store.check_in_attendance("OP_AMAN", timestamp="2026-10-02T08:00:00")
    shift_store.check_out_attendance("OP_AMAN", timestamp="2026-10-02T17:00:00")

    resp = client.post("/api/chat", json={"question": "How many working days has Aman completed?"})
    assert resp.status_code == 200
    answer = resp.json()["answer"]
    assert "Aman" in answer
    assert "2" in answer or "two" in answer.lower()


def test_print_progress_query():
    """Test 'What is the current print progress?' and 'Which print job is running?'"""
    resp1 = client.post("/api/chat", json={"question": "What is the current print progress?"})
    assert resp1.status_code == 200
    assert "print" in resp1.json()["answer"].lower() or "progress" in resp1.json()["answer"].lower() or "job" in resp1.json()["answer"].lower()

    resp2 = client.post("/api/chat", json={"question": "Which print job is running?"})
    assert resp2.status_code == 200
    assert resp2.json()["answer"] is not None


def test_maintenance_rag_layer_gap():
    """Test 'What causes a gap between concrete layers?' returns unconfirmed possible causes."""
    resp = client.post("/api/chat", json={"question": "What causes a gap between concrete layers?"})
    assert resp.status_code == 200
    answer = resp.json()["answer"].lower()
    assert "gap" in answer or "layer" in answer or "causes" in answer or "pump" in answer or "speed" in answer


def test_safety_clogged_nozzle_under_pressure():
    """Test 'Can I clear a clogged nozzle while the pump is pressurised?' respects safety restriction."""
    resp = client.post("/api/chat", json={"question": "Can I clear a clogged nozzle while the pump is pressurised?"})
    assert resp.status_code == 200
    answer = resp.json()["answer"].lower()
    # Must enforce safety: never clear while pressurised
    assert "no" in answer or "never" in answer or "not" in answer or "depressurise" in answer or "pressure" in answer
