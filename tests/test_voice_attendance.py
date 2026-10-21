import sys
from unittest.mock import MagicMock

# Mock heavy deps

import pytest
from agent.assistant import Assistant
from agent import db, shift_store

@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_voice.db"
    path = str(db_file)
    db.init_db(path)
    
    import agent.config as config
    original_db = config.DATABASE_PATH
    config.DATABASE_PATH = path
    
    # Pre-populate an operator
    shift_store.create_operator("OP_JOHN", "John Doe", role="operator", active=1, db_path=path)
    
    yield path
    
    config.DATABASE_PATH = original_db
    if db_file.exists():
        db_file.unlink()

def test_voice_shift_start_and_end(temp_db):
    assistant = Assistant()
    
    # Start Shift for John
    resp1 = assistant.ask("this is John Doe. start my shift", session_id="test1")
    assert "Shall I start the shift for John Doe?" in resp1
    
    # Confirm
    resp2 = assistant.ask("yes", session_id="test1")
    assert "John Doe, your shift has started." in resp2
    
    # Query status
    resp3 = assistant.ask("who is currently working", session_id="test1")
    assert "Currently clocked in: John Doe." in resp3
    
    # End Shift
    resp4 = assistant.ask("this is John Doe. end my shift", session_id="test1")
    assert "Shall I end the shift for John Doe?" in resp4
    
    # Confirm
    resp5 = assistant.ask("yes", session_id="test1")
    assert "John Doe, your shift has ended." in resp5
    
    # Query status again
    resp6 = assistant.ask("who is currently working", session_id="test1")
    assert "No employees are currently clocked in." in resp6

def test_voice_new_employee(temp_db):
    assistant = Assistant()
    
    # Start Shift for unknown user
    resp1 = assistant.ask("this is Alice. start my shift", session_id="test2")
    assert "New employee detected" in resp1
    assert "Would you like to register as a new employee and start your shift?" in resp1
    
    # Confirm
    resp2 = assistant.ask("yes", session_id="test2")
    assert "Alice, your shift has started." in resp2
    
    # Verify in DB
    records = shift_store.list_attendance_records(db_path=temp_db)
    assert any(r["employee_name"] == "Alice" for r in records)

import time
def test_voice_overtime_queries(temp_db):
    assistant = Assistant()
    
    # Pre-populate 10 hours shift for Aman
    now = time.time()
    shift_start = now - (10 * 3600)  # 10 hours ago
    
    # Needs operators table for voice routing
    shift_store.create_operator("OP_AMAN", "Aman", role="operator", active=1, db_path=temp_db)
    
    with db.connect(temp_db) as conn:
        conn.execute(
            """
            INSERT INTO attendance_records
            (operator_id, employee_name, attendance_date, shift_start, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("OP_AMAN", "Aman", shift_store.local_today(), shift_start, "ACTIVE", now, now)
        )
    
    # Query overtime
    resp = assistant.ask("how much overtime has Aman worked?", session_id="test_ot")
    # Expected phrasing based on query_router: "Aman has 1 hour and 0 minutes of overtime." (or similar natural phrasing)
    assert "Aman" in resp
    assert "1 hour" in resp
