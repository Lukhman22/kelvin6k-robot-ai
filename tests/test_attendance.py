import sys
from unittest.mock import MagicMock

# Mock out heavy and unrelated dependencies

import pytest
import os
import time
from fastapi.testclient import TestClient

from api.server import app
from agent import db, shift_store

client = TestClient(app)

@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_kelvin6k.db"
    path = str(db_file)
    db.init_db(path)
    
    import agent.config as config
    original_db = config.DATABASE_PATH
    config.DATABASE_PATH = path
    
    yield path
    
    config.DATABASE_PATH = original_db
    if db_file.exists():
        db_file.unlink()

def test_attendance_flow(temp_db):
    op_resp = client.post("/api/operators", json={"operator_id": "op123", "name": "John Doe"})
    assert op_resp.status_code == 200, op_resp.text

    check_in_resp = client.post("/api/v1/attendance/check-in", json={"employee_id": "op123"})
    assert check_in_resp.status_code == 200, check_in_resp.text
    data = check_in_resp.json()
    assert data["employee_name"] == "John Doe"
    assert data["status"] == "ACTIVE"
    assert "shift_start" in data
    assert data["shift_end"] is None

    fail_check_in = client.post("/api/v1/attendance/check-in", json={"employee_id": "op123"})
    assert fail_check_in.status_code == 400
    assert "already has an active shift" in fail_check_in.json()["detail"].lower()

    client.post("/api/operators", json={"operator_id": "op456", "name": "Jane Smith"})
    check_in_2 = client.post("/api/v1/attendance/check-in", json={"employee_id": "op456"})
    assert check_in_2.status_code == 200, check_in_2.text
    assert check_in_2.json()["employee_name"] == "Jane Smith"
    
    list_resp = client.get("/api/v1/attendance")
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert len(list_data) == 2
    
    check_out_resp = client.post("/api/v1/attendance/check-out", json={"employee_id": "op123"})
    assert check_out_resp.status_code == 200, check_out_resp.text
    out_data = check_out_resp.json()
    assert out_data["status"] == "COMPLETED"
    assert out_data["shift_end"] is not None
    assert out_data["shift_end"] >= out_data["shift_start"]
    
    fail_check_out = client.post("/api/v1/attendance/check-out", json={"employee_id": "op123"})
    assert fail_check_out.status_code == 400
    assert "No ACTIVE attendance record found" in fail_check_out.json()["detail"]
    
    list_resp_2 = client.get("/api/v1/attendance")
    assert list_resp_2.status_code == 200
    statuses = {r["employee_name"]: r["status"] for r in list_resp_2.json()}
    assert statuses["John Doe"] == "COMPLETED"
    assert statuses["Jane Smith"] == "ACTIVE"


def test_add_new_employee(temp_db):
    # 1. Empty employee name is rejected
    fail_create = client.post("/api/operators", json={"name": "", "role": "operator"})
    assert fail_create.status_code == 400
    assert "Employee name is required" in fail_create.json()["detail"]

    # 2. New operator can be created & 3. Receives unique operator_id
    create_resp = client.post("/api/operators", json={"name": "New Worker", "employee_code": "NW123", "role": "operator"})
    assert create_resp.status_code == 200, create_resp.text
    new_op = create_resp.json()
    assert "operator_id" in new_op
    assert new_op["operator_id"].startswith("OP_NEW_WORKER_")
    assert new_op["name"] == "New Worker"
    assert new_op["employee_code"] == "NW123"

    op_id = new_op["operator_id"]

    # 4. New operator appears in the operator list
    list_resp = client.get("/api/operators")
    assert list_resp.status_code == 200
    operators = list_resp.json()["operators"]
    assert any(o["operator_id"] == op_id for o in operators)

    # 5. Duplicate employee code is rejected
    fail_dup = client.post("/api/operators", json={"name": "Another Worker", "employee_code": "NW123", "role": "operator"})
    assert fail_dup.status_code == 409
    assert "Employee code already exists" in fail_dup.json()["detail"]

    # 7. Inactive employees are not returned for attendance selection
    client.post("/api/operators", json={"name": "Inactive Worker", "employee_code": "IW456", "role": "operator", "active": 0})
    list_active = client.get("/api/operators?active_only=true")
    active_ops = list_active.json()["operators"]
    assert not any(o["name"] == "Inactive Worker" for o in active_ops)
    
    list_all = client.get("/api/operators?active_only=false")
    all_ops = list_all.json()["operators"]
    assert any(o["name"] == "Inactive Worker" for o in all_ops)

    # 8. New employee can immediately start an attendance record
    check_in = client.post("/api/v1/attendance/check-in", json={"employee_id": op_id})
    assert check_in.status_code == 200, check_in.text
    
    # 9. Attendance correctly stores the new employee's operator_id
    att_list = client.get("/api/v1/attendance")
    records = att_list.json()
    assert any(r["operator_id"] == op_id and r["employee_name"] == "New Worker" for r in records)

def test_overtime_calculations():
    # Regular 8 hours
    row_8h = {
        "shift_start": 1000,
        "shift_end": 1000 + 8 * 3600,
        "status": "COMPLETED"
    }
    enriched_8h = shift_store.enrich_attendance_record(row_8h, now=1000 + 8*3600)
    assert enriched_8h["worked_seconds"] == 8 * 3600
    assert enriched_8h["overtime_seconds"] == 0
    assert enriched_8h["overtime_display"] == "0m"

    # Exactly 9 hours
    row_9h = {
        "shift_start": 1000,
        "shift_end": 1000 + 9 * 3600,
        "status": "COMPLETED"
    }
    enriched_9h = shift_store.enrich_attendance_record(row_9h, now=1000 + 9*3600)
    assert enriched_9h["worked_seconds"] == 9 * 3600
    assert enriched_9h["overtime_seconds"] == 0
    assert enriched_9h["overtime_display"] == "0m"

    # 9 hours 1 min
    row_9h1m = {
        "shift_start": 1000,
        "shift_end": 1000 + 9 * 3600 + 60,
        "status": "COMPLETED"
    }
    enriched_9h1m = shift_store.enrich_attendance_record(row_9h1m, now=1000 + 9*3600+60)
    assert enriched_9h1m["worked_seconds"] == 9 * 3600 + 60
    assert enriched_9h1m["overtime_seconds"] == 60
    assert enriched_9h1m["overtime_display"] == "1m"

    # 10 hours active shift
    row_10h_active = {
        "shift_start": 1000,
        "shift_end": None,
        "status": "ACTIVE"
    }
    # Time now is 10 hours later
    enriched_10h = shift_store.enrich_attendance_record(row_10h_active, now=1000 + 10*3600)
    assert enriched_10h["worked_seconds"] == 10 * 3600
    assert enriched_10h["overtime_seconds"] == 3600
    assert enriched_10h["overtime_display"] == "1h 00m"
    assert enriched_10h["overtime_spoken"] == "1 hour and 0 minutes"

def test_today_filter(temp_db):
    # Insert records directly
    now = time.time()
    shift_store.create_operator("OP_1", "Old Employee", role="operator", active=1, db_path=temp_db)
    shift_store.create_operator("OP_2", "Today Employee", role="operator", active=1, db_path=temp_db)
    with db.connect(temp_db) as conn:
        conn.execute(
            """
            INSERT INTO attendance_records
            (operator_id, employee_name, attendance_date, shift_start, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("OP_1", "Old Employee", "2023-01-01", 1000, "COMPLETED", now, now)
        )
        conn.execute(
            """
            INSERT INTO attendance_records
            (operator_id, employee_name, attendance_date, shift_start, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("OP_2", "Today Employee", shift_store.local_today(), 1000, "ACTIVE", now, now)
        )
    
    # List should only return Today Employee when date_filter is omitted
    records = shift_store.list_attendance_records(db_path=temp_db)
    assert len(records) == 1
    assert records[0]["employee_name"] == "Today Employee"

