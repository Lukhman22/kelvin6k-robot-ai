import pytest
import time
from agent import db, shift_store, query_router

@pytest.fixture
def test_db(tmp_path):
    db_path = str(tmp_path / "test_overtime.sqlite")
    db.init_db(db_path)
    # create operator
    shift_store.create_operator("OP123", "Test Operator", db_path=db_path)
    return db_path

def test_midnight_crossing_shift(test_db):
    # Shift crosses midnight
    start_time = time.mktime(time.strptime("2026-10-10 22:00:00", "%Y-%m-%d %H:%M:%S"))
    end_time = time.mktime(time.strptime("2026-10-11 02:00:00", "%Y-%m-%d %H:%M:%S"))
    
    shift_id = shift_store.start_shift("OP123", "2026-10-10", "Night", start_time=start_time, db_path=test_db)
    shift_store.end_shift(shift_id, end_time=end_time, db_path=test_db)
    
    shifts = shift_store.list_shifts(db_path=test_db)
    assert len(shifts) == 1
    shift = shifts[0]
    
    # 4 hours difference
    duration = shift["end_time"] - shift["start_time"]
    assert duration == 4 * 3600

def test_9_hour_printing_overtime(test_db):
    # Print job lasting 10 hours
    start_time = time.mktime(time.strptime("2026-10-10 08:00:00", "%Y-%m-%d %H:%M:%S"))
    end_time = time.mktime(time.strptime("2026-10-10 18:00:00", "%Y-%m-%d %H:%M:%S"))
    
    job_id = db.create_print_job("Wall 1", 1000, db_path=test_db)
    db.update_job_details(job_id, operator="OP123", db_path=test_db)
    
    # Update actual timestamps to fake a 10 hr job
    with db.connect(test_db) as conn:
        conn.execute("UPDATE print_jobs SET started_at = ?, completed_at = ?, status = 'completed' WHERE id = ?", (start_time, end_time, job_id))
        
    jobs = db.get_print_jobs_overtime(db_path=test_db)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["printing_duration_hours"] == 10.0
    assert job["regular_printing_hours"] == 9.0
    assert job["overtime_hours"] == 1.0
    assert job["overtime_status"] == "Overtime"
    
def test_missing_data_strictly_unavailable(test_db, monkeypatch):
    from agent import config
    monkeypatch.setattr(config, 'DATABASE_PATH', test_db)
    # Missing end time should use current time for calculation, but not hallucinate an end time string.
    start_time = time.time() - 3600 # started 1 hour ago
    
    shift_id = shift_store.start_shift("OP123", "2026-10-10", "Morning", start_time=start_time, db_path=test_db)
    
    shifts = shift_store.list_shifts(db_path=test_db)
    assert len(shifts) == 1
    
    # Test query router behavior manually to verify formatting handles missing end_time gracefully
    # If the user asks when shift ended, it should be active/ongoing
    context = {"operator_id": "OP123"}
    res = query_router.resolve(query_router.Intent.GET_SHIFT_BY_OPERATOR, "what time did my shift end", context)
    assert res.available == True
    assert res.facts["status"] == "active"
    
