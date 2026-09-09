from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.main import app, get_db
from app.models import CaseMemory, Job, VerifiedOutcome
from app.services.unified_assistant import _clean_user_facing_content


def test_user_facing_output_filter_removes_reasoning_wrapper_and_markdown():
    content = "Analysis: internal plan\nSafest next step: **confirm outdoor unit operation**."
    assert _clean_user_facing_content(content) == "Safest next step: confirm outdoor unit operation."


def test_unified_assistant_creates_job_retrieves_memory_and_persists_turns(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'assistant.db'}", connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(engine)

    def override_get_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    def fake_answer(job, message, cases, entries, manuals, history, skill_route=None):
        return f"Recorded on work order #{job.id}. Found {len(cases)} similar verified case(s)."

    monkeypatch.setattr("app.services.unified_assistant._llm_answer", fake_answer)
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    try:
        db = Session()
        old_job = Job(customer_name="Prior", equipment_model="MODEL-NC", technician_notes="warm air E102", error_code="E102", status="closed")
        db.add(old_job)
        db.flush()
        outcome = VerifiedOutcome(job_id=old_job.id, actual_cause="control communication fault", repair_action="repaired communication wiring", first_time_fix=True, verified_by="Tech", approved_for_retrieval=True)
        db.add(outcome)
        db.flush()
        db.add(CaseMemory(outcome_id=outcome.id, job_id=old_job.id, equipment_model="MODEL-NC", equipment_type="split_system", error_code="E102", symptoms="warm air E102", actual_cause=outcome.actual_cause, repair_action=outcome.repair_action, first_time_fix=True, return_visit_required=False, searchable_text="warm air E102 control communication fault repaired wiring"))
        db.commit()
        db.close()

        response = client.post("/assistant/turns", json={"message": "Carrier MODEL-NC blows warm air with E102", "customer_name": "Demo", "manufacturer": "Carrier", "equipment_model": "MODEL-NC", "equipment_type": "split_system"})
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["created_work_order"] is True
        assert data["similar_cases"][0]["actual_cause"] == "control communication fault"
        assert "saved_conversation" in data["actions"]
        assert data["selected_skill"] == "no_cooling"

        followup = client.post("/assistant/turns", json={"conversation_id": data["conversation_id"], "message": "Supply temperature is 78 F"})
        assert followup.status_code == 200
        history = client.get(f"/assistant/conversations/{data['conversation_id']}").json()
        assert len(history["turns"]) == 4
        db = Session()
        job = db.get(Job, data["job_id"])
        assert "Supply temperature is 78 F" in job.technician_notes
        db.close()
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def test_knowledge_entries_are_persisted_and_returned(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'knowledge.db'}", connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(engine)
    def override_get_db():
        db = Session()
        try: yield db
        finally: db.close()
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    try:
        created = client.post("/knowledge", json={"title": "E102 triage", "category": "error_code", "keywords": "Carrier E102 communication", "content": "Verify model-specific documentation before diagnosis."})
        assert created.status_code == 201
        assert client.get("/knowledge?category=error_code").json()[0]["title"] == "E102 triage"
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def test_estimate_follow_up_review_and_dashboard(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'operations.db'}", connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(engine)
    def override_get_db():
        db = Session()
        try: yield db
        finally: db.close()
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    try:
        db = Session()
        job = Job(customer_name="Homeowner", equipment_model="HP-20", technician_notes="compressor failed", status="open")
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id
        db.close()
        estimate = client.post("/estimates", json={"job_id": job_id, "amount": 6500, "description": "Heat pump replacement"})
        assert estimate.status_code == 201
        follow_up = client.post("/follow-ups", json={"estimate_id": estimate.json()["id"]})
        assert follow_up.status_code == 201
        assert follow_up.json()["requires_review"] is True
        assert client.post(f"/follow-ups/{follow_up.json()['id']}/send").status_code == 409
        assert client.post(f"/follow-ups/{follow_up.json()['id']}/approve").status_code == 200
        assert client.post(f"/follow-ups/{follow_up.json()['id']}/send").json()["status"] == "sent"
        dashboard = client.get("/operations/dashboard").json()
        assert dashboard["open_pipeline_value"] == 6500
        assert dashboard["active_jobs"] == 1
    finally:
        app.dependency_overrides.clear()
        engine.dispose()
