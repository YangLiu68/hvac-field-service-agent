import json
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.main import app, get_db
from app.schemas import AnalysisResult, SourceItem


def test_complete_job_feedback_lifecycle(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'stage2.db'}",
        connect_args={"check_same_thread": False},
    )
    TestingSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(engine)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    fake_analysis = AnalysisResult(
        likely_cause="Restricted airflow",
        confidence=0.82,
        recommendation="Inspect the filter and evaporator coil.",
        sources=[
            SourceItem(
                document="document01.pdf",
                page=17,
                text="Airflow Too Low",
                score=0.49,
            )
        ],
    )

    try:
        equipment = client.post(
            "/equipment",
            json={
                "manufacturer": "Carrier",
                "model_number": "MODEL-123",
                "equipment_type": "air_handler",
                "serial_number": "SERIAL-001",
                "refrigerant": "R-410A",
            },
        )
        assert equipment.status_code == 201, equipment.text

        job = client.post(
            "/jobs",
            json={
                "customer_name": "Demo Customer",
                "equipment_id": equipment.json()["id"],
                "technician_notes": "AC is not cooling and airflow is weak",
                "error_code": "E101",
            },
        )
        assert job.status_code == 201, job.text
        job_id = job.json()["id"]

        with patch("app.main.analyze_job", return_value=fake_analysis):
            diagnosis = client.post(f"/jobs/{job_id}/diagnose")
        assert diagnosis.status_code == 200, diagnosis.text
        assert diagnosis.json()["status"] == "completed"
        assert json.loads(diagnosis.json()["sources"])[0]["page"] == 17
        run_id = diagnosis.json()["id"]

        technician_feedback = client.post(
            f"/jobs/{job_id}/technician-feedback",
            json={
                "diagnostic_run_id": run_id,
                "technician_name": "Alex Tech",
                "recommendation_accepted": True,
                "action_taken": "Replaced the clogged filter",
            },
        )
        assert technician_feedback.status_code == 201, technician_feedback.text

        customer_feedback = client.post(
            f"/jobs/{job_id}/customer-feedback",
            json={"rating": 5, "issue_resolved": True, "comments": "Cooling restored"},
        )
        assert customer_feedback.status_code == 201, customer_feedback.text

        closed = client.post(
            f"/jobs/{job_id}/close",
            json={
                "actual_cause": "Clogged return-air filter",
                "repair_action": "Replaced filter and verified airflow",
                "first_time_fix": True,
                "return_visit_required": False,
                "verified_by": "Alex Tech",
                "approved_for_retrieval": False,
            },
        )
        assert closed.status_code == 201, closed.text

        current_job = client.get(f"/jobs/{job_id}")
        assert current_job.json()["status"] == "closed"

        timeline = client.get(f"/jobs/{job_id}/timeline")
        assert timeline.status_code == 200
        assert [event["event_type"] for event in timeline.json()] == [
            "job_created",
            "diagnosis_started",
            "diagnosis_completed",
            "technician_feedback_added",
            "customer_feedback_added",
            "job_closed",
        ]
    finally:
        app.dependency_overrides.clear()
        engine.dispose()
