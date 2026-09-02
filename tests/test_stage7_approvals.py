import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent import ApprovalRequired, ToolExecutor, build_default_registry
from app.database import Base
from app.main import app, get_db
from app.models import AgentRun, DiagnosticRun, Equipment, Job, ApprovalRequest


def _setup(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'approval.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db = Session()
    equipment = Equipment(manufacturer="Carrier", model_number="FK4B-001", equipment_type="split_system", refrigerant="R-410A")
    db.add(equipment)
    db.flush()
    job = Job(customer_name="Approval Test", equipment_id=equipment.id, equipment_model=equipment.model_number, technician_notes="Low suction pressure", status="diagnosing")
    db.add(job)
    db.flush()
    diagnostic = DiagnosticRun(job_id=job.id, status="running", model_name="fake")
    db.add(diagnostic)
    db.flush()
    run = AgentRun(job_id=job.id, diagnostic_run_id=diagnostic.id, skill_name="refrigerant_issue", max_steps=10, status="running")
    db.add(run)
    db.commit()
    return engine, db, job.id, run.id


def test_high_risk_measurement_creates_approval_and_pauses(tmp_path):
    engine, db, _job_id, run_id = _setup(tmp_path)
    try:
        with pytest.raises(ApprovalRequired) as error:
            ToolExecutor(build_default_registry()).execute(
                db=db, agent_run_id=run_id, tool_name="request_technician_measurement",
                arguments={"measurement_type": "suction_pressure", "instructions": "Measure suction pressure", "unit": "psi"},
            )
        request = db.query(ApprovalRequest).one()
        assert error.value.approval_id == request.id
        assert request.status == "pending"
        assert request.agent_run.status == "waiting_for_approval"
        assert request.risk_level == "high"
    finally:
        db.close()
        engine.dispose()


def test_approval_endpoint_executes_tool_after_approval(tmp_path):
    engine, db, job_id, run_id = _setup(tmp_path)
    try:
        with pytest.raises(ApprovalRequired):
            ToolExecutor(build_default_registry()).execute(
                db=db, agent_run_id=run_id, tool_name="request_technician_measurement",
                arguments={"measurement_type": "suction_pressure", "instructions": "Measure suction pressure", "unit": "psi"},
            )
        approval_id = db.query(ApprovalRequest).one().id
        db.close()

        def override_get_db():
            session = sessionmaker(bind=engine, autocommit=False, autoflush=False)()
            try:
                yield session
            finally:
                session.close()

        app.dependency_overrides[get_db] = override_get_db
        response = TestClient(app).post(f"/approvals/{approval_id}/approve", json={"decided_by": "Supervisor", "reason": "Qualified technician assigned"})
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "approved"
        check = sessionmaker(bind=engine)()
        try:
            request = check.get(ApprovalRequest, approval_id)
            assert request.status == "approved"
            assert request.agent_run.status == "waiting_for_technician"
            assert request.decisions[0].decision == "approved"
        finally:
            check.close()
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def test_critical_action_cannot_be_approved(tmp_path):
    engine, db, _job_id, run_id = _setup(tmp_path)
    try:
        request = ApprovalRequest(job_id=1, agent_run_id=run_id, tool_name="bypass_safety_switch", arguments="{}", risk_level="critical", reason="Safety bypass is prohibited")
        db.add(request)
        db.commit()
        approval_id = request.id
        db.close()

        def override_get_db():
            session = sessionmaker(bind=engine, autocommit=False, autoflush=False)()
            try:
                yield session
            finally:
                session.close()

        app.dependency_overrides[get_db] = override_get_db
        response = TestClient(app).post(f"/approvals/{approval_id}/approve", json={"decided_by": "Supervisor"})
        assert response.status_code == 403
    finally:
        app.dependency_overrides.clear()
        engine.dispose()
