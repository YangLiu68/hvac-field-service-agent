import json
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent import ToolExecutor, build_default_registry
from app.agent.orchestrator import AgentOrchestrator, forced_escalation_reason
from app.agent.skills import SkillRouter, build_default_skill_registry
from app.database import Base
from app.models import AgentRun, DiagnosticRun, Equipment, Job


class FakeResponses:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs.pop(0)


class FakeClient:
    def __init__(self, outputs):
        self.responses = FakeResponses(outputs)


def _setup(tmp_path, notes="AC is not cooling"):
    engine = create_engine(f"sqlite:///{tmp_path / 'orchestrator.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db = Session()
    equipment = Equipment(manufacturer="Carrier", model_number="FK4B-001", equipment_type="fan_coil")
    db.add(equipment)
    db.flush()
    job = Job(customer_name="Agent Test", equipment_id=equipment.id, equipment_model=equipment.model_number, technician_notes=notes, status="diagnosing")
    db.add(job)
    db.flush()
    diagnostic = DiagnosticRun(job_id=job.id, status="running", model_name="fake")
    db.add(diagnostic)
    db.flush()
    run = AgentRun(job_id=job.id, diagnostic_run_id=diagnostic.id, skill_name="no_cooling", max_steps=5, status="tool_testing")
    db.add(run)
    db.commit()
    return engine, db, run.id


def _orchestrator(client):
    registry = build_default_registry()
    skills = build_default_skill_registry(registry)
    return AgentOrchestrator(registry, skills, ToolExecutor(registry), client=client)


def test_orchestrator_plan_act_observe_completes(tmp_path, monkeypatch):
    engine, db, run_id = _setup(tmp_path)
    try:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        client = FakeClient([
            SimpleNamespace(id="resp_1", output=[SimpleNamespace(type="function_call", call_id="call_1", name="get_job_context", arguments="{}")], output_text=""),
            SimpleNamespace(id="resp_2", output=[SimpleNamespace(type="function_call", call_id="call_2", name="complete_diagnosis", arguments=json.dumps({"likely_cause": "Restricted airflow", "confidence": 0.8, "recommendation": "Inspect filter and coil."}))], output_text=""),
        ])
        result = _orchestrator(client).run(db, run_id)
        assert result["status"] == "completed"
        assert result["current_step"] == 2
        assert len(client.responses.calls) == 2
        assert client.responses.calls[1]["previous_response_id"] == "resp_1"
    finally:
        db.close()
        engine.dispose()


def test_openrouter_uses_stateless_tool_continuation(tmp_path, monkeypatch):
    engine, db, run_id = _setup(tmp_path)
    try:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        client = FakeClient([
            SimpleNamespace(id="resp_1", output=[SimpleNamespace(type="function_call", call_id="call_1", name="get_job_context", arguments="{}")], output_text=""),
            SimpleNamespace(id="resp_2", output=[SimpleNamespace(type="function_call", call_id="call_2", name="complete_diagnosis", arguments=json.dumps({"likely_cause": "Restricted airflow", "confidence": 0.8, "recommendation": "Inspect filter and coil."}))], output_text=""),
        ])
        result = _orchestrator(client).run(db, run_id)
        assert result["status"] == "completed"
        assert "previous_response_id" not in client.responses.calls[1]
        continuation = client.responses.calls[1]["input"]
        assert continuation[1]["role"] == "user"
        assert "Diagnostic tool result from get_job_context" in continuation[1]["content"]
    finally:
        db.close()
        engine.dispose()


def test_orchestrator_pauses_for_measurement_and_resumes(tmp_path):
    engine, db, run_id = _setup(tmp_path)
    try:
        client = FakeClient([
            SimpleNamespace(id="resp_1", output=[SimpleNamespace(type="function_call", call_id="call_1", name="request_technician_measurement", arguments=json.dumps({"measurement_type": "temperature_delta", "instructions": "Measure supply-return delta.", "unit": "F"}))], output_text=""),
            SimpleNamespace(id="resp_2", output=[SimpleNamespace(type="function_call", call_id="call_2", name="complete_diagnosis", arguments=json.dumps({"likely_cause": "Restricted airflow", "confidence": 0.8, "recommendation": "Inspect filter and coil."}))], output_text=""),
        ])
        orchestrator = _orchestrator(client)
        paused = orchestrator.run(db, run_id)
        assert paused["status"] == "waiting_for_technician"
        request_id = paused["pending_action"]["request_id"]

        # Simulate the technician response through the same audited Stage 4 executor.
        executor = ToolExecutor(build_default_registry())
        executor.execute(db=db, agent_run_id=run_id, tool_name="record_measurement", arguments={
            "request_id": request_id, "value": "18", "unit": "F", "recorded_by": "Alex",
        })
        resumed = orchestrator.run(db, run_id)
        assert resumed["status"] == "completed"
        assert len(client.responses.calls) == 2
    finally:
        db.close()
        engine.dispose()


def test_orchestrator_replays_identical_read_only_tool_call(tmp_path):
    engine, db, run_id = _setup(tmp_path)
    try:
        client = FakeClient([
            SimpleNamespace(id="resp_1", output=[SimpleNamespace(type="function_call", call_id="call_1", name="get_job_context", arguments="{}")], output_text=""),
            SimpleNamespace(id="resp_2", output=[SimpleNamespace(type="function_call", call_id="call_2", name="get_job_context", arguments="{}")], output_text=""),
            SimpleNamespace(id="resp_3", output=[SimpleNamespace(type="function_call", call_id="call_3", name="complete_diagnosis", arguments=json.dumps({"likely_cause": "Restricted airflow", "confidence": 0.8, "recommendation": "Inspect filter and coil."}))], output_text=""),
        ])
        result = _orchestrator(client).run(db, run_id)
        assert result["status"] == "completed"
        assert result["current_step"] == 2
    finally:
        db.close()
        engine.dispose()


def test_forced_safety_escalation_detects_intake_hazards():
    assert forced_escalation_reason("Possible refrigerant leak with hissing sound")
    assert forced_escalation_reason("Burning smell near electrical panel")
    assert forced_escalation_reason("Routine filter inspection") is None
