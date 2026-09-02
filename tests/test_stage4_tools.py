from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.main import app, get_db


def test_stage4_tool_registry_and_audited_execution(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'stage4.db'}", connect_args={"check_same_thread": False})
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
    try:
        equipment = client.post("/equipment", json={
            "manufacturer": "Carrier", "model_number": "FK4B-001",
            "equipment_type": "fan_coil", "serial_number": "S4",
        })
        assert equipment.status_code == 201, equipment.text
        job = client.post("/jobs", json={
            "customer_name": "Tool Test", "equipment_id": equipment.json()["id"],
            "technician_notes": "Low airflow during cooling", "error_code": "E101",
        })
        assert job.status_code == 201, job.text
        job_id = job.json()["id"]

        tools = client.get("/agent/tools")
        assert tools.status_code == 200
        names = {item["name"] for item in tools.json()["tools"]}
        assert {"get_job_context", "search_service_manuals", "request_technician_measurement", "complete_diagnosis"} <= names

        run = client.post(f"/jobs/{job_id}/agent-runs", json={"max_steps": 10})
        assert run.status_code == 201, run.text
        run_id = run.json()["id"]

        context = client.post(f"/agent-runs/{run_id}/tools", json={"tool_name": "get_job_context", "arguments": {}})
        assert context.status_code == 200, context.text
        assert context.json()["result"]["job_id"] == job_id

        requested = client.post(f"/agent-runs/{run_id}/tools", json={
            "tool_name": "request_technician_measurement",
            "arguments": {
                "measurement_type": "supply_return_temperature_delta",
                "instructions": "Measure supply and return temperatures after five minutes of cooling.",
                "unit": "F",
            },
        })
        assert requested.status_code == 200, requested.text
        request_id = requested.json()["result"]["request_id"]
        assert client.get(f"/jobs/{job_id}").json()["status"] == "waiting_for_technician"

        recorded = client.post(f"/agent-runs/{run_id}/tools", json={
            "tool_name": "record_measurement",
            "arguments": {"request_id": request_id, "value": "18", "unit": "F", "recorded_by": "Alex"},
        })
        assert recorded.status_code == 200, recorded.text
        assert client.get(f"/jobs/{job_id}").json()["status"] == "diagnosing"

        calls = client.get(f"/agent-runs/{run_id}/tool-calls")
        assert calls.status_code == 200
        assert [call["status"] for call in calls.json()] == ["completed", "completed", "completed"]
        assert all(call["agent_run_id"] == run_id for call in calls.json())
    finally:
        app.dependency_overrides.clear()
        engine.dispose()
