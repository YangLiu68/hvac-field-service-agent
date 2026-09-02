from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.main import app, get_db


def test_technician_workspace_endpoints_create_brief_task_and_summary(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'workspace.db'}", connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(engine)

    def override_get_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    try:
        equipment = client.post("/equipment", json={
            "manufacturer": "Carrier", "model_number": "DEMO-1", "equipment_type": "split_system",
        }).json()
        job = client.post("/jobs", json={
            "customer_name": "Demo Site", "equipment_id": equipment["id"],
            "technician_notes": "AC is not cooling and blows warm air", "error_code": "E101",
        }).json()
        tasks = client.get("/technician/tasks")
        assert tasks.status_code == 200
        assert tasks.json()[0]["job_id"] == job["id"]

        brief = client.get(f"/jobs/{job['id']}/brief")
        assert brief.status_code == 200
        assert "Carrier DEMO-1" in brief.json()["summary"]

        summary = client.post(f"/jobs/{job['id']}/summary")
        assert summary.status_code == 201
        assert "AC is not cooling" in summary.json()["content"]
    finally:
        app.dependency_overrides.clear()
        engine.dispose()
