from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.main import app, get_db


def test_customer_request_is_matched_assigned_and_accepted(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'field_service.db'}", connect_args={"check_same_thread": False})
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
        customer = client.post("/users", json={"display_name": "Casey Customer", "email": "casey@example.com", "role": "customer"}).json()
        dispatcher = client.post("/users", json={"display_name": "Drew Dispatcher", "email": "drew@example.com", "role": "dispatcher"}).json()
        tech_user = client.post("/users", json={"display_name": "Taylor Tech", "email": "taylor@example.com", "role": "technician"}).json()
        profile = client.post("/technicians", json={
            "user_id": tech_user["id"], "skills": ["rooftop_unit"],
            "certifications": ["epa_608"], "service_postal_codes": ["94107"], "on_call": True,
        })
        assert profile.status_code == 201, profile.text
        equipment = client.post("/equipment", json={"manufacturer": "Carrier", "model_number": "RTU-1", "equipment_type": "rooftop_unit"}).json()
        request = client.post("/service-requests", json={
            "customer_id": customer["id"], "equipment_id": equipment["id"],
            "description": "Possible refrigerant leak with hissing sound", "service_address": "1 Market St",
            "postal_code": "94107", "urgency": "emergency",
        })
        assert request.status_code == 201, request.text
        request_id = request.json()["id"]
        matches = client.get(f"/service-requests/{request_id}/matches")
        assert matches.status_code == 200, matches.text
        assert matches.json()[0]["technician_name"] == "Taylor Tech"
        assert matches.json()[0]["score"] == 110
        assignment = client.post(f"/service-requests/{request_id}/assignments", json={
            "dispatcher_id": dispatcher["id"], "technician_profile_id": profile.json()["id"],
        })
        assert assignment.status_code == 201, assignment.text
        assert assignment.json()["status"] == "assigned"
        assert assignment.json()["job_id"] is not None
        accepted = client.post(f"/assignments/{assignment.json()['id']}/accept?technician_profile_id={profile.json()['id']}")
        assert accepted.status_code == 200, accepted.text
        assert accepted.json()["status"] == "accepted"
        assert client.get("/jobs").status_code == 200
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def test_local_admin_can_bootstrap_and_sign_in(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'auth.db'}", connect_args={"check_same_thread": False})
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
        created = client.post("/auth/bootstrap", json={
            "display_name": "Admin", "email": "admin@example.com", "password": "safe-pass-123", "role": "admin",
        })
        assert created.status_code == 201, created.text
        signed_in = client.post("/auth/login", json={"email": "admin@example.com", "password": "safe-pass-123"})
        assert signed_in.status_code == 200, signed_in.text
        token = signed_in.json()["access_token"]
        assert client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).json()["role"] == "admin"
    finally:
        app.dependency_overrides.clear()
        engine.dispose()
