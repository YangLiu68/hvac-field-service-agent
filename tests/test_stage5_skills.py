from app.agent import build_default_registry
from app.agent.skills import SkillRouter, build_default_skill_registry
from fastapi.testclient import TestClient
from app.main import app, get_db
from app.database import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def test_all_skills_validate_against_registered_tools():
    tool_registry = build_default_registry()
    skills = build_default_skill_registry(tool_registry)
    assert {skill.name for skill in skills.list()} == {
        "general_hvac_triage", "no_cooling", "low_airflow", "refrigerant_issue",
        "electrical_fault", "thermostat_issue",
    }
    for skill in skills.list():
        assert set(skill.allowed_tools) <= set(tool_registry.names())
        assert skill.completion_criteria


def test_skill_router_selects_specialized_workflows_and_fallback():
    router = SkillRouter(build_default_skill_registry(build_default_registry()))
    assert router.route(notes="AC is not cooling and blowing warm air").selected_skill == "no_cooling"
    assert router.route(notes="The breaker keeps tripping").selected_skill == "electrical_fault"
    assert router.route(notes="Suction pressure is low", error_code="E7").selected_skill == "refrigerant_issue"
    fallback = router.route(notes="There is an unusual problem")
    assert fallback.selected_skill == "general_hvac_triage"
    assert fallback.matched_signals == ["fallback:no_matching_signal"]


def test_skill_max_steps_and_allowlist_are_exposed():
    skill = build_default_skill_registry(build_default_registry()).get("low_airflow")
    assert skill.max_steps == 18
    assert "search_service_manuals" in skill.allowed_tools
    assert "complete_diagnosis" in skill.allowed_tools


def test_skill_api_routes_a_job_without_invoking_llm(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'skills.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    def override_get_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    try:
        job = client.post("/jobs", json={
            "customer_name": "Skill Test",
            "equipment_model": "MODEL-1",
            "technician_notes": "AC is not cooling and blowing warm air",
        })
        assert job.status_code == 201, job.text
        route = client.post(f"/jobs/{job.json()['id']}/skill-route")
        assert route.status_code == 200, route.text
        assert route.json()["selected_skill"] == "no_cooling"
        assert client.get("/agent/skills/no_cooling").status_code == 200
    finally:
        app.dependency_overrides.clear()
        engine.dispose()
