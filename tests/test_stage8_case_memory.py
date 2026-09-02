import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import AgentRun, DiagnosticHypothesis, DiagnosticRun, Equipment, Job, TechnicianFeedback, VerifiedOutcome
from app.services.case_memory import evaluate_closed_outcome, evaluation_metrics, index_verified_outcome, search_case_memories, sanitize_text


def _setup(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'memory.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    equipment = Equipment(manufacturer="Carrier", model_number="FK4B-001", equipment_type="fan_coil")
    db.add(equipment)
    db.flush()
    job = Job(customer_name="Jane Doe", equipment_id=equipment.id, equipment_model="FK4B-001", technician_notes="AC not cooling; customer: Jane Doe", error_code="E101", status="in_service")
    db.add(job)
    db.flush()
    diagnostic = DiagnosticRun(job_id=job.id, status="completed", likely_cause="Clogged return air filter", confidence=0.8, recommendation="Replace the filter")
    db.add(diagnostic)
    db.flush()
    db.add_all([
        DiagnosticHypothesis(diagnostic_run_id=diagnostic.id, rank=1, cause="Clogged return air filter", confidence=0.8, selected=True),
        DiagnosticHypothesis(diagnostic_run_id=diagnostic.id, rank=2, cause="Low refrigerant", confidence=0.4),
        TechnicianFeedback(job_id=job.id, diagnostic_run_id=diagnostic.id, technician_name="Alex", recommendation_accepted=True),
    ])
    db.flush()
    outcome = VerifiedOutcome(job_id=job.id, actual_cause="Clogged return air filter", repair_action="Replaced filter", first_time_fix=True, return_visit_required=False, verified_by="Alex", approved_for_retrieval=True)
    db.add(outcome)
    db.flush()
    return engine, db, job, outcome


def test_only_approved_outcome_enters_sanitized_case_memory(tmp_path):
    engine, db, job, outcome = _setup(tmp_path)
    try:
        memory = index_verified_outcome(db, outcome)
        assert memory is not None
        assert "Jane Doe" not in memory.searchable_text
        assert "customer:" not in memory.searchable_text.lower()
        assert search_case_memories(db, job=job, query="not cooling clogged filter", limit=5) == []

        other = Job(customer_name="Other", equipment_model="FK4B-001", technician_notes="warm air clogged filter", status="closed")
        db.add(other)
        db.flush()
        results = search_case_memories(db, job=other, query="warm air clogged filter", limit=5)
        assert len(results) == 1
        assert results[0]["evidence_type"] == "verified_historical_case"

        outcome.approved_for_retrieval = False
        assert index_verified_outcome(db, outcome) is None
        db.flush()
        assert search_case_memories(db, job=other, query="warm air clogged filter", limit=5) == []
    finally:
        db.close()
        engine.dispose()


def test_evaluation_compares_predictions_and_reports_metrics(tmp_path):
    engine, db, _job, outcome = _setup(tmp_path)
    try:
        evaluation = evaluate_closed_outcome(db, outcome)
        db.commit()
        assert evaluation.top1_cause_correct is True
        assert evaluation.top3_cause_correct is True
        assert evaluation.recommendation_accepted is True
        details = json.loads(evaluation.details)
        assert details["actual_cause"] == "Clogged return air filter"
        metrics = evaluation_metrics(db)
        assert metrics["evaluations"] == 1
        assert metrics["top1_accuracy"] == 1.0
        assert metrics["technician_acceptance_rate"] == 1.0
        assert metrics["first_time_fix_rate"] == 1.0
    finally:
        db.close()
        engine.dispose()


def test_sanitizer_redacts_contact_information():
    value = sanitize_text("Call Jane at 415-555-0199 or jane@example.com")
    assert "415-555-0199" not in value
    assert "jane@example.com" not in value
