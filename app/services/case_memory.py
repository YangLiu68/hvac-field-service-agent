import json
import re
from datetime import datetime

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models import (
    AgentRun,
    CaseMemory,
    DiagnosticEvaluation,
    DiagnosticHypothesis,
    DiagnosticRun,
    Job,
    TechnicianFeedback,
    VerifiedOutcome,
)


STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "issue", "problem", "hvac"}


def sanitize_text(value: str | None) -> str:
    """Remove common PII before a case is persisted or embedded."""
    value = value or ""
    value = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[redacted-email]", value)
    value = re.sub(r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-])\d{3}[\s.-]\d{4}(?!\d)", "[redacted-phone]", value)
    value = re.sub(r"\b(?:customer|resident|owner|tenant)\s*:\s*[^,.;\n]+", "[redacted-person]", value, flags=re.I)
    return " ".join(value.split())[:4000]


def _terms(value: str | None) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9][a-z0-9_/-]+", (value or "").lower()) if token not in STOPWORDS}


def _cause_matches(predicted: str | None, actual: str) -> bool:
    predicted_terms = _terms(predicted)
    actual_terms = _terms(actual)
    if not predicted_terms or not actual_terms:
        return False
    return bool(predicted_terms & actual_terms) and (
        predicted_terms <= actual_terms or actual_terms <= predicted_terms or len(predicted_terms & actual_terms) >= 2
    )


def index_verified_outcome(db: Session, outcome: VerifiedOutcome) -> CaseMemory | None:
    """Create/update memory only after a supervisor explicitly approves retrieval."""
    existing = db.query(CaseMemory).filter(CaseMemory.outcome_id == outcome.id).first()
    if not outcome.approved_for_retrieval:
        if existing:
            db.delete(existing)
        return None
    job = outcome.job or db.get(Job, outcome.job_id)
    equipment = job.equipment
    symptoms = sanitize_text(job.technician_notes)
    actual_cause = sanitize_text(outcome.actual_cause)
    repair_action = sanitize_text(outcome.repair_action)
    searchable = sanitize_text(" ".join(filter(None, [symptoms, job.error_code, actual_cause, repair_action])))
    values = {
        "outcome_id": outcome.id,
        "job_id": job.id,
        "manufacturer": sanitize_text(equipment.manufacturer) if equipment else None,
        "equipment_model": sanitize_text(job.equipment_model),
        "equipment_type": sanitize_text(equipment.equipment_type) if equipment else None,
        "error_code": sanitize_text(job.error_code),
        "symptoms": symptoms,
        "actual_cause": actual_cause,
        "repair_action": repair_action,
        "first_time_fix": outcome.first_time_fix,
        "return_visit_required": outcome.return_visit_required,
        "searchable_text": searchable,
    }
    if existing:
        for key, value in values.items():
            setattr(existing, key, value)
        return existing
    memory = CaseMemory(**values)
    db.add(memory)
    db.flush()
    return memory


def search_case_memories(db: Session, *, job: Job, query: str, limit: int = 5) -> list[dict]:
    """Lexical, metadata-aware search over approved cases; no unapproved outcome is queried."""
    terms = _terms(query)
    equipment = job.equipment
    rows = db.query(CaseMemory).filter(CaseMemory.job_id != job.id)
    if equipment:
        rows = rows.filter(or_(CaseMemory.equipment_model == equipment.model_number, CaseMemory.equipment_type == equipment.equipment_type))
    candidates = rows.order_by(CaseMemory.created_at.desc()).limit(200).all()
    ranked = []
    for memory in candidates:
        haystack = _terms(memory.searchable_text)
        overlap = len(terms & haystack)
        metadata_bonus = int(bool(equipment and memory.equipment_model == equipment.model_number))
        error_bonus = int(bool(job.error_code and memory.error_code == job.error_code))
        score = overlap + 2 * metadata_bonus + 2 * error_bonus
        if score > 0:
            ranked.append((score, memory.created_at or datetime.min, memory))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [{
        "case_id": memory.id,
        "job_id": memory.job_id,
        "manufacturer": memory.manufacturer,
        "equipment_model": memory.equipment_model,
        "equipment_type": memory.equipment_type,
        "error_code": memory.error_code,
        "symptoms": memory.symptoms,
        "actual_cause": memory.actual_cause,
        "repair_action": memory.repair_action,
        "first_time_fix": memory.first_time_fix,
        "return_visit_required": memory.return_visit_required,
        "score": float(score),
        "evidence_type": "verified_historical_case",
    } for score, _created, memory in ranked[:limit]]


def evaluate_closed_outcome(db: Session, outcome: VerifiedOutcome) -> DiagnosticEvaluation:
    job = outcome.job or db.get(Job, outcome.job_id)
    diagnostic_run = db.query(DiagnosticRun).filter(DiagnosticRun.job_id == job.id).order_by(DiagnosticRun.id.desc()).first()
    hypotheses = []
    if diagnostic_run:
        hypotheses = db.query(DiagnosticHypothesis).filter(
            DiagnosticHypothesis.diagnostic_run_id == diagnostic_run.id
        ).order_by(DiagnosticHypothesis.rank.asc()).limit(3).all()
        if not hypotheses and diagnostic_run.likely_cause:
            hypotheses = [DiagnosticHypothesis(cause=diagnostic_run.likely_cause, rank=1, confidence=diagnostic_run.confidence or 0.0)]
    predicted = [hypothesis.cause for hypothesis in hypotheses]
    feedback = db.query(TechnicianFeedback).filter(
        TechnicianFeedback.job_id == job.id
    ).order_by(TechnicianFeedback.created_at.desc()).first()
    agent_run = db.query(AgentRun).filter(AgentRun.job_id == job.id).order_by(AgentRun.id.desc()).first()
    details = {
        "predicted_causes": predicted,
        "actual_cause": sanitize_text(outcome.actual_cause),
        "diagnostic_run_id": diagnostic_run.id if diagnostic_run else None,
        "evidence_scope": "manual_and_agent_hypotheses",
    }
    evaluation = db.query(DiagnosticEvaluation).filter(DiagnosticEvaluation.outcome_id == outcome.id).first()
    values = {
        "job_id": job.id,
        "diagnostic_run_id": diagnostic_run.id if diagnostic_run else None,
        "agent_run_id": agent_run.id if agent_run else None,
        "outcome_id": outcome.id,
        "top1_cause_correct": _cause_matches(predicted[0] if predicted else None, outcome.actual_cause) if predicted else None,
        "top3_cause_correct": any(_cause_matches(item, outcome.actual_cause) for item in predicted) if predicted else None,
        "recommendation_accepted": feedback.recommendation_accepted if feedback else None,
        "first_time_fix": outcome.first_time_fix,
        "return_visit_required": outcome.return_visit_required,
        "details": json.dumps(details),
    }
    if evaluation:
        for key, value in values.items():
            setattr(evaluation, key, value)
    else:
        evaluation = DiagnosticEvaluation(**values)
        db.add(evaluation)
    db.flush()
    return evaluation


def evaluation_metrics(db: Session) -> dict:
    evaluations = db.query(DiagnosticEvaluation).all()
    known_top1 = [item for item in evaluations if item.top1_cause_correct is not None]
    known_top3 = [item for item in evaluations if item.top3_cause_correct is not None]
    accepted = [item for item in evaluations if item.recommendation_accepted is not None]
    return {
        "evaluations": len(evaluations),
        "top1_accuracy": sum(item.top1_cause_correct for item in known_top1) / len(known_top1) if known_top1 else None,
        "top3_accuracy": sum(item.top3_cause_correct for item in known_top3) / len(known_top3) if known_top3 else None,
        "technician_acceptance_rate": sum(item.recommendation_accepted for item in accepted) / len(accepted) if accepted else None,
        "first_time_fix_rate": sum(item.first_time_fix for item in evaluations) / len(evaluations) if evaluations else None,
        "return_visit_rate": sum(item.return_visit_required for item in evaluations) / len(evaluations) if evaluations else None,
    }
