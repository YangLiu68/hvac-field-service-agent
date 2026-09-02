import json

from app.agent.context import ToolContext
from app.agent.contracts import (
    CompleteDiagnosisInput,
    CompleteDiagnosisOutput,
    EscalationInput,
    EscalationOutput,
    JobStatusInput,
    JobStatusOutput,
    MeasurementRequestInput,
    MeasurementRequestOutput,
    MeasurementResultInput,
    MeasurementResultOutput,
)
from app.agent.tool_registry import ToolDefinition
from app.agent.tools.job_tools import OPEN_JOB_STATUSES
from app.models import JobEvent, MeasurementRequest, MeasurementResult, utcnow


ALLOWED_STATUS_TRANSITIONS = {
    "open": {"diagnosing", "waiting_for_technician", "escalated"},
    "diagnosing": {"waiting_for_technician", "awaiting_technician_feedback", "escalated", "open"},
    "waiting_for_technician": {"diagnosing", "escalated", "open"},
    "awaiting_technician_feedback": {"in_service", "diagnosing", "escalated"},
    "in_service": {"diagnosing", "awaiting_technician_feedback", "escalated"},
    "escalated": {"diagnosing", "open", "waiting_for_technician"},
}


def _event(context: ToolContext, event_type: str, data: dict) -> None:
    context.db.add(JobEvent(job_id=context.job.id, event_type=event_type, event_data=json.dumps(data)))


def request_technician_measurement(context: ToolContext, payload: MeasurementRequestInput) -> MeasurementRequestOutput:
    request = MeasurementRequest(job_id=context.job.id, agent_run_id=context.agent_run.id, **payload.model_dump())
    context.db.add(request)
    context.db.flush()
    previous_status = context.job.status
    context.job.status = "waiting_for_technician"
    context.job.updated_at = utcnow()
    _event(context, "measurement_requested", {"request_id": request.id, "measurement_type": payload.measurement_type, "previous_status": previous_status})
    return MeasurementRequestOutput(request_id=request.id, status=request.status)


def record_measurement(context: ToolContext, payload: MeasurementResultInput) -> MeasurementResultOutput:
    request = context.db.get(MeasurementRequest, payload.request_id)
    if request is None or request.job_id != context.job.id:
        raise ValueError("Measurement request does not belong to this job")
    if request.status != "pending":
        raise ValueError("Measurement request is not pending")
    result = MeasurementResult(request_id=request.id, **payload.model_dump(exclude={"request_id"}))
    context.db.add(result)
    context.db.flush()
    request.status = "completed"
    request.completed_at = utcnow()
    context.job.status = "diagnosing"
    context.job.updated_at = utcnow()
    _event(context, "measurement_recorded", {"request_id": request.id, "result_id": result.id, "measurement_type": request.measurement_type})
    return MeasurementResultOutput(result_id=result.id, request_id=request.id, status=request.status)


def update_job_status(context: ToolContext, payload: JobStatusInput) -> JobStatusOutput:
    previous = context.job.status
    allowed = ALLOWED_STATUS_TRANSITIONS.get(previous, set())
    if payload.status not in allowed:
        raise ValueError(f"Invalid job status transition: {previous} -> {payload.status}")
    context.job.status = payload.status
    context.job.updated_at = utcnow()
    _event(context, "job_status_updated", {"previous_status": previous, "status": payload.status, "reason": payload.reason})
    return JobStatusOutput(previous_status=previous, status=payload.status)


def escalate_to_human(context: ToolContext, payload: EscalationInput) -> EscalationOutput:
    context.job.status = "escalated"
    context.job.updated_at = utcnow()
    context.agent_run.status = "escalated"
    _event(context, "human_escalation_requested", payload.model_dump())
    return EscalationOutput(status="escalated", priority=payload.priority)


def complete_diagnosis(context: ToolContext, payload: CompleteDiagnosisInput) -> CompleteDiagnosisOutput:
    diagnostic_run = context.agent_run.diagnostic_run
    diagnostic_run.status = "completed"
    diagnostic_run.likely_cause = payload.likely_cause
    diagnostic_run.confidence = payload.confidence
    diagnostic_run.recommendation = payload.recommendation
    diagnostic_run.completed_at = utcnow()
    context.job.status = "awaiting_technician_feedback"
    context.job.updated_at = utcnow()
    context.agent_run.status = "completed"
    context.agent_run.completed_at = utcnow()
    _event(context, "diagnosis_completed", {"diagnostic_run_id": diagnostic_run.id, "likely_cause": payload.likely_cause, "confidence": payload.confidence})
    return CompleteDiagnosisOutput(diagnostic_run_id=diagnostic_run.id, job_status=context.job.status)


WORKFLOW_TOOLS = [
    ToolDefinition("request_technician_measurement", "Request a specific field measurement and pause diagnosis until a technician responds.", MeasurementRequestInput, MeasurementRequestOutput, request_technician_measurement, OPEN_JOB_STATUSES),
    ToolDefinition("record_measurement", "Record the technician's result for a pending measurement request.", MeasurementResultInput, MeasurementResultOutput, record_measurement, OPEN_JOB_STATUSES),
    ToolDefinition("update_job_status", "Apply a validated work-order state transition and record the reason.", JobStatusInput, JobStatusOutput, update_job_status, OPEN_JOB_STATUSES),
    ToolDefinition("escalate_to_human", "Stop autonomous progress and escalate the case to a qualified human.", EscalationInput, EscalationOutput, escalate_to_human, OPEN_JOB_STATUSES),
    ToolDefinition("complete_diagnosis", "Complete the diagnostic run with a supported cause, confidence, and next action.", CompleteDiagnosisInput, CompleteDiagnosisOutput, complete_diagnosis, OPEN_JOB_STATUSES),
]
