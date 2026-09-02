from app.agent.context import ToolContext
from app.agent.contracts import EmptyInput, EquipmentHistoryOutput, EquipmentProfileOutput, JobContextOutput
from app.agent.tool_registry import ToolDefinition
from app.models import Job, MeasurementRequest


def get_equipment_profile(context: ToolContext, _payload: EmptyInput) -> EquipmentProfileOutput:
    equipment = context.job.equipment
    return EquipmentProfileOutput(
        equipment_id=equipment.id if equipment else None,
        manufacturer=equipment.manufacturer if equipment else None,
        model_number=equipment.model_number if equipment else context.job.equipment_model,
        equipment_type=equipment.equipment_type if equipment else None,
        serial_number=equipment.serial_number if equipment else None,
        refrigerant=equipment.refrigerant if equipment else None,
        installation_date=equipment.installation_date if equipment else None,
    )


def get_job_context(context: ToolContext, _payload: EmptyInput) -> JobContextOutput:
    return JobContextOutput(
        job_id=context.job.id,
        customer_name=context.job.customer_name,
        status=context.job.status,
        technician_notes=context.job.technician_notes,
        error_code=context.job.error_code,
        equipment=get_equipment_profile(context, EmptyInput()),
        diagnostic_run_count=len(context.job.diagnostic_runs),
        pending_measurements=context.db.query(MeasurementRequest).filter(
            MeasurementRequest.job_id == context.job.id,
            MeasurementRequest.status == "pending",
        ).count(),
    )


def get_equipment_history(context: ToolContext, _payload: EmptyInput) -> EquipmentHistoryOutput:
    if context.job.equipment_id is None:
        return EquipmentHistoryOutput(jobs=[])
    jobs = context.db.query(Job).filter(
        Job.equipment_id == context.job.equipment_id,
        Job.id != context.job.id,
    ).order_by(Job.created_at.desc()).limit(20).all()
    return EquipmentHistoryOutput(
        jobs=[
            {
                "job_id": job.id,
                "status": job.status,
                "error_code": job.error_code,
                "technician_notes": job.technician_notes,
                "created_at": job.created_at.isoformat(),
                "verified_outcomes": [
                    {
                        "actual_cause": outcome.actual_cause,
                        "repair_action": outcome.repair_action,
                        "first_time_fix": outcome.first_time_fix,
                    }
                    for outcome in job.outcomes
                    if outcome.approved_for_retrieval
                ],
            }
            for job in jobs
        ]
    )


OPEN_JOB_STATUSES = frozenset({"open", "diagnosing", "awaiting_technician_feedback", "in_service", "waiting_for_technician", "escalated"})

JOB_TOOLS = [
    ToolDefinition("get_job_context", "Read the work order, equipment summary, and current diagnostic state.", EmptyInput, JobContextOutput, get_job_context, OPEN_JOB_STATUSES),
    ToolDefinition("get_equipment_profile", "Read manufacturer, model, serial, refrigerant, and equipment metadata.", EmptyInput, EquipmentProfileOutput, get_equipment_profile, OPEN_JOB_STATUSES),
    ToolDefinition("get_equipment_history", "Read prior work orders and approved outcomes for the same equipment.", EmptyInput, EquipmentHistoryOutput, get_equipment_history, OPEN_JOB_STATUSES),
]
