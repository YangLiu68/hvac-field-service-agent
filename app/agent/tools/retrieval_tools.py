from app.agent.context import ToolContext
from app.agent.contracts import (
    ErrorCodeInput,
    EvidenceItem,
    ManualSearchInput,
    ManualSearchOutput,
    VerifiedRepairSearchInput,
    VerifiedRepairSearchOutput,
)
from app.agent.tool_registry import ToolDefinition
from app.agent.tools.job_tools import OPEN_JOB_STATUSES
from app.services.case_memory import search_case_memories
from app.services.rag_service import hybrid_search


def _manual_search(context: ToolContext, query: str, top_k: int) -> ManualSearchOutput:
    equipment = context.job.equipment
    results = hybrid_search(
        query,
        top_k=top_k,
        manufacturer=equipment.manufacturer if equipment else None,
        equipment_model=context.job.equipment_model,
        equipment_type=equipment.equipment_type if equipment else None,
        db=context.db,
    )
    return ManualSearchOutput(results=[EvidenceItem(
        document=item["document"], page=item["page"], section=item.get("section"),
        text=item["text"], score=item["score"],
    ) for item in results])


def search_service_manuals(context: ToolContext, payload: ManualSearchInput) -> ManualSearchOutput:
    return _manual_search(context, payload.query, payload.top_k)


def search_error_codes(context: ToolContext, payload: ErrorCodeInput) -> ManualSearchOutput:
    query = f"error code {payload.error_code} fault troubleshooting diagnostic procedure"
    return _manual_search(context, query, payload.top_k)


def search_verified_repairs(context: ToolContext, payload: VerifiedRepairSearchInput) -> VerifiedRepairSearchOutput:
    return VerifiedRepairSearchOutput(cases=search_case_memories(context.db, job=context.job, query=payload.query, limit=payload.limit))


RETRIEVAL_TOOLS = [
    ToolDefinition("search_service_manuals", "Search model-filtered service manuals and return cited evidence.", ManualSearchInput, ManualSearchOutput, search_service_manuals, OPEN_JOB_STATUSES, timeout_seconds=60),
    ToolDefinition("search_error_codes", "Search service manuals for a reported HVAC error code and troubleshooting procedure.", ErrorCodeInput, ManualSearchOutput, search_error_codes, OPEN_JOB_STATUSES, timeout_seconds=60),
    ToolDefinition("search_verified_repairs", "Search only technician-approved historical repair outcomes for similar evidence.", VerifiedRepairSearchInput, VerifiedRepairSearchOutput, search_verified_repairs, OPEN_JOB_STATUSES),
]
