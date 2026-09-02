import json

from app.agent.context import ToolContext
from app.agent.contracts import HypothesisInput, HypothesisOutput
from app.agent.tool_registry import ToolDefinition
from app.agent.tools.job_tools import OPEN_JOB_STATUSES
from app.models import DiagnosticHypothesis


def create_diagnostic_hypothesis(context: ToolContext, payload: HypothesisInput) -> HypothesisOutput:
    count = context.db.query(DiagnosticHypothesis).filter(
        DiagnosticHypothesis.diagnostic_run_id == context.agent_run.diagnostic_run_id
    ).count()
    if payload.selected:
        context.db.query(DiagnosticHypothesis).filter(
            DiagnosticHypothesis.diagnostic_run_id == context.agent_run.diagnostic_run_id
        ).update({"selected": False})
    hypothesis = DiagnosticHypothesis(
        diagnostic_run_id=context.agent_run.diagnostic_run_id,
        rank=count + 1,
        cause=payload.cause,
        confidence=payload.confidence,
        supporting_evidence=json.dumps([item.model_dump() for item in payload.supporting_evidence]),
        contradicting_evidence=json.dumps(payload.contradicting_evidence),
        next_test=payload.next_test,
        selected=payload.selected,
    )
    context.db.add(hypothesis)
    context.db.flush()
    return HypothesisOutput(hypothesis_id=hypothesis.id, rank=hypothesis.rank)


DIAGNOSTIC_TOOLS = [
    ToolDefinition(
        "create_diagnostic_hypothesis",
        "Persist a ranked diagnostic hypothesis with supporting and contradicting evidence.",
        HypothesisInput,
        HypothesisOutput,
        create_diagnostic_hypothesis,
        OPEN_JOB_STATUSES,
    )
]
