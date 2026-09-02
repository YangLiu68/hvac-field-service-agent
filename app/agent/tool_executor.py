import json
import time
from datetime import timedelta

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.agent.context import ToolContext
from app.agent.tool_registry import ToolRegistry
from app.models import AgentRun, AgentStep, ApprovalRequest, ToolCall, utcnow
from app.agent.approval_policy import approval_requirement


class ToolExecutionError(RuntimeError):
    pass


class ApprovalRequired(ToolExecutionError):
    def __init__(self, approval_id: int, message: str):
        super().__init__(message)
        self.approval_id = approval_id


class ToolExecutor:
    # Idempotent lookups are safe to replay when a stateless compatible LLM
    # endpoint asks for the same context again.  State-changing tools remain
    # protected by the duplicate-call block below.
    READ_ONLY_TOOLS = {
        "get_job_context",
        "get_equipment_profile",
        "search_service_manuals",
        "search_error_codes",
        "search_verified_repairs",
    }

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(
        self,
        *,
        db: Session,
        agent_run_id: int,
        tool_name: str,
        arguments: dict,
        approved: bool = False,
    ) -> dict:
        agent_run = db.get(AgentRun, agent_run_id)
        if agent_run is None:
            raise ToolExecutionError("Agent run not found")
        if agent_run.status in {"completed", "failed", "cancelled"}:
            raise ToolExecutionError(f"Agent run is not executable: {agent_run.status}")
        if agent_run.current_step >= agent_run.max_steps:
            raise ToolExecutionError("Agent run reached its maximum number of steps")

        try:
            definition = self.registry.get(tool_name)
        except KeyError as exc:
            raise ToolExecutionError(str(exc)) from exc
        if definition.allowed_job_statuses and agent_run.job.status not in definition.allowed_job_statuses:
            raise ToolExecutionError(
                f"Tool {tool_name} is not allowed while job status is {agent_run.job.status}"
            )
        if agent_run.skill_name:
            from app.agent.skills import build_default_skill_registry
            allowed_tools = build_default_skill_registry(self.registry).get(agent_run.skill_name).allowed_tools
            if tool_name not in allowed_tools:
                raise ToolExecutionError(f"Tool {tool_name} is not allowed by skill {agent_run.skill_name}")
        requirement = approval_requirement(tool_name, arguments)
        if (definition.requires_approval or requirement) and not approved:
            risk_level, reason = requirement or ("high", f"Tool {tool_name} requires human approval")
            canonical_arguments = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
            pending = db.query(ApprovalRequest).filter(
                ApprovalRequest.agent_run_id == agent_run.id,
                ApprovalRequest.tool_name == tool_name,
                ApprovalRequest.arguments == canonical_arguments,
                ApprovalRequest.status == "pending",
            ).first()
            if pending is None:
                pending = ApprovalRequest(
                    job_id=agent_run.job_id,
                    agent_run_id=agent_run.id,
                    tool_name=tool_name,
                    arguments=canonical_arguments,
                    risk_level=risk_level,
                    reason=reason,
                    expires_at=utcnow() + timedelta(minutes=30),
                )
                db.add(pending)
                agent_run.status = "waiting_for_approval"
                db.flush()
                db.commit()
            raise ApprovalRequired(pending.id, f"Tool {tool_name} requires approval (request {pending.id})")

        canonical_arguments = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
        duplicate = db.query(ToolCall).filter(
            ToolCall.agent_run_id == agent_run.id,
            ToolCall.tool_name == tool_name,
            ToolCall.arguments == canonical_arguments,
            ToolCall.status == "completed",
        ).first()
        if duplicate is not None:
            if tool_name in self.READ_ONLY_TOOLS:
                try:
                    cached_result = json.loads(duplicate.result or "{}")
                except json.JSONDecodeError:
                    cached_result = {}
                return {
                    "tool_call_id": duplicate.id,
                    "agent_step_id": duplicate.agent_step_id,
                    "tool_name": tool_name,
                    "status": "cached",
                    "duration_ms": 0,
                    "result": cached_result,
                }
            raise ToolExecutionError(f"Duplicate tool call blocked: {tool_name} with identical arguments")

        step_number = agent_run.current_step + 1
        started = time.perf_counter()
        step = AgentStep(
            agent_run_id=agent_run.id,
            step_number=step_number,
            step_type="tool_call",
            status="running",
            input_data=json.dumps({"tool_name": tool_name, "arguments": arguments}, sort_keys=True),
        )
        db.add(step)
        db.flush()

        try:
            payload = definition.input_model.model_validate(arguments)
            result = definition.output_model.model_validate(
                definition.handler(ToolContext(db=db, job=agent_run.job, agent_run=agent_run), payload)
            )
            duration_ms = int((time.perf_counter() - started) * 1000)
            if duration_ms > definition.timeout_seconds * 1000:
                raise RuntimeError(f"Tool {tool_name} exceeded its {definition.timeout_seconds}s timeout")
            result_data = result.model_dump(mode="json")
            step.status = "completed"
            step.output_data = json.dumps(result_data)
            step.completed_at = utcnow()
            call = ToolCall(
                diagnostic_run_id=agent_run.diagnostic_run_id,
                agent_run_id=agent_run.id,
                agent_step_id=step.id,
                tool_name=tool_name,
                arguments=canonical_arguments,
                result=json.dumps(result_data),
                status="completed",
                duration_ms=duration_ms,
            )
            db.add(call)
            agent_run.current_step = step_number
            db.commit()
            db.refresh(call)
            return {
                "tool_call_id": call.id,
                "agent_step_id": step.id,
                "tool_name": tool_name,
                "status": "completed",
                "duration_ms": duration_ms,
                "result": result_data,
            }
        except (ValidationError, ValueError, RuntimeError) as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            db.rollback()
            agent_run = db.get(AgentRun, agent_run_id)
            failed_step = AgentStep(
                agent_run_id=agent_run_id,
                step_number=step_number,
                step_type="tool_call",
                status="failed",
                input_data=json.dumps({"tool_name": tool_name, "arguments": arguments}),
                error_message=str(exc),
                completed_at=utcnow(),
            )
            db.add(failed_step)
            db.flush()
            call = ToolCall(
                diagnostic_run_id=agent_run.diagnostic_run_id,
                agent_run_id=agent_run.id,
                agent_step_id=failed_step.id,
                tool_name=tool_name,
                arguments=canonical_arguments,
                result=json.dumps({"error": str(exc)}),
                status="failed",
                duration_ms=duration_ms,
            )
            db.add(call)
            agent_run.current_step = step_number
            db.commit()
            raise ToolExecutionError(f"Tool {tool_name} failed: {exc}") from exc
