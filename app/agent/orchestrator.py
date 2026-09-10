"""Stage 6 plan/act/observe loop.

The model can request only tools that are registered and allowed by the selected
Skill. The executor remains the authority for validation, state transitions, and
auditing; this module only coordinates model turns and durable continuation.
"""

import json
import os
from typing import Any

from openai import OpenAI
from sqlalchemy.orm import Session

from app.agent.tool_executor import ApprovalRequired, ToolExecutionError, ToolExecutor
from app.agent.tool_registry import ToolRegistry
from app.agent.skills.registry import SkillRegistry
from app.models import AgentRun, DiagnosticRun, JobEvent, utcnow


class AgentOrchestrationError(RuntimeError):
    pass


FORCED_ESCALATION_SIGNALS = {
    "refrigerant leak": "Suspected refrigerant leak requires qualified technician escalation.",
    "hissing sound": "Possible refrigerant release requires qualified technician escalation.",
    "burning smell": "Burning smell may indicate an electrical fire hazard.",
    "live voltage": "Live-voltage exposure requires qualified technician escalation.",
    "electric shock": "Electrical shock hazard requires qualified technician escalation.",
    "exposed live": "Exposed live electrical parts require qualified technician escalation.",
}


def forced_escalation_reason(notes: str, error_code: str | None = None) -> str | None:
    """Return a non-optional safety escalation reason from intake evidence."""
    text = " ".join(filter(None, [notes, error_code])).lower()
    for signal, reason in FORCED_ESCALATION_SIGNALS.items():
        if signal in text:
            return reason
    return None


def _value(item: Any, key: str, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


class AgentOrchestrator:
    def __init__(self, tool_registry: ToolRegistry, skill_registry: SkillRegistry,
                 tool_executor: ToolExecutor, client: OpenAI | None = None):
        self.tool_registry = tool_registry
        self.skill_registry = skill_registry
        self.tool_executor = tool_executor
        self.client = client

    def _client(self) -> OpenAI:
        if self.client is not None:
            return self.client
        key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise AgentOrchestrationError("OPENAI_API_KEY is not configured")
        base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        if not base_url and os.getenv("OPENROUTER_API_KEY"):
            base_url = "https://openrouter.ai/api/v1"
        return OpenAI(
            api_key=key,
            base_url=base_url or None,
            timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "45")),
            max_retries=0,
        )

    @staticmethod
    def _uses_openrouter() -> bool:
        """Whether the active OpenAI-compatible endpoint is OpenRouter.

        OpenRouter's Responses compatibility endpoint does not support the
        stateful ``previous_response_id`` continuation parameter.  It accepts
        a new stateless turn instead, so we carry the relevant tool result in
        the next prompt below.
        """
        return bool(os.getenv("OPENROUTER_API_KEY")) or "openrouter.ai" in os.getenv("OPENAI_BASE_URL", "").lower()

    def _instructions(self, run: AgentRun) -> str:
        skill = self.skill_registry.get(run.skill_name or "general_hvac_triage")
        return (
            "You are a conservative HVAC field diagnostic agent. Follow the selected Skill exactly. "
            "Never ask the technician a question in plain text. Any request for information must use request_technician_measurement. Every response must call exactly one tool until complete_diagnosis or escalate_to_human is called. "
            "Use tools for evidence and measurements; never invent measurements or citations. "
            "Do not claim diagnosis completion until calling complete_diagnosis. "
            "If evidence is insufficient, request a technician measurement or escalate. "
            "Never perform a high-risk action directly.\n\n"
            f"Selected Skill: {skill.name} v{skill.version}\n"
            f"Workflow: {skill.description}\n"
            f"Required checks: {', '.join(skill.required_checks)}\n"
            f"Completion criteria: {', '.join(skill.completion_criteria)}\n"
            f"Escalate when: {', '.join(skill.escalation_conditions)}"
        )

    def _tools_for(self, run: AgentRun) -> list[dict]:
        skill = self.skill_registry.get(run.skill_name or "general_hvac_triage")
        allowed = set(skill.allowed_tools)
        # A technician (or the benchmark fixture) supplies measurements through
        # the API. The model may request a measurement but must not fabricate
        # or re-submit it itself.
        allowed.discard("record_measurement")
        output = []
        for schema in self.tool_registry.schemas():
            if schema["name"] in allowed:
                output.append({key: schema[key] for key in ("type", "name", "description", "parameters", "strict")})
        return output

    def _initial_input(self, run: AgentRun) -> str:
        job = run.job
        equipment = job.equipment
        return (
            f"Begin diagnosis for work order {job.id}.\n"
            f"Technician notes: {job.technician_notes}\n"
            f"Error code: {job.error_code or 'none'}\n"
            f"Equipment: {(equipment.manufacturer + ' ' if equipment else '')}{job.equipment_model}\n"
            f"Equipment type: {equipment.equipment_type if equipment else 'unknown'}"
        )

    def _response(self, run: AgentRun, input_value: Any):
        if self._uses_openrouter():
            return self._openrouter_tool_response(run, input_value)
        # OpenRouter uses stateless Responses turns.  Reattach the work-order
        # context whenever we are continuing from a tool observation.
        if self._uses_openrouter() and isinstance(input_value, list):
            input_value = [
                {"role": "user", "content": self._initial_input(run)},
                *input_value,
            ]
        kwargs = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            "instructions": self._instructions(run),
            "input": input_value,
            "tools": self._tools_for(run),
            # A terminal diagnosis is itself a tool call.  Requiring a tool on
            # every turn prevents a model from abandoning a paused workflow
            # with an unstructured prose question or recommendation.
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "max_output_tokens": 1200,
        }
        if run.last_response_id and not self._uses_openrouter():
            kwargs["previous_response_id"] = run.last_response_id
        try:
            return self._client().responses.create(**kwargs, timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "45")))
        except Exception as exc:
            raise AgentOrchestrationError(f"LLM response failed: {exc}") from exc

    def _openrouter_tool_response(self, run: AgentRun, input_value: Any) -> dict:
        """Use OpenRouter's stable Chat Completions tools protocol.

        OpenRouter's Responses compatibility endpoint is stateless and, with
        ``openrouter/auto``, can take minutes to route every tool round.  Chat
        Completions is the native compatibility path and is also what the
        conversational assistant uses successfully.
        """
        if isinstance(input_value, str):
            latest_input = input_value
        elif isinstance(input_value, list):
            latest_input = "\n".join(
                str(item.get("content", item)) if isinstance(item, dict) else str(item)
                for item in input_value
            )
        else:
            latest_input = str(input_value)
        messages = [
            {"role": "system", "content": self._instructions(run)},
            {"role": "user", "content": self._initial_input(run)},
            {"role": "user", "content": latest_input},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for tool in self._tools_for(run)
        ]
        try:
            completion = self._client().chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                messages=messages,
                tools=tools,
                tool_choice="required",
                parallel_tool_calls=False,
                max_tokens=1200,
                timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "45")),
            )
        except Exception as exc:
            raise AgentOrchestrationError(f"LLM response failed: {exc}") from exc

        choice = completion.choices[0] if completion.choices else None
        message = choice.message if choice else None
        tool_calls = getattr(message, "tool_calls", None) or []
        output = [
            {
                "type": "function_call",
                "call_id": call.id,
                "name": call.function.name,
                "arguments": call.function.arguments,
            }
            for call in tool_calls
        ]
        usage = getattr(completion, "usage", None)
        return {
            "id": getattr(completion, "id", None),
            "usage": {
                "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
            },
            "output": output,
            "output_text": getattr(message, "content", None) or "",
        }

    def run(self, db: Session, run_id: int, operator_message: str | None = None) -> dict:
        run = db.get(AgentRun, run_id)
        if run is None:
            raise AgentOrchestrationError("Agent run not found")
        if run.status in {"completed", "failed", "cancelled", "escalated"}:
            return self._result(run)
        if not run.skill_name:
            run.skill_name = "general_hvac_triage"
        safety_reason = forced_escalation_reason(run.job.technician_notes, run.job.error_code)
        if safety_reason:
            try:
                self.tool_executor.execute(
                    db=db,
                    agent_run_id=run.id,
                    tool_name="escalate_to_human",
                    arguments={"reason": safety_reason, "priority": "high"},
                )
            except ToolExecutionError as exc:
                raise AgentOrchestrationError(f"Safety escalation failed: {exc}") from exc
            run = db.get(AgentRun, run_id)
            db.add(JobEvent(
                job_id=run.job_id,
                event_type="forced_safety_escalation",
                event_data=json.dumps({"agent_run_id": run.id, "reason": safety_reason}),
            ))
            db.commit()
            return self._result(run, message=safety_reason)
        if operator_message:
            input_value: Any = [{"role": "user", "content": operator_message}]
            db.add(JobEvent(job_id=run.job_id, event_type="technician_agent_message", event_data=json.dumps({"agent_run_id": run.id, "message": operator_message})))
        else:
            input_value = json.loads(run.pending_input) if run.pending_input else self._initial_input(run)
        run.status = "running"
        db.commit()

        try:
            while run.current_step < run.max_steps:
                response = self._response(run, input_value)
                usage = _value(response, "usage", {}) or {}
                input_tokens = _value(usage, "input_tokens", 0) or 0
                output_tokens = _value(usage, "output_tokens", 0) or 0
                # Keep provider-reported usage in the audit trail so evaluation
                # can report measured tokens instead of character estimates.
                db.add(JobEvent(
                    job_id=run.job_id,
                    event_type="agent_llm_usage",
                    event_data=json.dumps({
                        "agent_run_id": run.id,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }),
                ))
                response_id = _value(response, "id")
                if response_id:
                    run.last_response_id = response_id
                output = _value(response, "output", []) or []
                function_calls = [item for item in output if _value(item, "type") == "function_call"]
                if not function_calls:
                    text = _value(response, "output_text") or ""
                    run.final_message = text
                    run.status = "failed"
                    run.error_message = "Agent ended without calling complete_diagnosis"
                    run.completed_at = utcnow()
                    db.add(JobEvent(job_id=run.job_id, event_type="agent_stopped_without_completion", event_data=json.dumps({"message": text})))
                    db.commit()
                    return self._result(run, message=text)

                call = function_calls[0]
                tool_name = _value(call, "name")
                raw_arguments = _value(call, "arguments", "{}") or "{}"
                try:
                    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be a JSON object")
                except (TypeError, json.JSONDecodeError, ValueError) as exc:
                    raise AgentOrchestrationError(f"Invalid arguments for {tool_name}: {exc}") from exc

                try:
                    tool_result = self.tool_executor.execute(
                        db=db, agent_run_id=run.id, tool_name=tool_name, arguments=arguments,
                    )
                except ApprovalRequired as exc:
                    run = db.get(AgentRun, run_id)
                    run.status = "waiting_for_approval"
                    run.pending_approval_call = json.dumps({
                        "approval_id": exc.approval_id,
                        "call_id": _value(call, "call_id") or _value(call, "id"),
                        "tool_name": tool_name,
                        "arguments": arguments,
                    }, sort_keys=True)
                    db.commit()
                    return self._result(run, pending_action={"approval_id": exc.approval_id, "tool_name": tool_name, "risk": "high"})
                except ToolExecutionError as exc:
                    run = db.get(AgentRun, run_id)
                    run.status = "failed"
                    run.error_message = str(exc)
                    run.completed_at = utcnow()
                    db.commit()
                    return self._result(run, message=str(exc))

                run = db.get(AgentRun, run_id)
                call_id = _value(call, "call_id") or _value(call, "id")
                if self._uses_openrouter():
                    # OpenRouter rejects previous_response_id (it must be
                    # null), and a bare function_call_output is not a valid
                    # standalone request there.  Supply a compact, explicit
                    # observation so the next stateless turn can continue.
                    input_value = [{
                        "role": "user",
                        "content": (
                            f"Diagnostic tool result from {tool_name}: "
                            f"{json.dumps(tool_result['result'])}. "
                            "Continue the selected workflow and call the next required tool."
                        ),
                    }]
                else:
                    input_value = [{
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(tool_result["result"]),
                    }]
                run.pending_input = json.dumps(input_value)
                db.add(JobEvent(job_id=run.job_id, event_type="agent_tool_observed", event_data=json.dumps({"tool_name": tool_name, "agent_run_id": run.id})))
                db.commit()

                # Measurement requests and escalations intentionally pause the loop.
                if run.status in {"escalated"} or run.job.status == "waiting_for_technician":
                    run.status = "waiting_for_technician" if run.job.status == "waiting_for_technician" else run.status
                    db.commit()
                    return self._result(run, pending_action=tool_result["result"])

                if run.status == "completed":
                    diagnostic = db.get(DiagnosticRun, run.diagnostic_run_id)
                    if diagnostic:
                        run.final_message = (
                            f"Diagnosis recorded for work order #{run.job_id}. "
                            f"Most likely cause: {diagnostic.likely_cause}. "
                            f"Confidence: {diagnostic.confidence:.0%}. "
                            f"Recommended action: {diagnostic.recommendation}"
                        )
                    run.pending_input = None
                    db.commit()
                    return self._result(run)
                input_value = json.loads(run.pending_input)

            run = db.get(AgentRun, run_id)
            run.status = "failed"
            run.error_message = "Agent reached maximum tool steps without completing diagnosis"
            run.completed_at = utcnow()
            db.commit()
            return self._result(run, message=run.error_message)
        except AgentOrchestrationError:
            run = db.get(AgentRun, run_id)
            run.status = "failed"
            run.error_message = "Agent orchestration failed"
            run.completed_at = utcnow()
            db.commit()
            raise

    @staticmethod
    def _result(run: AgentRun, message: str | None = None, pending_action: dict | None = None) -> dict:
        return {
            "run_id": run.id,
            "status": run.status,
            "current_step": run.current_step,
            "message": message if message is not None else run.final_message,
            "pending_action": pending_action,
        }
