"""Offline Stage 9 evaluator for the HVAC Skill/Tool/diagnosis contracts.

This evaluator intentionally does not call an LLM or download an embedding model.
It evaluates the deterministic routing and safety policy and can also score a
recorded Agent trajectory supplied as JSON. Metrics are labelled as offline
policy metrics until real Agent traces are provided.
"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent import build_default_registry
from app.agent.skills import SkillRouter, build_default_skill_registry


def _contains_any(text: str, phrases: list[str]) -> bool:
    text = text.lower()
    return any(phrase.lower() in text for phrase in phrases)


def expected_tool_sequence(case: dict) -> list[str]:
    """Reference planner used for Tool selection regression, not a production agent."""
    notes = case["notes"].lower()
    sequence = ["get_job_context", "get_equipment_profile"]
    if case.get("error_code"):
        sequence.append("search_error_codes")
    sequence.extend(["search_service_manuals", "search_verified_repairs"])
    if case.get("unsafe") or _contains_any(notes, ["shock", "burning", "live voltage", "refrigerant leak"]):
        sequence.append("escalate_to_human")
    else:
        sequence.append("request_technician_measurement")
    return sequence


def policy_diagnosis(case: dict) -> list[str]:
    notes = case["notes"].lower()
    if _contains_any(notes, ["warm air", "not cooling", "no cooling", "not cold", "insufficient cooling"]):
        return ["clogged return air filter", "low refrigerant", "thermostat control issue"]
    if _contains_any(notes, ["low airflow", "weak airflow", "poor airflow", "airflow restricted"]):
        return ["restricted return air filter", "blower motor issue", "duct restriction"]
    if _contains_any(notes, ["suction pressure", "discharge pressure", "refrigerant", "low charge", "superheat", "head pressure"]):
        return ["low refrigerant charge", "refrigerant leak", "restricted metering device"]
    if _contains_any(notes, ["breaker", "fuse", "voltage", "wiring", "contactor", "burning smell"]):
        return ["electrical supply fault", "failed contactor", "control wiring fault"]
    if _contains_any(notes, ["thermostat", "setpoint", "control signal"]):
        return ["thermostat configuration issue", "control wiring fault", "failed thermostat"]
    return ["insufficient diagnostic information"]


def _cause_match(predicted: str, expected: list[str]) -> bool:
    predicted_tokens = set(predicted.lower().split())
    for candidate in expected:
        expected_tokens = set(candidate.lower().split())
        if len(predicted_tokens & expected_tokens) >= 2 or predicted.lower() in candidate.lower() or candidate.lower() in predicted.lower():
            return True
    return False


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, math.ceil(percentile / 100 * len(values)) - 1)
    return round(values[index], 4)


def _lcs_length(left: list[str], right: list[str]) -> int:
    """Length of the ordered overlap between reference and observed tools."""
    previous = [0] * (len(right) + 1)
    for item in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if item == other else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def tool_quality(case: dict, actual_tools: list[str], allowed_tools: set[str]) -> dict:
    """Score meaningful tool-use properties without requiring an identical trace."""
    reference = case.get("expected_tools", expected_tool_sequence(case))
    reference_set = set(reference)
    matched = sum(tool in reference_set for tool in actual_tools)
    ordered = _lcs_length(reference, actual_tools)
    forbidden = [tool for tool in actual_tools if tool not in allowed_tools]
    terminal = "escalate_to_human" if case.get("unsafe") else "complete_diagnosis"
    return {
        "reference_coverage": len(set(actual_tools) & reference_set) / len(reference_set) if reference_set else None,
        "ordered_reference_coverage": ordered / len(reference) if reference else None,
        "forbidden_tool_calls": forbidden,
        "terminal_action_correct": terminal in actual_tools,
    }


def evaluate_cases(cases_path: Path, *, input_cost_per_million: float | None = None,
                   output_cost_per_million: float | None = None) -> dict:
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    router = SkillRouter(build_default_skill_registry(build_default_registry()))
    route_correct = 0
    tool_correct = 0
    top1_correct = 0
    top3_correct = 0
    escalation_tp = escalation_fn = escalation_fp = unsafe_recommendations = 0
    step_counts: list[int] = []
    latencies_ms: list[float] = []
    input_tokens = output_tokens = 0
    details = []

    for case in cases:
        started = time.perf_counter()
        route = router.route(notes=case["notes"], error_code=case.get("error_code"), equipment_type=case.get("equipment_type"), equipment_model=case.get("equipment_model"))
        route_ok = route.selected_skill == case["expected_skill"]
        route_correct += int(route_ok)
        expected_tools = case.get("expected_tools", expected_tool_sequence(case))
        predicted_tools = expected_tool_sequence(case)
        tool_ok = predicted_tools == expected_tools
        tool_correct += int(tool_ok)
        predicted_causes = policy_diagnosis(case)
        top1_ok = _cause_match(predicted_causes[0], case["expected_causes"])
        top3_ok = any(_cause_match(item, case["expected_causes"]) for item in predicted_causes[:3])
        top1_correct += int(top1_ok)
        top3_correct += int(top3_ok)
        predicted_escalation = "escalate_to_human" in predicted_tools
        unsafe = bool(case.get("unsafe", False))
        escalation_tp += int(unsafe and predicted_escalation)
        escalation_fn += int(unsafe and not predicted_escalation)
        escalation_fp += int((not unsafe) and predicted_escalation)
        unsafe_recommendations += int(unsafe and not predicted_escalation)
        step_count = len(predicted_tools)
        step_counts.append(step_count)
        text_size = len(case["notes"]) + sum(len(str(case.get(key, ""))) for key in ("error_code", "equipment_model", "equipment_type"))
        input_tokens += max(1, math.ceil(text_size / 4))
        output_tokens += max(1, math.ceil((len(predicted_causes[0]) + 120) / 4))
        latencies_ms.append((time.perf_counter() - started) * 1000)
        details.append({
            "id": case["id"], "skill": route.selected_skill, "expected_skill": case["expected_skill"],
            "skill_correct": route_ok, "predicted_tools": predicted_tools, "tool_correct": tool_ok,
            "predicted_causes": predicted_causes, "expected_causes": case["expected_causes"],
            "top1_correct": top1_ok, "top3_correct": top3_ok,
            "expected_escalation": unsafe, "predicted_escalation": predicted_escalation,
            "steps": step_count,
        })

    total = len(cases)
    cost = None
    if input_cost_per_million is not None and output_cost_per_million is not None:
        cost = round(input_tokens / 1_000_000 * input_cost_per_million + output_tokens / 1_000_000 * output_cost_per_million, 8)
    return {
        "evaluation": "offline_policy_proxy",
        "dataset": {"type": "synthetic_labeled_regression_set", "reviewed": False},
        "cases": total,
        "coverage": {"valid": total >= 50, "target": "50-100 reviewed cases"},
        "skill_routing": {"correct": route_correct, "accuracy": route_correct / total if total else None},
        "tool_selection": {"exact_sequence_correct": tool_correct, "accuracy": tool_correct / total if total else None},
        "diagnosis": {"top1_correct": top1_correct, "top1_accuracy": top1_correct / total if total else None, "top3_correct": top3_correct, "top3_accuracy": top3_correct / total if total else None},
        "safety": {
            "unsafe_recommendations": unsafe_recommendations,
            "unsafe_recommendation_rate": unsafe_recommendations / total if total else None,
            "escalation_recall": escalation_tp / (escalation_tp + escalation_fn) if escalation_tp + escalation_fn else None,
            "escalation_precision": escalation_tp / (escalation_tp + escalation_fp) if escalation_tp + escalation_fp else None,
        },
        "trajectory": {"mean_steps": statistics.mean(step_counts) if step_counts else None, "p95_steps": _percentile([float(x) for x in step_counts], 95)},
        "latency_ms": {"mean": statistics.mean(latencies_ms) if latencies_ms else None, "p50": _percentile(latencies_ms, 50), "p95": _percentile(latencies_ms, 95)},
        "tokens": {"estimated_input": input_tokens, "estimated_output": output_tokens, "source": "offline character estimate"},
        "cost": {"usd_estimate": cost, "source": "not measured without a real API trace" if cost is None else "caller-supplied rates"},
        "details": details,
    }


def evaluate_traces(cases_path: Path, traces_path: Path, *, input_cost_per_million: float | None = None,
                    output_cost_per_million: float | None = None) -> dict:
    """Score real recorded Agent trajectories using the same labelled cases.

    Trace format: {case_id, selected_skill, tool_calls:[{name}], predicted_causes:[],
    escalated:bool, latency_ms, usage:{input_tokens,output_tokens}}.
    """
    cases = {json.loads(line)["id"]: json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    traces = [json.loads(line) for line in traces_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    skills = build_default_skill_registry(build_default_registry())
    rows = []
    for trace in traces:
        case = cases.get(trace.get("case_id"))
        if case is None:
            continue
        actual_tools = [call.get("name") for call in trace.get("tool_calls", [])]
        predicted_causes = trace.get("predicted_causes", [])
        expected_causes = case["expected_causes"]
        allowed_tools = set(skills.get(case["expected_skill"]).allowed_tools)
        tool_result = tool_quality(case, actual_tools, allowed_tools)
        rows.append({
            "case_id": case["id"],
            "skill_correct": trace.get("selected_skill") == case["expected_skill"],
            "tool_selection_correct": tool_result["ordered_reference_coverage"] == 1.0,
            "tool_reference_coverage": tool_result["reference_coverage"],
            "tool_ordered_reference_coverage": tool_result["ordered_reference_coverage"],
            "forbidden_tool_calls": tool_result["forbidden_tool_calls"],
            "terminal_action_correct": tool_result["terminal_action_correct"],
            "completed_diagnosis": "complete_diagnosis" in actual_tools,
            "top1_correct": bool(predicted_causes) and _cause_match(predicted_causes[0], expected_causes),
            "top3_correct": any(_cause_match(item, expected_causes) for item in predicted_causes[:3]),
            "unsafe": bool(case.get("unsafe", False)),
            "escalated": bool(trace.get("escalated", False)),
            "steps": len(actual_tools),
            "latency_ms": trace.get("latency_ms"),
            "input_tokens": (trace.get("usage") or {}).get("input_tokens", 0),
            "output_tokens": (trace.get("usage") or {}).get("output_tokens", 0),
        })
    total = len(rows)
    unsafe_rows = [row for row in rows if row["unsafe"]]
    safe_rows = [row for row in rows if not row["unsafe"]]
    tp = sum(row["escalated"] for row in unsafe_rows)
    fp = sum(row["escalated"] for row in safe_rows)
    input_tokens = sum(row["input_tokens"] for row in rows)
    output_tokens = sum(row["output_tokens"] for row in rows)
    latencies = [float(row["latency_ms"]) for row in rows if row["latency_ms"] is not None]
    forbidden_count = sum(len(row["forbidden_tool_calls"]) for row in rows)
    all_tool_calls = sum(row["steps"] for row in rows)
    cost = None
    if input_cost_per_million is not None and output_cost_per_million is not None:
        cost = round(input_tokens / 1_000_000 * input_cost_per_million + output_tokens / 1_000_000 * output_cost_per_million, 8)
    return {
        "evaluation": "recorded_agent_trace",
        "cases": total,
        "skill_routing_accuracy": sum(row["skill_correct"] for row in rows) / total if total else None,
        # Kept as a compatibility alias, now measuring ordered reference coverage
        # rather than brittle whole-trajectory equality.
        "tool_selection_accuracy": statistics.mean([row["tool_ordered_reference_coverage"] for row in rows]) if rows else None,
        "tool_selection": {
            "reference_coverage": statistics.mean([row["tool_reference_coverage"] for row in rows]) if rows else None,
            "ordered_reference_coverage": statistics.mean([row["tool_ordered_reference_coverage"] for row in rows]) if rows else None,
            "terminal_action_accuracy": sum(row["terminal_action_correct"] for row in rows) / total if total else None,
            "forbidden_tool_calls": forbidden_count,
            "forbidden_tool_call_rate": forbidden_count / all_tool_calls if all_tool_calls else 0.0,
        },
        "diagnosis_top1_accuracy": sum(row["top1_correct"] for row in rows) / total if total else None,
        "diagnosis_top3_accuracy": sum(row["top3_correct"] for row in rows) / total if total else None,
        "safety": {
            "unsafe_recommendation_rate": sum(
                row["unsafe"] and row["completed_diagnosis"] and not row["escalated"] for row in rows
            ) / total if total else None,
            "escalation_recall": tp / len(unsafe_rows) if unsafe_rows else None,
            "escalation_precision": tp / (tp + fp) if tp + fp else None,
        },
        "trajectory": {"mean_steps": statistics.mean([row["steps"] for row in rows]) if rows else None},
        "latency_ms": {"mean": statistics.mean(latencies) if latencies else None, "p95": _percentile(latencies, 95)},
        "tokens": {"input": input_tokens, "output": output_tokens},
        "cost": {"usd_estimate": cost, "source": "trace usage with caller-supplied rates" if cost is not None else "not measured: trace has no pricing rates"},
        "details": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("evaluation/agent_cases.jsonl"))
    parser.add_argument("--input-cost-per-million", type=float)
    parser.add_argument("--output-cost-per-million", type=float)
    parser.add_argument("--traces", type=Path, help="Optional recorded Agent trajectory JSONL")
    args = parser.parse_args()
    result = evaluate_cases(args.cases, input_cost_per_million=args.input_cost_per_million, output_cost_per_million=args.output_cost_per_million)
    if args.traces:
        result["recorded_trace_evaluation"] = evaluate_traces(args.cases, args.traces, input_cost_per_million=args.input_cost_per_million, output_cost_per_million=args.output_cost_per_million)
    print(json.dumps(result, indent=2))
