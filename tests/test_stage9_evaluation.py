import json
from pathlib import Path

from evaluation.agent_eval import evaluate_cases, evaluate_traces


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "evaluation" / "agent_cases.jsonl"


def test_stage9_benchmark_has_60_cases_and_reports_all_dimensions():
    result = evaluate_cases(CASES)
    assert result["cases"] == 60
    assert result["coverage"]["valid"] is True
    assert result["skill_routing"]["accuracy"] == 1.0
    assert result["tool_selection"]["accuracy"] == 1.0
    assert result["diagnosis"]["top3_accuracy"] >= result["diagnosis"]["top1_accuracy"]
    assert result["safety"]["unsafe_recommendation_rate"] == 0.0
    assert result["latency_ms"]["p95"] is not None
    assert result["tokens"]["estimated_input"] > 0
    assert result["cost"]["usd_estimate"] is None


def test_recorded_trace_evaluator_scores_usage_and_cost(tmp_path):
    trace_path = tmp_path / "traces.jsonl"
    cases = [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]
    first = cases[0]
    trace_path.write_text(json.dumps({
        "case_id": first["id"],
        "selected_skill": first["expected_skill"],
        "tool_calls": [{"name": name} for name in ["get_job_context", "get_equipment_profile", "search_error_codes", "search_service_manuals", "search_verified_repairs", "request_technician_measurement"]],
        "predicted_causes": ["clogged return air filter"],
        "escalated": False,
        "latency_ms": 120,
        "usage": {"input_tokens": 1000, "output_tokens": 200},
    }) + "\n")
    result = evaluate_traces(CASES, trace_path, input_cost_per_million=1.0, output_cost_per_million=2.0)
    assert result["cases"] == 1
    assert result["skill_routing_accuracy"] == 1.0
    assert result["tool_selection_accuracy"] == 1.0
    assert result["diagnosis_top1_accuracy"] == 1.0
    assert result["cost"]["usd_estimate"] == 0.0014
