"""Run the Stage 9 labelled cases against the live Agent API.

This is a benchmark harness, not a safety override: it never approves a
protected operation.  Runs which request approval are retained as-is in the
trace so the evaluator can score the model's actual behaviour.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_json(value: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def fixture_for(case: dict[str, Any], measurement_type: str) -> tuple[str, str | None]:
    """Deterministic synthetic technician response for repeatable comparisons."""
    name = measurement_type.lower().replace("-", "_").replace(" ", "_")
    expected = " ".join(case.get("expected_causes", [])).lower()
    if "thermostat" in name and ("call" in name or "operating_mode" in name):
        return "calling", None
    if "filter" in name:
        return ("dirty and clogged" if "filter" in expected else "clean"), None
    if "airflow" in name or "static_pressure" in name:
        restricted = "airflow" in expected or "filter" in expected or "airflow" in case.get("notes", "").lower()
        return ("low airflow" if restricted else "normal airflow"), None
    if any(term in name for term in ("temperature", "temp", "delta")):
        return "18", "F"
    if "outdoor" in name or "compressor" in name or "blower" in name:
        return "running", None
    if "voltage" in name or "pressure" in name or "superheat" in name or "subcooling" in name:
        # These are normally protected by the server's approval policy.  If a
        # provider asks for one without triggering that policy, the fixture is
        # deliberately non-actionable rather than inventing hazardous readings.
        return "not measured", None
    return "confirmed", None


def latest_measurement_request(calls: list[dict[str, Any]]) -> tuple[int, str]:
    for call in reversed(calls):
        if call.get("tool_name") != "request_technician_measurement":
            continue
        result = parse_json(call.get("result"))
        arguments = parse_json(call.get("arguments"))
        request_id = result.get("request_id")
        if request_id is not None:
            return int(request_id), str(arguments.get("measurement_type", "unknown"))
    raise RuntimeError("Run paused for a technician but no measurement request was found")


def predicted_causes(calls: list[dict[str, Any]]) -> list[str]:
    completed: list[str] = []
    hypotheses: list[str] = []
    for call in calls:
        arguments = parse_json(call.get("arguments"))
        if call.get("tool_name") == "complete_diagnosis" and arguments.get("likely_cause"):
            completed.append(str(arguments["likely_cause"]))
        elif call.get("tool_name") == "create_diagnostic_hypothesis" and arguments.get("cause"):
            hypotheses.append(str(arguments["cause"]))
    return (completed + hypotheses)[:3]


def usage_from_timeline(timeline: list[dict[str, Any]]) -> dict[str, int]:
    usage_events = [event.get("data", {}) for event in timeline if event.get("event_type") == "agent_llm_usage"]
    return {
        "input_tokens": sum(int(event.get("input_tokens") or 0) for event in usage_events),
        "output_tokens": sum(int(event.get("output_tokens") or 0) for event in usage_events),
    }


def request(client: httpx.Client, method: str, url: str, **kwargs) -> dict[str, Any]:
    response = client.request(method, url, **kwargs)
    response.raise_for_status()
    return response.json()


def run_case(client: httpx.Client, case: dict[str, Any], max_steps: int, max_turns: int) -> dict[str, Any]:
    started = time.perf_counter()
    equipment = request(client, "POST", "/equipment", json={
        "manufacturer": "Stage9 Evaluation",
        "model_number": case.get("equipment_model") or "UNKNOWN",
        "equipment_type": case.get("equipment_type") or "unknown",
    })
    job = request(client, "POST", "/jobs", json={
        "customer_name": f"Stage9 {case['id']}",
        "equipment_id": equipment["id"],
        "technician_notes": case["notes"],
        "error_code": case.get("error_code"),
    })
    run = request(client, "POST", f"/jobs/{job['id']}/agent-runs", json={"max_steps": max_steps})
    run_result = request(client, "POST", f"/agent-runs/{run['id']}/start")
    turns = 1
    while run_result["status"] == "waiting_for_technician" and turns < max_turns:
        calls = request(client, "GET", f"/agent-runs/{run['id']}/tool-calls")
        request_id, measurement_type = latest_measurement_request(calls)
        value, unit = fixture_for(case, measurement_type)
        arguments: dict[str, Any] = {
            "request_id": request_id,
            "value": value,
            "recorded_by": "Stage9 benchmark fixture",
            "notes": f"Deterministic synthetic fixture for {measurement_type}.",
        }
        if unit:
            arguments["unit"] = unit
        request(client, "POST", f"/agent-runs/{run['id']}/tools", json={
            "tool_name": "record_measurement", "arguments": arguments,
        })
        run_result = request(client, "POST", f"/agent-runs/{run['id']}/continue")
        turns += 1
    if run_result["status"] == "waiting_for_technician":
        run_result["status"] = "failed"
        run_result["message"] = f"Harness stopped after {max_turns} model turns"

    calls = request(client, "GET", f"/agent-runs/{run['id']}/tool-calls")
    timeline = request(client, "GET", f"/jobs/{job['id']}/timeline")
    return {
        "case_id": case["id"],
        "selected_skill": run.get("skill_name"),
        # record_measurement is an evaluator/technician observation, not an
        # LLM-selected action, so exclude it from the model tool trajectory.
        "tool_calls": [
            {"name": call["tool_name"]}
            for call in calls
            if call["tool_name"] != "record_measurement"
        ],
        "predicted_causes": predicted_causes(calls),
        "escalated": run_result["status"] == "escalated",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "usage": usage_from_timeline(timeline),
        "run_status": run_result["status"],
        "run_id": run["id"],
        "job_id": job["id"],
        "message": run_result.get("message"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real Stage 9 Agent traces against a running API server.")
    parser.add_argument("--base-url", default=os.getenv("STAGE9_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--cases", type=Path, default=Path("evaluation/agent_cases.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/real_traces.jsonl"))
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--limit", type=int, help="Run only the first N cases (useful for a smoke test).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error(f"{args.output} exists; choose another output path or pass --overwrite")
    cases = read_jsonl(args.cases)
    if args.limit:
        cases = cases[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=120.0) as client, args.output.open("w", encoding="utf-8") as handle:
        for number, case in enumerate(cases, start=1):
            try:
                trace = run_case(client, case, args.max_steps, args.max_turns)
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                trace = {"case_id": case["id"], "error": str(exc), "tool_calls": [], "predicted_causes": [], "escalated": False, "usage": {"input_tokens": 0, "output_tokens": 0}}
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{number}/{len(cases)}] {case['id']}: {trace.get('run_status', 'error')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
