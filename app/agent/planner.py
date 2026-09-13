"""Deterministic diagnostic state and next-best-action policy.

The LLM is useful for language understanding, but it should not decide whether
to skip cheap, safe checks.  This module keeps the first-pass diagnostic graph
small and explicit; the orchestrator persists its state on ``AgentRun`` and
lets the model take over only after the field observation is recorded.
"""

from __future__ import annotations

import re
from typing import Any


NO_COOLING_CHECKS = [
    "thermostat_call",
    "indoor_blower",
    "outdoor_unit_operation",
    "filter_airflow",
    "supply_return_temperature",
    "evaporator_ice",
]

# Explicit graph edges used by the heuristic planner.  Each edge represents a
# safe observation before the branch that needs specialized measurements.
DIAGNOSTIC_GRAPH = {
    "warm_air": {"next": "thermostat_call"},
    "thermostat_call": {"yes": "indoor_blower", "no": "low_voltage_control"},
    "indoor_blower": {"yes": "outdoor_unit_operation", "no": "blower_or_power"},
    "outdoor_unit_operation": {"fan_only": "compressor_start_circuit", "off": "power_or_contactor", "yes": "filter_airflow"},
    "filter_airflow": {"normal": "supply_return_temperature", "restricted": "airflow_restriction"},
    "supply_return_temperature": {"low_delta": "evaporator_ice", "normal": "refrigerant_circuit_measurements"},
    "evaporator_ice": {"yes": "defrost_or_airflow", "no": "refrigerant_circuit_measurements"},
}

CHECK_LABELS = {
    "thermostat_call": "thermostat cooling call",
    "indoor_blower": "indoor blower operation",
    "outdoor_unit_operation": "outdoor condenser operation",
    "filter_airflow": "filter and airflow condition",
    "supply_return_temperature": "supply and return air temperatures",
    "evaporator_ice": "evaporator coil icing",
    "refrigerant_circuit_measurements": "refrigerant-circuit readings",
}


def new_state(skill_name: str = "general_hvac_triage") -> dict[str, Any]:
    checks = {name: {"status": "unknown", "evidence": []} for name in NO_COOLING_CHECKS}
    return {
        "skill": skill_name,
        "graph": "no_cooling_v1" if skill_name == "no_cooling" else "generic_triage_v1",
        "checks": checks,
        "observations": [],
        "measurements": [],
        "hypotheses": [],
        "retrieval": {"job_context": False, "equipment": False, "history": False, "cases": False, "manuals": False},
        "next_action": None,
        "confidence": "low",
    }


def _set_check(state: dict[str, Any], name: str, status: str, evidence: str) -> None:
    item = state.setdefault("checks", {}).setdefault(name, {"status": "unknown", "evidence": []})
    item["status"] = status
    if evidence and evidence not in item.setdefault("evidence", []):
        item["evidence"].append(evidence[:500])


def update_from_text(state: dict[str, Any], text: str) -> dict[str, Any]:
    """Extract conservative observations from technician language.

    This is intentionally a parser, not a diagnosis.  Ambiguous language stays
    ``unknown`` so the planner asks for one observation instead of guessing.
    """
    raw = (text or "").strip()
    lower = raw.lower()
    if not raw:
        return state
    state.setdefault("observations", []).append(raw[:1000])

    if re.search(r"thermostat.*(call|calling|cooling)|calling.*cool", lower):
        _set_check(state, "thermostat_call", "confirmed", raw)
    elif re.search(r"thermostat.*(not|no).*(call|cool)|not calling for cool", lower):
        _set_check(state, "thermostat_call", "negative", raw)

    if re.search(r"(indoor|blower|air handler).*\b(run|running|on)\b|\bblower\s+is\s+running", lower):
        _set_check(state, "indoor_blower", "confirmed", raw)
    elif re.search(r"(indoor|blower|air handler).*\b(off|stopped|not running)\b", lower):
        _set_check(state, "indoor_blower", "negative", raw)

    outdoor_running = re.search(r"outdoor.*(fan|condenser).*(run|running|on)|compressor.*\b(run|running|on)", lower)
    outdoor_stopped = re.search(r"outdoor.*(fan|condenser).*(off|stopped|not running)|compressor.*\b(off|stopped|not running)", lower)
    if outdoor_running and not outdoor_stopped:
        _set_check(state, "outdoor_unit_operation", "confirmed", raw)
    elif outdoor_stopped:
        _set_check(state, "outdoor_unit_operation", "negative", raw)

    if re.search(r"filter.*\b(clean|new|clear)\b|airflow.*(normal|not restricted|unrestricted)", lower):
        _set_check(state, "filter_airflow", "confirmed", raw)
    elif re.search(r"filter.*(dirty|clog|blocked)|airflow.*(restricted|weak|low)", lower):
        _set_check(state, "filter_airflow", "negative", raw)

    supply = re.search(r"supply(?:\s+air)?(?:\s+temperature)?\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)", lower)
    ret = re.search(r"return(?:\s+air)?(?:\s+temperature)?\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)", lower)
    if supply and ret:
        state["temperature"] = {"supply": supply.group(1), "return": ret.group(1), "unit": "F"}
        _set_check(state, "supply_return_temperature", "confirmed", raw)

    if re.search(r"(evaporator|coil).*(frozen|ice|icing)|ice.*(evaporator|coil)", lower):
        _set_check(state, "evaporator_ice", "confirmed", raw)
    elif re.search(r"(evaporator|coil).*(not frozen|no ice|clear)", lower):
        _set_check(state, "evaporator_ice", "negative", raw)

    return state


def record_measurement(state: dict[str, Any], measurement_type: str, value: str, unit: str | None = None) -> None:
    state.setdefault("measurements", []).append({"type": measurement_type, "value": value, "unit": unit})
    if measurement_type in state.setdefault("checks", {}):
        _set_check(state, measurement_type, "confirmed", f"Recorded: {value}{(' ' + unit) if unit else ''}")


def choose_next_action(state: dict[str, Any], skill_name: str) -> dict[str, Any] | None:
    if skill_name == "no_cooling":
        for index, check in enumerate(NO_COOLING_CHECKS):
            item = state.get("checks", {}).get(check, {})
            if item.get("status") == "unknown":
                instructions = {
                    "thermostat_call": "With the thermostat set below room temperature, confirm it is in COOL mode and calling for cooling.",
                    "indoor_blower": "With cooling requested, confirm the indoor blower is running.",
                    "outdoor_unit_operation": "With cooling requested, observe the outdoor unit: is the condenser fan running and is the compressor operating?",
                    "filter_airflow": "Inspect the return filter and confirm whether it is clean and airflow is unrestricted.",
                    "supply_return_temperature": "Measure supply and return air temperatures after the system has run for several minutes.",
                    "evaporator_ice": "Visually check the accessible evaporator coil and suction line for ice; do not open energized panels.",
                }
                return {
                    "measurement_type": check,
                    "instructions": instructions[check],
                    "unit": "F" if check == "supply_return_temperature" else None,
                    "safety_note": "Use normal field precautions; do not open energized compartments unless qualified.",
                    "score": round(0.98 - index * 0.06, 2),
                    "reason": "safe, low-cost check with high information gain",
                }
        # Once the protected refrigerant-circuit request has been completed,
        # hand the evidence to the LLM/recovery path for interpretation rather
        # than issuing the same request again.
        if any(item.get("type") == "refrigerant_circuit_measurements" for item in state.get("measurements", [])):
            return None
        return {
            "measurement_type": "refrigerant_circuit_measurements",
            "instructions": "Only a qualified technician should measure suction/liquid pressure, superheat, subcooling, and compressor current using the manufacturer procedure.",
            "unit": None,
            "safety_note": "Refrigerant and energized electrical measurements require appropriate qualification and PPE.",
            "score": 0.42,
            "reason": "basic checks are complete; quantify the remaining cooling-capacity hypothesis",
        }
    return {
        "measurement_type": "diagnostic_observation",
        "instructions": "Record the next observable symptom or manufacturer-recommended measurement for this equipment.",
        "unit": None,
        "safety_note": "Do not access energized compartments unless qualified.",
        "score": 0.7,
        "reason": "insufficient evidence for a specialized branch",
    }


def state_summary(state: dict[str, Any]) -> str:
    checks = state.get("checks", {})
    completed = [CHECK_LABELS.get(name, name) for name, item in checks.items() if item.get("status") != "unknown"]
    pending = [CHECK_LABELS.get(name, name) for name, item in checks.items() if item.get("status") == "unknown"]
    return f"Completed: {', '.join(completed) or 'none'}. Pending: {', '.join(pending) or 'none'}. Confidence: {state.get('confidence', 'low')}."
