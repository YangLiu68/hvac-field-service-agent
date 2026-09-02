from typing import Any


HIGH_RISK_TOOLS = {
    "perform_refrigerant_service",
    "recover_or_charge_refrigerant",
    "access_electrical_panel",
    "measure_live_voltage",
    "replace_component",
    "bypass_safety_switch",
}

HIGH_RISK_MEASUREMENT_TERMS = {
    "live_voltage", "voltage_live", "electrical_panel", "refrigerant_pressure",
    "suction_pressure", "discharge_pressure", "superheat", "subcooling",
}


def approval_requirement(tool_name: str, arguments: dict[str, Any]) -> tuple[str, str] | None:
    """Return (risk, reason) for actions requiring qualified human approval."""
    normalized = tool_name.lower()
    if normalized in HIGH_RISK_TOOLS:
        if normalized == "bypass_safety_switch":
            return "critical", "Bypassing a safety switch is prohibited and requires supervisor review."
        return "high", f"{tool_name} can expose a technician to electrical or refrigerant hazards."
    measurement = str(arguments.get("measurement_type", "")).lower().replace("-", "_").replace(" ", "_")
    for term in HIGH_RISK_MEASUREMENT_TERMS:
        if term in measurement:
            return "high", f"The requested measurement ({measurement}) may involve hazardous HVAC equipment access."
    return None
