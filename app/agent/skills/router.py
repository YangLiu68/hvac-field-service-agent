import re

from .registry import SkillRegistry
from .schemas import SkillRouteResult


class SkillRouter:
    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def route(self, *, notes: str = "", error_code: str | None = None,
              equipment_type: str | None = None, equipment_model: str | None = None) -> SkillRouteResult:
        text = " ".join(filter(None, [notes, error_code, equipment_type, equipment_model])).lower()
        tokens = set(re.findall(r"[a-z0-9_/-]+", text))
        scored = []
        for skill in self.registry.list():
            matched = []
            for trigger in skill.triggers:
                trigger_tokens = set(re.findall(r"[a-z0-9_/-]+", trigger))
                if trigger in text or trigger_tokens <= tokens:
                    matched.append(trigger)
            # Equipment type is a tie-breaker only after a symptom/error signal;
            # otherwise every specialized skill would match common air-handler types.
            type_match = bool(matched and equipment_type and skill.equipment_types and equipment_type.lower() in skill.equipment_types)
            if type_match:
                matched.append(f"equipment_type:{equipment_type.lower()}")
            exact_error = bool(error_code and error_code.lower() in skill.triggers)
            score = len(matched) + (3 if exact_error else 0) + (1 if type_match else 0)
            scored.append((score, len(matched), skill, matched))
        scored.sort(key=lambda item: (item[0], item[1], item[2].name), reverse=True)
        score, _, skill, matched = scored[0]
        if score == 0:
            skill = self.registry.get("general_hvac_triage")
            matched = ["fallback:no_matching_signal"]
        # Confidence is intentionally conservative and reflects routing evidence, not diagnosis certainty.
        confidence = min(0.99, 0.45 + 0.12 * len(matched) + (0.20 if score >= 3 else 0.0))
        return SkillRouteResult(
            selected_skill=skill.name,
            confidence=round(confidence, 2),
            matched_signals=matched,
            reason=f"Selected {skill.name} from {', '.join(matched)}.",
            allowed_tools=skill.allowed_tools,
            required_checks=skill.required_checks,
            human_approval_required=skill.human_approval_required,
            max_steps=skill.max_steps,
        )
