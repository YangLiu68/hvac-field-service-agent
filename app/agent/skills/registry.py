from pathlib import Path

import yaml

from app.agent.tool_registry import ToolRegistry

from .schemas import SkillDefinition


SKILL_DIR = Path(__file__).resolve().parent / "definitions"


class SkillRegistry:
    def __init__(self, skills: list[SkillDefinition]):
        self._skills = {skill.name: skill for skill in skills}

    @classmethod
    def from_directory(cls, directory: Path = SKILL_DIR, tool_registry: ToolRegistry | None = None):
        skills = []
        for path in sorted(directory.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError(f"Skill file must contain a mapping: {path.name}")
            try:
                skill = SkillDefinition.model_validate(raw)
            except Exception as exc:
                raise ValueError(f"Invalid skill {path.name}: {exc}") from exc
            if tool_registry is not None:
                unknown = sorted(set(skill.allowed_tools) - set(tool_registry.names()))
                if unknown:
                    raise ValueError(f"Skill {skill.name} references unknown tools: {unknown}")
            skills.append(skill)
        if not skills:
            raise ValueError(f"No skill definitions found in {directory}")
        return cls(skills)

    def get(self, name: str) -> SkillDefinition:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise KeyError(f"Unknown skill: {name}") from exc

    def list(self) -> list[SkillDefinition]:
        return [self._skills[name] for name in sorted(self._skills)]

    def public(self, name: str) -> dict:
        return self.get(name).model_dump(mode="json")


def build_default_skill_registry(tool_registry=None) -> SkillRegistry:
    return SkillRegistry.from_directory(SKILL_DIR, tool_registry=tool_registry)
