from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable
    allowed_job_statuses: frozenset[str] | None = None
    requires_approval: bool = False
    timeout_seconds: int = 30

    def public_schema(self) -> dict:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.input_model.model_json_schema(),
            "strict": False,
            "metadata": {
                "requires_approval": self.requires_approval,
                "allowed_job_statuses": sorted(self.allowed_job_statuses or []),
                "timeout_seconds": self.timeout_seconds,
            },
        }


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._tools:
            raise ValueError(f"Tool already registered: {definition.name}")
        self._tools[definition.name] = definition

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def schemas(self) -> list[dict]:
        return [self._tools[name].public_schema() for name in sorted(self._tools)]

    def names(self) -> list[str]:
        return sorted(self._tools)


def build_default_registry() -> ToolRegistry:
    from app.agent.tools import TOOL_DEFINITIONS

    registry = ToolRegistry()
    for definition in TOOL_DEFINITIONS:
        registry.register(definition)
    return registry
