from pydantic import BaseModel, Field, model_validator


class SkillDefinition(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: str = "1.0.0"
    description: str = Field(min_length=1)
    triggers: list[str] = Field(min_length=1)
    equipment_types: list[str] = Field(default_factory=list)
    required_context: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(min_length=1)
    required_checks: list[str] = Field(default_factory=list)
    human_approval_required: list[str] = Field(default_factory=list)
    completion_criteria: list[str] = Field(min_length=1)
    escalation_conditions: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def normalize_lists(self):
        self.triggers = [item.strip().lower() for item in self.triggers if item.strip()]
        self.equipment_types = [item.strip().lower() for item in self.equipment_types if item.strip()]
        self.allowed_tools = list(dict.fromkeys(self.allowed_tools))
        return self


class SkillRouteRequest(BaseModel):
    notes: str | None = None
    error_code: str | None = None
    equipment_type: str | None = None
    equipment_model: str | None = None


class SkillRouteResult(BaseModel):
    selected_skill: str
    confidence: float = Field(ge=0.0, le=1.0)
    matched_signals: list[str]
    reason: str
    allowed_tools: list[str]
    required_checks: list[str]
    human_approval_required: list[str]
    max_steps: int
