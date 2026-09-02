from pydantic import BaseModel, Field


class EmptyInput(BaseModel):
    pass


class EquipmentProfileOutput(BaseModel):
    equipment_id: int | None
    manufacturer: str | None
    model_number: str
    equipment_type: str | None
    serial_number: str | None
    refrigerant: str | None
    installation_date: str | None


class JobContextOutput(BaseModel):
    job_id: int
    customer_name: str
    status: str
    technician_notes: str
    error_code: str | None
    equipment: EquipmentProfileOutput
    diagnostic_run_count: int
    pending_measurements: int


class ManualSearchInput(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class EvidenceItem(BaseModel):
    document: str
    page: int
    section: str | None = None
    text: str
    score: float


class ManualSearchOutput(BaseModel):
    results: list[EvidenceItem]


class ErrorCodeInput(BaseModel):
    error_code: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class EquipmentHistoryOutput(BaseModel):
    jobs: list[dict]


class VerifiedRepairSearchInput(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


class VerifiedRepairSearchOutput(BaseModel):
    cases: list[dict]


class MeasurementRequestInput(BaseModel):
    measurement_type: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    unit: str | None = None
    safety_note: str | None = None


class MeasurementRequestOutput(BaseModel):
    request_id: int
    status: str


class MeasurementResultInput(BaseModel):
    request_id: int
    value: str = Field(min_length=1)
    unit: str | None = None
    recorded_by: str = Field(min_length=1)
    notes: str | None = None


class MeasurementResultOutput(BaseModel):
    result_id: int
    request_id: int
    status: str


class HypothesisInput(BaseModel):
    cause: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_evidence: list[EvidenceItem] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    next_test: str | None = None
    selected: bool = False


class HypothesisOutput(BaseModel):
    hypothesis_id: int
    rank: int


class JobStatusInput(BaseModel):
    status: str
    reason: str = Field(min_length=1)


class JobStatusOutput(BaseModel):
    previous_status: str
    status: str


class EscalationInput(BaseModel):
    reason: str = Field(min_length=1)
    priority: str = Field(default="normal", pattern="^(low|normal|high|critical)$")


class EscalationOutput(BaseModel):
    status: str
    priority: str


class CompleteDiagnosisInput(BaseModel):
    likely_cause: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    recommendation: str = Field(min_length=1)


class CompleteDiagnosisOutput(BaseModel):
    diagnostic_run_id: int
    job_status: str
