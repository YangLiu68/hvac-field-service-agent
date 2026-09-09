from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class EquipmentCreate(BaseModel):
    manufacturer: str = Field(min_length=1)
    model_number: str = Field(min_length=1)
    equipment_type: str = Field(min_length=1)
    serial_number: str | None = None
    refrigerant: str | None = None
    installation_date: str | None = None


class EquipmentRead(ORMModel, EquipmentCreate):
    id: int
    created_at: datetime


class UserCreate(BaseModel):
    display_name: str = Field(min_length=1)
    email: str = Field(min_length=3)
    phone: str | None = None
    role: str = Field(pattern="^(customer|technician|dispatcher|admin)$")
    password: str | None = Field(default=None, min_length=8, exclude=True)


class UserRead(ORMModel):
    id: int
    display_name: str
    email: str
    phone: str | None = None
    role: str
    active: bool
    created_at: datetime


class LoginRequest(BaseModel):
    email: str = Field(min_length=3)
    password: str = Field(min_length=8)


class LoginResponse(BaseModel):
    access_token: str
    user: UserRead


class TechnicianProfileCreate(BaseModel):
    user_id: int
    skills: list[str] = Field(min_length=1)
    certifications: list[str] = Field(default_factory=list)
    service_postal_codes: list[str] = Field(min_length=1)
    available: bool = True
    on_call: bool = False


class TechnicianProfileRead(TechnicianProfileCreate):
    id: int
    technician_name: str
    created_at: datetime
    updated_at: datetime


class ServiceRequestCreate(BaseModel):
    customer_id: int
    equipment_id: int
    description: str = Field(min_length=8)
    service_address: str = Field(min_length=5)
    postal_code: str = Field(min_length=3, max_length=16)
    preferred_window: str | None = None
    urgency: str = Field(default="normal", pattern="^(low|normal|high|emergency)$")


class CustomerPortalRequestCreate(BaseModel):
    customer_name: str = Field(min_length=1)
    email: str = Field(min_length=3)
    phone: str | None = None
    manufacturer: str = Field(min_length=1)
    model_number: str = Field(min_length=1)
    equipment_type: str = Field(min_length=1)
    description: str = Field(min_length=8)
    service_address: str = Field(min_length=5)
    postal_code: str = Field(min_length=3, max_length=16)
    preferred_window: str | None = None
    urgency: str = Field(default="normal", pattern="^(low|normal|high|emergency)$")


class ServiceRequestRead(BaseModel):
    id: int
    customer_id: int
    customer_name: str
    equipment_id: int
    description: str
    service_address: str
    postal_code: str
    preferred_window: str | None
    urgency: str
    status: str
    job_id: int | None
    created_at: datetime
    updated_at: datetime


class TechnicianMatchRead(BaseModel):
    technician_profile_id: int
    technician_name: str
    score: float
    rationale: list[str]


class AssignmentCreate(BaseModel):
    # Kept optional for the documented API migration path; authenticated clients never send it.
    dispatcher_id: int | None = None
    technician_profile_id: int
    scheduled_for: str | None = None


class AssignmentRead(BaseModel):
    id: int
    service_request_id: int
    technician_profile_id: int
    technician_name: str
    dispatcher_id: int
    status: str
    match_score: float
    rationale: list[str]
    scheduled_for: str | None
    job_id: int | None
    created_at: datetime
    accepted_at: datetime | None


class JobCreate(BaseModel):
    customer_name: str = Field(min_length=1)
    technician_notes: str = Field(min_length=1)
    equipment_id: int | None = None
    equipment_model: str | None = None
    error_code: str | None = None

    @model_validator(mode="after")
    def require_equipment_reference(self):
        if self.equipment_id is None and not self.equipment_model:
            raise ValueError("Provide equipment_id or equipment_model")
        return self


class JobRead(ORMModel):
    id: int
    customer_name: str
    equipment_id: int | None
    equipment_model: str
    technician_notes: str
    error_code: str | None
    status: str
    created_at: datetime
    updated_at: datetime


class TechnicianTaskRead(BaseModel):
    job_id: int
    title: str
    customer_name: str
    status: str
    equipment_model: str
    error_code: str | None
    service_address: str | None
    scheduled_for: datetime
    brief_ready: bool


class JobBriefRead(BaseModel):
    id: int
    job_id: int
    summary: str
    safety_flags: list[str]
    created_at: datetime
    updated_at: datetime


class ServiceSummaryRead(BaseModel):
    id: int
    job_id: int
    content: str
    created_at: datetime


class SourceItem(BaseModel):
    document: str
    page: int
    text: str
    score: float


class DiagnosticContent(BaseModel):
    likely_cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    recommendation: str


class AnalysisResult(DiagnosticContent):
    sources: list[SourceItem]


class DiagnosticRunRead(ORMModel):
    id: int
    job_id: int
    status: str
    model_name: str | None
    likely_cause: str | None
    confidence: float | None
    recommendation: str | None
    sources: str | None
    error_message: str | None
    started_at: datetime
    completed_at: datetime | None


class TechnicianFeedbackCreate(BaseModel):
    diagnostic_run_id: int | None = None
    technician_name: str = Field(min_length=1)
    recommendation_accepted: bool
    corrected_cause: str | None = None
    action_taken: str | None = None
    notes: str | None = None


class TechnicianFeedbackRead(ORMModel, TechnicianFeedbackCreate):
    id: int
    job_id: int
    created_at: datetime


class CustomerFeedbackCreate(BaseModel):
    rating: int = Field(ge=1, le=5)
    issue_resolved: bool
    comments: str | None = None


class CustomerFeedbackRead(ORMModel, CustomerFeedbackCreate):
    id: int
    job_id: int
    created_at: datetime


class JobCloseRequest(BaseModel):
    actual_cause: str = Field(min_length=1)
    repair_action: str = Field(min_length=1)
    first_time_fix: bool
    return_visit_required: bool = False
    verified_by: str = Field(min_length=1)
    approved_for_retrieval: bool = False


class VerifiedOutcomeRead(ORMModel, JobCloseRequest):
    id: int
    job_id: int
    created_at: datetime


class TimelineEvent(BaseModel):
    event_type: str
    created_at: datetime
    data: dict


class ManualSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    manufacturer: str | None = None
    equipment_model: str | None = None
    equipment_type: str | None = None


class ManualSearchResult(BaseModel):
    id: int
    document: str
    title: str
    manufacturer: str
    page: int
    section: str | None
    text: str
    keyword_score: float
    vector_score: float
    rrf_score: float
    rerank_score: float | None = None
    score: float


class ManualIngestResponse(BaseModel):
    indexed_documents: int
    indexed_chunks: int
    skipped_documents: list[dict]
    unchanged_documents: int


class AgentRunCreate(BaseModel):
    max_steps: int = Field(default=20, ge=1, le=100)
    skill_name: str | None = None


class AgentRunRead(ORMModel):
    id: int
    job_id: int
    diagnostic_run_id: int
    status: str
    skill_name: str | None
    current_step: int
    max_steps: int
    last_response_id: str | None
    pending_input: str | None
    final_message: str | None
    error_message: str | None
    started_at: datetime
    completed_at: datetime | None


class ToolExecuteRequest(BaseModel):
    tool_name: str = Field(min_length=1)
    arguments: dict = Field(default_factory=dict)
    approved: bool = False


class ToolExecuteResponse(BaseModel):
    tool_call_id: int
    agent_step_id: int
    tool_name: str
    status: str
    duration_ms: int
    result: dict


class ToolCallRead(ORMModel):
    id: int
    diagnostic_run_id: int
    agent_run_id: int | None
    agent_step_id: int | None
    tool_name: str
    arguments: str | None
    result: str | None
    status: str
    duration_ms: int | None
    created_at: datetime


class AgentRunResult(BaseModel):
    run_id: int
    status: str
    current_step: int
    message: str | None = None
    pending_action: dict | None = None


class AgentMessageCreate(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class AgentMessageRead(ORMModel):
    id: int
    agent_run_id: int
    role: str
    content: str
    created_at: datetime


class ApprovalRequestRead(ORMModel):
    id: int
    job_id: int
    agent_run_id: int
    tool_name: str
    arguments: str
    risk_level: str
    reason: str
    status: str
    requested_by: str
    decided_by: str | None
    decision_reason: str | None
    created_at: datetime
    expires_at: datetime | None
    decided_at: datetime | None


class ApprovalDecisionCreate(BaseModel):
    decided_by: str = Field(min_length=1)
    reason: str | None = None


class ApprovalDecisionRead(BaseModel):
    approval_request_id: int
    status: str
    message: str
    agent_run_id: int
    tool_result: dict | None = None


class CaseMemorySearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


class KnowledgeEntryCreate(BaseModel):
    title: str = Field(min_length=1)
    category: str = "general"
    keywords: str = ""
    content: str = Field(min_length=1)
    source: str | None = None


class KnowledgeEntryRead(ORMModel):
    id: int
    title: str
    category: str
    keywords: str
    content: str
    source: str | None
    active: bool
    created_at: datetime
    updated_at: datetime


class UnifiedAssistantRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: int | None = None
    job_id: int | None = None
    customer_name: str | None = None
    manufacturer: str | None = None
    equipment_model: str | None = None
    equipment_type: str | None = None
    channel: str = "web"


class CaseMemorySearchResult(BaseModel):
    case_id: int
    job_id: int
    manufacturer: str | None
    equipment_model: str
    equipment_type: str | None
    error_code: str | None
    symptoms: str
    actual_cause: str
    repair_action: str
    first_time_fix: bool
    return_visit_required: bool
    score: float
    evidence_type: str


class UnifiedAssistantResponse(BaseModel):
    conversation_id: int
    job_id: int
    answer: str
    intent: str
    created_work_order: bool
    similar_cases: list[CaseMemorySearchResult]
    knowledge_entries: list[KnowledgeEntryRead]
    actions: list[str]
    selected_skill: str
    required_checks: list[str]
    retrieval_warnings: list[str]


class EstimateCreate(BaseModel):
    job_id: int
    amount: float = Field(gt=0)
    description: str = Field(min_length=1)
    customer_phone: str | None = None
    next_follow_up_at: datetime | None = None


class EstimateRead(ORMModel):
    id: int
    job_id: int
    amount: float
    description: str
    status: str
    customer_phone: str | None
    next_follow_up_at: datetime | None
    created_at: datetime


class FollowUpCreate(BaseModel):
    estimate_id: int
    content: str | None = None
    channel: str = "sms"
    scheduled_at: datetime | None = None


class FollowUpRead(ORMModel):
    id: int
    estimate_id: int
    channel: str
    content: str
    status: str
    requires_review: bool
    scheduled_at: datetime | None
    sent_at: datetime | None
    created_at: datetime


class EstimateStatusUpdate(BaseModel):
    status: str = Field(pattern="^(open|won|lost|expired)$")


class OperationsDashboard(BaseModel):
    active_jobs: int
    open_estimates: int
    open_pipeline_value: float
    won_revenue: float
    queued_follow_ups: int
    needs_review: int
    automation_rate: float


class DiagnosticEvaluationRead(ORMModel):
    id: int
    job_id: int
    diagnostic_run_id: int | None
    agent_run_id: int | None
    outcome_id: int
    top1_cause_correct: bool | None
    top3_cause_correct: bool | None
    recommendation_accepted: bool | None
    first_time_fix: bool
    return_visit_required: bool
    details: str
    evaluated_at: datetime


class SkillRead(BaseModel):
    name: str
    version: str
    description: str
    triggers: list[str]
    equipment_types: list[str]
    required_context: list[str]
    allowed_tools: list[str]
    required_checks: list[str]
    human_approval_required: list[str]
    completion_criteria: list[str]
    escalation_conditions: list[str]
    max_steps: int


class SkillRouteRead(BaseModel):
    selected_skill: str
    confidence: float
    matched_signals: list[str]
    reason: str
    allowed_tools: list[str]
    required_checks: list[str]
    human_approval_required: list[str]
    max_steps: int
