from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from .database import Base


def utcnow():
    return datetime.now(timezone.utc)


class Equipment(Base):
    __tablename__ = "equipment"

    id = Column(Integer, primary_key=True, index=True)
    manufacturer = Column(String, nullable=False)
    model_number = Column(String, nullable=False, index=True)
    equipment_type = Column(String, nullable=False)
    serial_number = Column(String, unique=True, nullable=True)
    refrigerant = Column(String, nullable=True)
    installation_date = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    jobs = relationship("Job", back_populates="equipment")
    service_requests = relationship("ServiceRequest", back_populates="equipment")


class User(Base):
    """Minimal role model for customer, technician, dispatcher and admin workflows."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    display_name = Column(String, nullable=False)
    email = Column(String, nullable=False, unique=True, index=True)
    phone = Column(String, nullable=True)
    role = Column(String, nullable=False, index=True)
    # Empty only for legacy/guest customer records created before login was added.
    password_hash = Column(String, nullable=False, default="")
    active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    technician_profile = relationship("TechnicianProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    customer_requests = relationship("ServiceRequest", back_populates="customer", foreign_keys="ServiceRequest.customer_id")


class TechnicianProfile(Base):
    __tablename__ = "technician_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True, index=True)
    skills = Column(Text, nullable=False, default="[]")
    certifications = Column(Text, nullable=False, default="[]")
    service_postal_codes = Column(Text, nullable=False, default="[]")
    available = Column(Boolean, default=True, nullable=False, index=True)
    on_call = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    user = relationship("User", back_populates="technician_profile")
    assignments = relationship("Assignment", back_populates="technician")


class ServiceRequest(Base):
    __tablename__ = "service_requests"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    equipment_id = Column(Integer, ForeignKey("equipment.id"), nullable=False, index=True)
    description = Column(Text, nullable=False)
    service_address = Column(Text, nullable=False)
    postal_code = Column(String, nullable=False, index=True)
    preferred_window = Column(String, nullable=True)
    urgency = Column(String, default="normal", nullable=False, index=True)
    status = Column(String, default="submitted", nullable=False, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=True, unique=True, index=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    customer = relationship("User", back_populates="customer_requests", foreign_keys=[customer_id])
    equipment = relationship("Equipment", back_populates="service_requests")
    job = relationship("Job", foreign_keys=[job_id])
    assignments = relationship("Assignment", back_populates="service_request", cascade="all, delete-orphan")


class Assignment(Base):
    __tablename__ = "assignments"

    id = Column(Integer, primary_key=True, index=True)
    service_request_id = Column(Integer, ForeignKey("service_requests.id"), nullable=False, index=True)
    technician_id = Column(Integer, ForeignKey("technician_profiles.id"), nullable=False, index=True)
    dispatcher_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    status = Column(String, default="assigned", nullable=False, index=True)
    match_score = Column(Float, nullable=False)
    rationale = Column(Text, nullable=False)
    scheduled_for = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    accepted_at = Column(DateTime(timezone=True), nullable=True)

    service_request = relationship("ServiceRequest", back_populates="assignments")
    technician = relationship("TechnicianProfile", back_populates="assignments")
    dispatcher = relationship("User", foreign_keys=[dispatcher_id])


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    equipment_id = Column(Integer, ForeignKey("equipment.id"), nullable=True, index=True)
    customer_name = Column(String, nullable=False)
    equipment_model = Column(String, nullable=False)  # retained for Stage 1 compatibility
    technician_notes = Column(Text, nullable=False)
    error_code = Column(String, nullable=True)
    status = Column(String, default="open", nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    equipment = relationship("Equipment", back_populates="jobs")
    diagnostic_runs = relationship("DiagnosticRun", back_populates="job", cascade="all, delete-orphan")
    technician_feedback = relationship("TechnicianFeedback", back_populates="job", cascade="all, delete-orphan")
    customer_feedback = relationship("CustomerFeedback", back_populates="job", cascade="all, delete-orphan")
    outcomes = relationship("VerifiedOutcome", back_populates="job", cascade="all, delete-orphan")
    events = relationship("JobEvent", back_populates="job", cascade="all, delete-orphan")
    agent_runs = relationship("AgentRun", back_populates="job", cascade="all, delete-orphan")
    measurement_requests = relationship("MeasurementRequest", back_populates="job", cascade="all, delete-orphan")
    case_memories = relationship("CaseMemory", back_populates="job", cascade="all, delete-orphan")
    diagnostic_evaluations = relationship("DiagnosticEvaluation", back_populates="job", cascade="all, delete-orphan")
    brief = relationship("JobBrief", back_populates="job", uselist=False, cascade="all, delete-orphan")
    summaries = relationship("ServiceSummary", back_populates="job", cascade="all, delete-orphan")
    assistant_conversations = relationship("AssistantConversation", back_populates="job", cascade="all, delete-orphan")
    estimates = relationship("Estimate", back_populates="job", cascade="all, delete-orphan")


class KnowledgeEntry(Base):
    """Curated reusable domain knowledge separate from verified repair cases."""

    __tablename__ = "knowledge_entries"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    category = Column(String, nullable=False, default="general", index=True)
    keywords = Column(Text, nullable=False, default="")
    content = Column(Text, nullable=False)
    source = Column(String, nullable=True)
    active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)


class AssistantConversation(Base):
    """Durable natural-language session that may create and update a work order."""

    __tablename__ = "assistant_conversations"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False, index=True)
    channel = Column(String, nullable=False, default="web", index=True)
    status = Column(String, nullable=False, default="active", index=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    job = relationship("Job", back_populates="assistant_conversations")
    turns = relationship("AssistantTurn", back_populates="conversation", cascade="all, delete-orphan")


class AssistantTurn(Base):
    __tablename__ = "assistant_turns"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("assistant_conversations.id"), nullable=False, index=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    intent = Column(String, nullable=True, index=True)
    evidence = Column(Text, nullable=False, default="[]")
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    conversation = relationship("AssistantConversation", back_populates="turns")


class AssistantCheckState(Base):
    """Structured progress for a skill check within one conversation."""

    __tablename__ = "assistant_check_states"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("assistant_conversations.id"), nullable=False, index=True)
    check_name = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False, default="pending", index=True)
    evidence = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)


class Estimate(Base):
    __tablename__ = "estimates"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String, nullable=False, default="open", index=True)
    customer_phone = Column(String, nullable=True)
    next_follow_up_at = Column(DateTime(timezone=True), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    job = relationship("Job", back_populates="estimates")
    follow_ups = relationship("FollowUp", back_populates="estimate", cascade="all, delete-orphan")


class FollowUp(Base):
    __tablename__ = "follow_ups"

    id = Column(Integer, primary_key=True, index=True)
    estimate_id = Column(Integer, ForeignKey("estimates.id"), nullable=False, index=True)
    channel = Column(String, nullable=False, default="sms")
    content = Column(Text, nullable=False)
    status = Column(String, nullable=False, default="draft", index=True)
    requires_review = Column(Boolean, nullable=False, default=False, index=True)
    scheduled_at = Column(DateTime(timezone=True), nullable=True)
    sent_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    estimate = relationship("Estimate", back_populates="follow_ups")


class AIAnalysis(Base):
    """Legacy Stage 1 table retained for backward compatibility."""

    __tablename__ = "ai_analyses"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    likely_cause = Column(String)
    confidence = Column(Float)
    recommendation = Column(Text)
    sources = Column(Text)


class DiagnosticRun(Base):
    __tablename__ = "diagnostic_runs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False, index=True)
    status = Column(String, default="running", nullable=False)
    model_name = Column(String, nullable=True)
    likely_cause = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    recommendation = Column(Text, nullable=True)
    sources = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    job = relationship("Job", back_populates="diagnostic_runs")
    hypotheses = relationship("DiagnosticHypothesis", back_populates="diagnostic_run", cascade="all, delete-orphan")
    tool_calls = relationship("ToolCall", back_populates="diagnostic_run", cascade="all, delete-orphan")
    agent_run = relationship("AgentRun", back_populates="diagnostic_run", uselist=False)


class DiagnosticHypothesis(Base):
    __tablename__ = "diagnostic_hypotheses"

    id = Column(Integer, primary_key=True, index=True)
    diagnostic_run_id = Column(Integer, ForeignKey("diagnostic_runs.id"), nullable=False, index=True)
    rank = Column(Integer, default=1, nullable=False)
    cause = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)
    supporting_evidence = Column(Text, nullable=True)
    contradicting_evidence = Column(Text, nullable=True)
    next_test = Column(Text, nullable=True)
    selected = Column(Boolean, default=False, nullable=False)

    diagnostic_run = relationship("DiagnosticRun", back_populates="hypotheses")


class ToolCall(Base):
    __tablename__ = "tool_calls"

    id = Column(Integer, primary_key=True, index=True)
    diagnostic_run_id = Column(Integer, ForeignKey("diagnostic_runs.id"), nullable=False, index=True)
    agent_run_id = Column(Integer, ForeignKey("agent_runs.id"), nullable=True, index=True)
    agent_step_id = Column(Integer, ForeignKey("agent_steps.id"), nullable=True, index=True)
    tool_name = Column(String, nullable=False)
    arguments = Column(Text, nullable=True)
    result = Column(Text, nullable=True)
    status = Column(String, default="completed", nullable=False)
    duration_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    diagnostic_run = relationship("DiagnosticRun", back_populates="tool_calls")
    agent_run = relationship("AgentRun", back_populates="tool_calls")
    agent_step = relationship("AgentStep", back_populates="tool_call")


class AgentRun(Base):
    """Persistent execution context; Stage 4 uses it for audited manual tool runs."""

    __tablename__ = "agent_runs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False, index=True)
    diagnostic_run_id = Column(Integer, ForeignKey("diagnostic_runs.id"), nullable=False, unique=True)
    status = Column(String, default="tool_testing", nullable=False, index=True)
    skill_name = Column(String, nullable=True, index=True)
    last_response_id = Column(String, nullable=True)
    pending_input = Column(Text, nullable=True)
    final_message = Column(Text, nullable=True)
    pending_approval_call = Column(Text, nullable=True)
    current_step = Column(Integer, default=0, nullable=False)
    max_steps = Column(Integer, default=20, nullable=False)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    job = relationship("Job", back_populates="agent_runs")
    diagnostic_run = relationship("DiagnosticRun", back_populates="agent_run")
    steps = relationship("AgentStep", back_populates="agent_run", cascade="all, delete-orphan")
    tool_calls = relationship("ToolCall", back_populates="agent_run")
    approval_requests = relationship("ApprovalRequest", back_populates="agent_run", cascade="all, delete-orphan")
    messages = relationship("AgentMessage", back_populates="agent_run", cascade="all, delete-orphan")


class AgentMessage(Base):
    __tablename__ = "agent_messages"

    id = Column(Integer, primary_key=True, index=True)
    agent_run_id = Column(Integer, ForeignKey("agent_runs.id"), nullable=False, index=True)
    role = Column(String, nullable=False)  # technician, assistant, system
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    agent_run = relationship("AgentRun", back_populates="messages")


class JobBrief(Base):
    __tablename__ = "job_briefs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False, unique=True, index=True)
    summary = Column(Text, nullable=False)
    safety_flags = Column(Text, nullable=False, default="[]")
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    job = relationship("Job", back_populates="brief")


class ServiceSummary(Base):
    __tablename__ = "service_summaries"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    job = relationship("Job", back_populates="summaries")


class AgentStep(Base):
    __tablename__ = "agent_steps"

    id = Column(Integer, primary_key=True, index=True)
    agent_run_id = Column(Integer, ForeignKey("agent_runs.id"), nullable=False, index=True)
    step_number = Column(Integer, nullable=False)
    step_type = Column(String, default="tool_call", nullable=False)
    status = Column(String, default="running", nullable=False)
    input_data = Column(Text, nullable=True)
    output_data = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    agent_run = relationship("AgentRun", back_populates="steps")
    tool_call = relationship("ToolCall", back_populates="agent_step", uselist=False)


class MeasurementRequest(Base):
    __tablename__ = "measurement_requests"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False, index=True)
    agent_run_id = Column(Integer, ForeignKey("agent_runs.id"), nullable=True, index=True)
    measurement_type = Column(String, nullable=False)
    instructions = Column(Text, nullable=False)
    unit = Column(String, nullable=True)
    safety_note = Column(Text, nullable=True)
    status = Column(String, default="pending", nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    job = relationship("Job", back_populates="measurement_requests")
    results = relationship("MeasurementResult", back_populates="request", cascade="all, delete-orphan")


class MeasurementResult(Base):
    __tablename__ = "measurement_results"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(Integer, ForeignKey("measurement_requests.id"), nullable=False, index=True)
    value = Column(String, nullable=False)
    unit = Column(String, nullable=True)
    recorded_by = Column(String, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    request = relationship("MeasurementRequest", back_populates="results")


class ApprovalRequest(Base):
    __tablename__ = "approval_requests"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False, index=True)
    agent_run_id = Column(Integer, ForeignKey("agent_runs.id"), nullable=False, index=True)
    tool_name = Column(String, nullable=False)
    arguments = Column(Text, nullable=False)
    risk_level = Column(String, nullable=False, index=True)
    reason = Column(Text, nullable=False)
    status = Column(String, default="pending", nullable=False, index=True)
    requested_by = Column(String, nullable=False, default="agent")
    decided_by = Column(String, nullable=True)
    decision_reason = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    decided_at = Column(DateTime(timezone=True), nullable=True)

    job = relationship("Job")
    agent_run = relationship("AgentRun", back_populates="approval_requests")
    decisions = relationship("ApprovalDecision", back_populates="request", cascade="all, delete-orphan")


class ApprovalDecision(Base):
    __tablename__ = "approval_decisions"

    id = Column(Integer, primary_key=True, index=True)
    approval_request_id = Column(Integer, ForeignKey("approval_requests.id"), nullable=False, index=True)
    decision = Column(String, nullable=False)
    decided_by = Column(String, nullable=False)
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    request = relationship("ApprovalRequest", back_populates="decisions")


class CaseMemory(Base):
    """A privacy-filtered, technician-approved repair case for future retrieval."""

    __tablename__ = "case_memories"

    id = Column(Integer, primary_key=True, index=True)
    outcome_id = Column(Integer, ForeignKey("verified_outcomes.id"), nullable=False, unique=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False, index=True)
    manufacturer = Column(String, nullable=True, index=True)
    equipment_model = Column(String, nullable=False, index=True)
    equipment_type = Column(String, nullable=True, index=True)
    error_code = Column(String, nullable=True, index=True)
    symptoms = Column(Text, nullable=False)
    actual_cause = Column(String, nullable=False)
    repair_action = Column(Text, nullable=False)
    first_time_fix = Column(Boolean, nullable=False)
    return_visit_required = Column(Boolean, nullable=False)
    searchable_text = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    outcome = relationship("VerifiedOutcome")
    job = relationship("Job", back_populates="case_memories")


class DiagnosticEvaluation(Base):
    __tablename__ = "diagnostic_evaluations"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False, index=True)
    diagnostic_run_id = Column(Integer, ForeignKey("diagnostic_runs.id"), nullable=True, index=True)
    agent_run_id = Column(Integer, ForeignKey("agent_runs.id"), nullable=True, index=True)
    outcome_id = Column(Integer, ForeignKey("verified_outcomes.id"), nullable=False, unique=True, index=True)
    top1_cause_correct = Column(Boolean, nullable=True)
    top3_cause_correct = Column(Boolean, nullable=True)
    recommendation_accepted = Column(Boolean, nullable=True)
    first_time_fix = Column(Boolean, nullable=False)
    return_visit_required = Column(Boolean, nullable=False)
    details = Column(Text, nullable=False, default="{}")
    evaluated_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    job = relationship("Job", back_populates="diagnostic_evaluations")
    diagnostic_run = relationship("DiagnosticRun")
    agent_run = relationship("AgentRun")
    outcome = relationship("VerifiedOutcome")


class TechnicianFeedback(Base):
    __tablename__ = "technician_feedback"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False, index=True)
    diagnostic_run_id = Column(Integer, ForeignKey("diagnostic_runs.id"), nullable=True)
    technician_name = Column(String, nullable=False)
    recommendation_accepted = Column(Boolean, nullable=False)
    corrected_cause = Column(String, nullable=True)
    action_taken = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    job = relationship("Job", back_populates="technician_feedback")


class CustomerFeedback(Base):
    __tablename__ = "customer_feedback"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False, index=True)
    rating = Column(Integer, nullable=False)
    issue_resolved = Column(Boolean, nullable=False)
    comments = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    job = relationship("Job", back_populates="customer_feedback")


class VerifiedOutcome(Base):
    __tablename__ = "verified_outcomes"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False, index=True)
    actual_cause = Column(String, nullable=False)
    repair_action = Column(Text, nullable=False)
    first_time_fix = Column(Boolean, nullable=False)
    return_visit_required = Column(Boolean, default=False, nullable=False)
    verified_by = Column(String, nullable=False)
    approved_for_retrieval = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    job = relationship("Job", back_populates="outcomes")


class JobEvent(Base):
    __tablename__ = "job_events"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False, index=True)
    event_type = Column(String, nullable=False)
    event_data = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    job = relationship("Job", back_populates="events")


class ManualDocument(Base):
    __tablename__ = "manual_documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, nullable=False)
    title = Column(String, nullable=False)
    manufacturer = Column(String, nullable=False, default="unknown", index=True)
    equipment_types = Column(Text, nullable=False, default="[]")
    model_families = Column(Text, nullable=False, default="[]")
    refrigerants = Column(Text, nullable=False, default="[]")
    document_version = Column(String, nullable=True)
    content_hash = Column(String, nullable=False)
    indexed_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    chunks = relationship("ManualChunk", back_populates="document", cascade="all, delete-orphan")


class ManualChunk(Base):
    __tablename__ = "manual_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("manual_documents.id"), nullable=False, index=True)
    page = Column(Integer, nullable=False)
    section = Column(String, nullable=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding_json = Column(Text, nullable=False)

    document = relationship("ManualDocument", back_populates="chunks")
