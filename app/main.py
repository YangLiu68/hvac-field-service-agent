import json
import os
import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.database import SessionLocal, initialize_database
from app.models import (
    AIAnalysis,
    AgentMessage,
    AgentRun,
    ApprovalDecision,
    ApprovalRequest,
    CustomerFeedback,
    DiagnosticHypothesis,
    DiagnosticRun,
    Equipment,
    Job,
    JobBrief,
    JobEvent,
    MeasurementResult,
    MeasurementRequest,
    ToolCall,
    TechnicianFeedback,
    TechnicianProfile,
    ServiceRequest,
    ServiceSummary,
    Assignment,
    User,
    VerifiedOutcome,
    utcnow,
)
from app.schemas import (
    CustomerFeedbackCreate,
    CustomerFeedbackRead,
    DiagnosticRunRead,
    EquipmentCreate,
    EquipmentRead,
    JobCloseRequest,
    JobCreate,
    JobRead,
    JobBriefRead,
    ManualIngestResponse,
    ManualSearchRequest,
    ManualSearchResult,
    TechnicianFeedbackCreate,
    TechnicianFeedbackRead,
    TimelineEvent,
    VerifiedOutcomeRead,
    AgentRunCreate,
    AgentRunRead,
    ToolCallRead,
    ToolExecuteRequest,
    ToolExecuteResponse,
    SkillRead,
    SkillRouteRead,
    AgentRunResult,
    AgentMessageCreate,
    AgentMessageRead,
    ApprovalRequestRead,
    ApprovalDecisionCreate,
    ApprovalDecisionRead,
    CaseMemorySearchRequest,
    CaseMemorySearchResult,
    DiagnosticEvaluationRead,
    UserCreate,
    UserRead,
    TechnicianProfileCreate,
    TechnicianProfileRead,
    ServiceRequestCreate,
    CustomerPortalRequestCreate,
    ServiceRequestRead,
    TechnicianMatchRead,
    AssignmentCreate,
    AssignmentRead,
    ServiceSummaryRead,
    TechnicianTaskRead,
    LoginRequest,
    LoginResponse,
)
from app.agent import ApprovalRequired, ToolExecutionError, ToolExecutor, build_default_registry
from app.agent.skills import SkillRouter, build_default_skill_registry
from app.agent.orchestrator import AgentOrchestrator, AgentOrchestrationError
from app.services.ai_service import AIServiceError, analyze_job, answer_field_question
from app.services.rag_service import ManualIndexError, hybrid_search, ingest_manuals
from app.services.case_memory import evaluation_metrics, evaluate_closed_outcome, index_verified_outcome, search_case_memories


initialize_database()
app = FastAPI(title="HVAC Field Agent", version="0.2.0")
# Vite may serve the local page as localhost, 127.0.0.1, or IPv6 localhost.
frontend_origins = [origin.strip() for origin in os.getenv("FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://[::1]:5173").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    # Vite chooses the next free port when another dev server is already
    # running (for example 5174 instead of 5173).  Keep local development
    # usable without broadening production origins.
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1|\[::1\]):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
tool_registry = build_default_registry()
tool_executor = ToolExecutor(tool_registry)
skill_registry = build_default_skill_registry(tool_registry)
skill_router = SkillRouter(skill_registry)
agent_orchestrator = AgentOrchestrator(tool_registry, skill_registry, tool_executor)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


_auth_secret = os.getenv("APP_SECRET_KEY", "fieldwise-local-development-secret").encode()


def _password_hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"{base64.urlsafe_b64encode(salt).decode()}${base64.urlsafe_b64encode(digest).decode()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_text, digest_text = stored.split("$", 1)
        expected = _password_hash(password, base64.urlsafe_b64decode(salt_text.encode())).split("$", 1)[1]
        return hmac.compare_digest(expected, digest_text)
    except (ValueError, TypeError):
        return False


def _issue_token(user: User) -> str:
    payload = f"{user.id}:{user.role}:{int((utcnow() + timedelta(hours=12)).timestamp())}"
    signature = hmac.new(_auth_secret, payload.encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{payload}:{signature}".encode()).decode()


def get_current_user(authorization: str | None = Header(default=None), db: Session = Depends(get_db)) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Sign in required")
    try:
        raw = base64.urlsafe_b64decode(authorization.removeprefix("Bearer ").encode()).decode()
        user_id, role, expires_at, signature = raw.rsplit(":", 3)
        payload = f"{user_id}:{role}:{expires_at}"
        valid = hmac.compare_digest(signature, hmac.new(_auth_secret, payload.encode(), hashlib.sha256).hexdigest())
        user = db.get(User, int(user_id))
        if not valid or int(expires_at) < int(utcnow().timestamp()) or not user or not user.active or user.role != role:
            raise ValueError
        return user
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(status_code=401, detail="Session is invalid or expired")


def require_roles(*roles: str):
    def dependency(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status_code=403, detail="This workspace is not available for your role")
        return user
    return dependency


def get_job_or_404(db: Session, job_id: int) -> Job:
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def get_agent_run_or_404(db: Session, run_id: int) -> AgentRun:
    run = db.get(AgentRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return run


def approval_expired(request: ApprovalRequest) -> bool:
    if request.expires_at is None:
        return False
    expires_at = request.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at < utcnow()


def add_event(db: Session, job_id: int, event_type: str, data: dict):
    db.add(JobEvent(job_id=job_id, event_type=event_type, event_data=json.dumps(data)))


def _json_list(value: str) -> list[str]:
    try:
        parsed = json.loads(value or "[]")
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def _service_request_read(request: ServiceRequest) -> dict:
    return {
        "id": request.id, "customer_id": request.customer_id,
        "customer_name": request.customer.display_name, "equipment_id": request.equipment_id,
        "description": request.description, "service_address": request.service_address,
        "postal_code": request.postal_code, "preferred_window": request.preferred_window,
        "urgency": request.urgency, "status": request.status, "job_id": request.job_id,
        "created_at": request.created_at, "updated_at": request.updated_at,
    }


def _profile_read(profile: TechnicianProfile) -> dict:
    return {
        "id": profile.id, "user_id": profile.user_id, "technician_name": profile.user.display_name,
        "skills": _json_list(profile.skills), "certifications": _json_list(profile.certifications),
        "service_postal_codes": _json_list(profile.service_postal_codes),
        "available": profile.available, "on_call": profile.on_call,
        "created_at": profile.created_at, "updated_at": profile.updated_at,
    }


def _required_certifications(request: ServiceRequest) -> set[str]:
    text = request.description.lower()
    required: set[str] = set()
    if any(term in text for term in ("refrigerant", "leak", "hissing", "pressure", "charge")):
        required.add("epa_608")
    if any(term in text for term in ("live voltage", "electrical panel", "burning smell", "shock")):
        required.add("electrical_authorized")
    return required


def _technician_matches(db: Session, request: ServiceRequest) -> list[dict]:
    required_certs = _required_certifications(request)
    matches = []
    for profile in db.query(TechnicianProfile).join(User).filter(User.active.is_(True), TechnicianProfile.available.is_(True)).all():
        skills = set(_json_list(profile.skills))
        certifications = set(_json_list(profile.certifications))
        postal_codes = set(_json_list(profile.service_postal_codes))
        if request.equipment.equipment_type not in skills and "all_hvac" not in skills:
            continue
        if request.postal_code not in postal_codes and "*" not in postal_codes:
            continue
        if not required_certs.issubset(certifications):
            continue
        score = 40.0 + 30.0 + 20.0
        rationale = ["service area matches", f"skill matches {request.equipment.equipment_type}", "available now"]
        if required_certs:
            score += 10.0
            rationale.append("required safety certification verified")
        if request.urgency in {"high", "emergency"} and profile.on_call:
            score += 10.0
            rationale.append("on-call for urgent work")
        matches.append({"profile": profile, "score": score, "rationale": rationale})
    return sorted(matches, key=lambda item: (-item["score"], item["profile"].user.display_name))


def _assignment_read(assignment: Assignment) -> dict:
    return {
        "id": assignment.id, "service_request_id": assignment.service_request_id,
        "technician_profile_id": assignment.technician_id,
        "technician_name": assignment.technician.user.display_name,
        "dispatcher_id": assignment.dispatcher_id, "status": assignment.status,
        "match_score": assignment.match_score, "rationale": _json_list(assignment.rationale),
        "scheduled_for": assignment.scheduled_for, "job_id": assignment.service_request.job_id,
        "created_at": assignment.created_at, "accepted_at": assignment.accepted_at,
    }


@app.post("/equipment", response_model=EquipmentRead, status_code=201)
def create_equipment(payload: EquipmentCreate, db: Session = Depends(get_db)):
    if payload.serial_number:
        existing = db.query(Equipment).filter(Equipment.serial_number == payload.serial_number).first()
        if existing:
            raise HTTPException(status_code=409, detail="Equipment serial number already exists")
    equipment = Equipment(**payload.model_dump())
    db.add(equipment)
    db.commit()
    db.refresh(equipment)
    return equipment


@app.get("/equipment", response_model=list[EquipmentRead])
def list_equipment(db: Session = Depends(get_db)):
    return db.query(Equipment).order_by(Equipment.created_at.desc()).all()


@app.post("/users", response_model=UserRead, status_code=201)
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=409, detail="Email is already registered")
    values = payload.model_dump(exclude={"password"})
    user = User(**values, password_hash=_password_hash(payload.password) if payload.password else "")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/auth/login", response_model=LoginResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not user.active or not user.password_hash or not _verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {"access_token": _issue_token(user), "user": user}


@app.get("/auth/me", response_model=UserRead)
def current_user(user: User = Depends(get_current_user)):
    return user


@app.post("/auth/bootstrap", response_model=LoginResponse, status_code=201)
def bootstrap_admin(payload: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.role == "admin", User.password_hash != "").first():
        raise HTTPException(status_code=409, detail="Initial administrator already exists; sign in instead")
    if payload.role != "admin" or not payload.password:
        raise HTTPException(status_code=400, detail="Bootstrap requires an admin account and password")
    # A customer record can already exist when its owner submitted a request
    # before the local team account was initialized. Reuse only a credentialless
    # record; never overwrite an account that already has a password.
    user = db.query(User).filter(User.email == payload.email).first()
    if user:
        if user.password_hash:
            raise HTTPException(status_code=409, detail="This email already has an account; sign in instead")
        user.display_name = payload.display_name
        user.role = "admin"
        user.password_hash = _password_hash(payload.password)
    else:
        user = User(**payload.model_dump(exclude={"password"}), password_hash=_password_hash(payload.password))
        db.add(user)
    db.commit()
    db.refresh(user)
    return {"access_token": _issue_token(user), "user": user}


@app.get("/auth/bootstrap-status")
def bootstrap_status(db: Session = Depends(get_db)):
    return {"setup_required": db.query(User).filter(User.role == "admin", User.password_hash != "").first() is None}


@app.get("/users", response_model=list[UserRead])
def list_users(role: str | None = None, db: Session = Depends(get_db)):
    query = db.query(User).order_by(User.created_at.desc())
    if role:
        query = query.filter(User.role == role)
    return query.all()


@app.post("/technicians", response_model=TechnicianProfileRead, status_code=201)
def create_technician_profile(payload: TechnicianProfileCreate, db: Session = Depends(get_db)):
    user = db.get(User, payload.user_id)
    if user is None or user.role != "technician":
        raise HTTPException(status_code=400, detail="user_id must belong to a technician")
    if user.technician_profile:
        raise HTTPException(status_code=409, detail="Technician profile already exists")
    profile = TechnicianProfile(
        user_id=user.id, skills=json.dumps(payload.skills), certifications=json.dumps(payload.certifications),
        service_postal_codes=json.dumps(payload.service_postal_codes), available=payload.available, on_call=payload.on_call,
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return _profile_read(profile)


@app.get("/technicians", response_model=list[TechnicianProfileRead])
def list_technicians(db: Session = Depends(get_db)):
    return [_profile_read(profile) for profile in db.query(TechnicianProfile).order_by(TechnicianProfile.id).all()]


@app.post("/service-requests", response_model=ServiceRequestRead, status_code=201)
def create_service_request(payload: ServiceRequestCreate, db: Session = Depends(get_db)):
    customer = db.get(User, payload.customer_id)
    if customer is None or customer.role != "customer":
        raise HTTPException(status_code=400, detail="customer_id must belong to a customer")
    if db.get(Equipment, payload.equipment_id) is None:
        raise HTTPException(status_code=404, detail="Equipment not found")
    request = ServiceRequest(**payload.model_dump())
    db.add(request)
    db.commit()
    db.refresh(request)
    return _service_request_read(request)


@app.post("/customer-portal/requests", response_model=ServiceRequestRead, status_code=201)
def create_customer_portal_request(payload: CustomerPortalRequestCreate, db: Session = Depends(get_db)):
    customer = db.query(User).filter(User.email == payload.email).first()
    if customer is None:
        customer = User(display_name=payload.customer_name, email=payload.email, phone=payload.phone, role="customer")
        db.add(customer)
        db.flush()
    elif customer.role != "customer":
        raise HTTPException(status_code=409, detail="Email belongs to a non-customer account")
    equipment = Equipment(manufacturer=payload.manufacturer, model_number=payload.model_number, equipment_type=payload.equipment_type)
    db.add(equipment)
    db.flush()
    request = ServiceRequest(customer_id=customer.id, equipment_id=equipment.id, description=payload.description,
                             service_address=payload.service_address, postal_code=payload.postal_code,
                             preferred_window=payload.preferred_window, urgency=payload.urgency)
    db.add(request)
    db.commit()
    db.refresh(request)
    return _service_request_read(request)


@app.get("/service-requests", response_model=list[ServiceRequestRead])
def list_service_requests(status: str | None = None, db: Session = Depends(get_db)):
    query = db.query(ServiceRequest).order_by(ServiceRequest.created_at.desc())
    if status:
        query = query.filter(ServiceRequest.status == status)
    return [_service_request_read(request) for request in query.all()]


@app.get("/service-requests/{request_id}/matches", response_model=list[TechnicianMatchRead])
def recommend_technicians(request_id: int, db: Session = Depends(get_db)):
    request = db.get(ServiceRequest, request_id)
    if request is None:
        raise HTTPException(status_code=404, detail="Service request not found")
    return [{"technician_profile_id": item["profile"].id, "technician_name": item["profile"].user.display_name,
             "score": item["score"], "rationale": item["rationale"]} for item in _technician_matches(db, request)]


@app.post("/service-requests/{request_id}/assignments", response_model=AssignmentRead, status_code=201)
def assign_service_request(request_id: int, payload: AssignmentCreate, authorization: str | None = Header(default=None), db: Session = Depends(get_db)):
    request = db.get(ServiceRequest, request_id)
    # New browser clients use their signed session. The payload fallback preserves
    # compatibility with the Stage 9 evaluation API while it is migrated.
    dispatcher = get_current_user(authorization, db) if authorization else db.get(User, payload.dispatcher_id)
    if request is None:
        raise HTTPException(status_code=404, detail="Service request not found")
    if dispatcher is None or dispatcher.role not in {"dispatcher", "admin"}:
        raise HTTPException(status_code=400, detail="dispatcher_id must belong to a dispatcher or admin")
    if request.status not in {"submitted", "triaged"}:
        raise HTTPException(status_code=409, detail="Service request is already assigned or closed")
    match = next((item for item in _technician_matches(db, request) if item["profile"].id == payload.technician_profile_id), None)
    if match is None:
        raise HTTPException(status_code=409, detail="Technician is not eligible for this request")
    job = Job(customer_name=request.customer.display_name, equipment_id=request.equipment_id,
              equipment_model=request.equipment.model_number, technician_notes=request.description, status="open")
    db.add(job)
    db.flush()
    request.job_id = job.id
    request.status = "assigned"
    assignment = Assignment(service_request_id=request.id, technician_id=match["profile"].id,
                            dispatcher_id=dispatcher.id, match_score=match["score"],
                            rationale=json.dumps(match["rationale"]), scheduled_for=payload.scheduled_for)
    db.add(assignment)
    db.flush()
    add_event(db, job.id, "service_request_assigned", {"service_request_id": request.id, "assignment_id": assignment.id,
              "technician_profile_id": match["profile"].id, "dispatcher_id": dispatcher.id})
    db.commit()
    db.refresh(assignment)
    return _assignment_read(assignment)


@app.post("/assignments/{assignment_id}/accept", response_model=AssignmentRead)
def accept_assignment(assignment_id: int, technician_profile_id: int, db: Session = Depends(get_db)):
    assignment = db.get(Assignment, assignment_id)
    if assignment is None:
        raise HTTPException(status_code=404, detail="Assignment not found")
    if assignment.technician_id != technician_profile_id:
        raise HTTPException(status_code=403, detail="Assignment belongs to a different technician")
    if assignment.status != "assigned":
        raise HTTPException(status_code=409, detail="Assignment is not awaiting acceptance")
    assignment.status = "accepted"
    assignment.accepted_at = utcnow()
    assignment.service_request.status = "in_service"
    add_event(db, assignment.service_request.job_id, "assignment_accepted", {"assignment_id": assignment.id, "technician_profile_id": technician_profile_id})
    db.commit()
    db.refresh(assignment)
    return _assignment_read(assignment)


@app.get("/jobs", response_model=list[JobRead])
def list_jobs(status: str | None = None, db: Session = Depends(get_db)):
    query = db.query(Job).order_by(Job.updated_at.desc())
    if status:
        query = query.filter(Job.status == status)
    return query.all()


@app.get("/technician/tasks", response_model=list[TechnicianTaskRead])
def list_technician_tasks(limit: int = 12, db: Session = Depends(get_db)):
    """Mobile-style task feed built from real, non-closed service jobs."""
    jobs = db.query(Job).filter(Job.status != "closed").order_by(Job.updated_at.desc()).limit(max(1, min(limit, 50))).all()
    tasks = []
    for job in jobs:
        request = db.query(ServiceRequest).filter(ServiceRequest.job_id == job.id).first()
        title = (request.description if request else job.technician_notes).split("\n", 1)[0][:72]
        tasks.append({
            "job_id": job.id,
            "title": title or "HVAC service visit",
            "customer_name": job.customer_name,
            "status": job.status,
            "equipment_model": job.equipment_model,
            "error_code": job.error_code,
            "service_address": request.service_address if request else None,
            "scheduled_for": job.created_at,
            "brief_ready": job.brief is not None,
        })
    return tasks


def _job_safety_flags(job: Job) -> list[str]:
    text = f"{job.technician_notes} {job.error_code or ''}".lower()
    flags = []
    if any(term in text for term in ("burning", "shock", "voltage", "electrical")):
        flags.append("Electrical safety check required")
    if any(term in text for term in ("leak", "refrigerant", "hissing", "pressure")):
        flags.append("Refrigerant service may require qualified handling")
    return flags


@app.get("/jobs/{job_id}/brief", response_model=JobBriefRead)
def get_or_create_job_brief(job_id: int, db: Session = Depends(get_db)):
    job = get_job_or_404(db, job_id)
    brief = job.brief
    if brief is None:
        equipment = job.equipment
        device = f"{equipment.manufacturer} {equipment.model_number}" if equipment else job.equipment_model
        flags = _job_safety_flags(job)
        brief = JobBrief(
            job_id=job.id,
            summary=(f"Visit {job.customer_name}. Equipment: {device}. "
                     f"Reported concern: {job.technician_notes}"),
            safety_flags=json.dumps(flags),
        )
        db.add(brief)
        db.commit()
        db.refresh(brief)
    return {"id": brief.id, "job_id": brief.job_id, "summary": brief.summary,
            "safety_flags": _json_list(brief.safety_flags), "created_at": brief.created_at, "updated_at": brief.updated_at}


@app.post("/jobs/{job_id}/summary", response_model=ServiceSummaryRead, status_code=201)
def create_service_summary(job_id: int, db: Session = Depends(get_db)):
    job = get_job_or_404(db, job_id)
    latest = db.query(DiagnosticRun).filter(DiagnosticRun.job_id == job.id).order_by(DiagnosticRun.id.desc()).first()
    measurements = db.query(MeasurementResult).join(MeasurementResult.request).filter(MeasurementRequest.job_id == job.id).all()
    measurement_text = "; ".join(f"{item.request.measurement_type}: {item.value}{(' ' + item.unit) if item.unit else ''}" for item in measurements)
    content = (
        f"Service summary — {job.customer_name}\n"
        f"Equipment: {job.equipment_model}. Reported issue: {job.technician_notes}\n"
        f"Assessment: {latest.likely_cause if latest and latest.likely_cause else 'Pending technician verification.'}\n"
        f"Recommended next step: {latest.recommendation if latest and latest.recommendation else 'Continue approved diagnostic checks.'}\n"
        f"Measurements: {measurement_text or 'No measurements recorded.'}"
    )
    summary = ServiceSummary(job_id=job.id, content=content)
    db.add(summary)
    add_event(db, job.id, "service_summary_created", {"summary_id": None})
    db.commit()
    db.refresh(summary)
    return summary


@app.get("/assignments", response_model=list[AssignmentRead])
def list_assignments(technician_profile_id: int | None = None, status: str | None = None, db: Session = Depends(get_db)):
    query = db.query(Assignment).order_by(Assignment.created_at.desc())
    if technician_profile_id is not None:
        query = query.filter(Assignment.technician_id == technician_profile_id)
    if status:
        query = query.filter(Assignment.status == status)
    return [_assignment_read(assignment) for assignment in query.all()]


@app.post("/jobs", response_model=JobRead, status_code=201)
def create_job(payload: JobCreate, db: Session = Depends(get_db)):
    equipment = None
    if payload.equipment_id is not None:
        equipment = db.get(Equipment, payload.equipment_id)
        if not equipment:
            raise HTTPException(status_code=404, detail="Equipment not found")

    model_number = payload.equipment_model or equipment.model_number
    job = Job(
        customer_name=payload.customer_name,
        equipment_id=payload.equipment_id,
        equipment_model=model_number,
        technician_notes=payload.technician_notes,
        error_code=payload.error_code,
        status="open",
    )
    db.add(job)
    db.flush()
    add_event(db, job.id, "job_created", {"status": "open", "equipment_model": model_number})
    db.commit()
    db.refresh(job)
    return job


@app.get("/jobs/{job_id}", response_model=JobRead)
def get_job(job_id: int, db: Session = Depends(get_db)):
    return get_job_or_404(db, job_id)


def run_diagnosis(job_id: int, db: Session) -> DiagnosticRun:
    job = get_job_or_404(db, job_id)
    if job.status == "closed":
        raise HTTPException(status_code=409, detail="Closed jobs cannot be diagnosed")

    diagnostic_run = DiagnosticRun(
        job_id=job.id,
        status="running",
        model_name=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    )
    job.status = "diagnosing"
    job.updated_at = utcnow()
    db.add(diagnostic_run)
    db.flush()
    add_event(db, job.id, "diagnosis_started", {"diagnostic_run_id": diagnostic_run.id})
    db.commit()

    notes = job.technician_notes
    if job.error_code:
        notes = f"{notes}\nReported error code: {job.error_code}"

    try:
        result = analyze_job(
            notes=notes,
            equipment_model=job.equipment_model,
            manufacturer=job.equipment.manufacturer if job.equipment else None,
            equipment_type=job.equipment.equipment_type if job.equipment else None,
        )
    except (AIServiceError, ManualIndexError, ValueError) as exc:
        diagnostic_run.status = "failed"
        diagnostic_run.error_message = str(exc)
        diagnostic_run.completed_at = utcnow()
        job.status = "open"
        job.updated_at = utcnow()
        add_event(
            db,
            job.id,
            "diagnosis_failed",
            {"diagnostic_run_id": diagnostic_run.id, "error": str(exc)},
        )
        db.commit()
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    sources_json = json.dumps([source.model_dump() for source in result.sources])
    diagnostic_run.status = "completed"
    diagnostic_run.likely_cause = result.likely_cause
    diagnostic_run.confidence = result.confidence
    diagnostic_run.recommendation = result.recommendation
    diagnostic_run.sources = sources_json
    diagnostic_run.completed_at = utcnow()

    db.add(
        DiagnosticHypothesis(
            diagnostic_run_id=diagnostic_run.id,
            rank=1,
            cause=result.likely_cause,
            confidence=result.confidence,
            supporting_evidence=sources_json,
            next_test=result.recommendation,
            selected=True,
        )
    )
    # Continue populating the Stage 1 table for backward compatibility.
    db.add(
        AIAnalysis(
            job_id=job.id,
            likely_cause=result.likely_cause,
            confidence=result.confidence,
            recommendation=result.recommendation,
            sources=sources_json,
        )
    )
    job.status = "awaiting_technician_feedback"
    job.updated_at = utcnow()
    add_event(
        db,
        job.id,
        "diagnosis_completed",
        {
            "diagnostic_run_id": diagnostic_run.id,
            "likely_cause": result.likely_cause,
            "confidence": result.confidence,
        },
    )
    db.commit()
    db.refresh(diagnostic_run)
    return diagnostic_run


@app.post("/jobs/{job_id}/diagnose", response_model=DiagnosticRunRead)
def diagnose(job_id: int, db: Session = Depends(get_db)):
    return run_diagnosis(job_id, db)


@app.post("/jobs/{job_id}/analyze", response_model=DiagnosticRunRead, deprecated=True)
def analyze_legacy(job_id: int, db: Session = Depends(get_db)):
    """Stage 1 compatibility alias; new clients should use /diagnose."""
    return run_diagnosis(job_id, db)


@app.post(
    "/jobs/{job_id}/technician-feedback",
    response_model=TechnicianFeedbackRead,
    status_code=201,
)
def add_technician_feedback(
    job_id: int,
    payload: TechnicianFeedbackCreate,
    db: Session = Depends(get_db),
):
    job = get_job_or_404(db, job_id)
    if payload.diagnostic_run_id is not None:
        diagnostic_run = db.get(DiagnosticRun, payload.diagnostic_run_id)
        if not diagnostic_run or diagnostic_run.job_id != job_id:
            raise HTTPException(status_code=400, detail="Diagnostic run does not belong to this job")

    feedback = TechnicianFeedback(job_id=job_id, **payload.model_dump())
    db.add(feedback)
    job.status = "in_service"
    job.updated_at = utcnow()
    db.flush()
    add_event(
        db,
        job_id,
        "technician_feedback_added",
        {
            "feedback_id": feedback.id,
            "recommendation_accepted": payload.recommendation_accepted,
        },
    )
    db.commit()
    db.refresh(feedback)
    return feedback


@app.post(
    "/jobs/{job_id}/customer-feedback",
    response_model=CustomerFeedbackRead,
    status_code=201,
)
def add_customer_feedback(
    job_id: int,
    payload: CustomerFeedbackCreate,
    db: Session = Depends(get_db),
):
    job = get_job_or_404(db, job_id)
    feedback = CustomerFeedback(job_id=job_id, **payload.model_dump())
    db.add(feedback)
    job.updated_at = utcnow()
    db.flush()
    add_event(
        db,
        job_id,
        "customer_feedback_added",
        {"feedback_id": feedback.id, "rating": payload.rating, "issue_resolved": payload.issue_resolved},
    )
    db.commit()
    db.refresh(feedback)
    return feedback


@app.post("/jobs/{job_id}/close", response_model=VerifiedOutcomeRead, status_code=201)
def close_job(job_id: int, payload: JobCloseRequest, db: Session = Depends(get_db)):
    job = get_job_or_404(db, job_id)
    if job.status == "closed":
        raise HTTPException(status_code=409, detail="Job is already closed")

    outcome = VerifiedOutcome(job_id=job_id, **payload.model_dump())
    db.add(outcome)
    job.status = "closed"
    job.updated_at = utcnow()
    db.flush()
    case_memory = index_verified_outcome(db, outcome)
    evaluation = evaluate_closed_outcome(db, outcome)
    add_event(
        db,
        job_id,
        "job_closed",
        {
            "outcome_id": outcome.id,
            "actual_cause": payload.actual_cause,
            "first_time_fix": payload.first_time_fix,
            "approved_for_retrieval": payload.approved_for_retrieval,
            "case_memory_id": case_memory.id if case_memory else None,
            "evaluation_id": evaluation.id,
        },
    )
    db.commit()
    db.refresh(outcome)
    return outcome


@app.get("/jobs/{job_id}/diagnostic-evaluation", response_model=DiagnosticEvaluationRead)
def get_diagnostic_evaluation(job_id: int, db: Session = Depends(get_db)):
    get_job_or_404(db, job_id)
    from app.models import DiagnosticEvaluation
    evaluation = db.query(DiagnosticEvaluation).filter(DiagnosticEvaluation.job_id == job_id).order_by(DiagnosticEvaluation.id.desc()).first()
    if evaluation is None:
        raise HTTPException(status_code=404, detail="Diagnostic evaluation not found")
    return evaluation


@app.get("/evaluation/metrics")
def get_evaluation_metrics(db: Session = Depends(get_db)):
    return evaluation_metrics(db)


@app.post("/jobs/{job_id}/case-memory/search", response_model=list[CaseMemorySearchResult])
def search_job_case_memory(job_id: int, payload: CaseMemorySearchRequest, db: Session = Depends(get_db)):
    job = get_job_or_404(db, job_id)
    return search_case_memories(db, job=job, query=payload.query, limit=payload.limit)


@app.get("/jobs/{job_id}/timeline", response_model=list[TimelineEvent])
def get_job_timeline(job_id: int, db: Session = Depends(get_db)):
    get_job_or_404(db, job_id)
    events = (
        db.query(JobEvent)
        .filter(JobEvent.job_id == job_id)
        .order_by(JobEvent.created_at.asc(), JobEvent.id.asc())
        .all()
    )
    return [
        TimelineEvent(
            event_type=event.event_type,
            created_at=event.created_at,
            data=json.loads(event.event_data or "{}"),
        )
        for event in events
    ]


@app.get("/agent/tools")
def list_agent_tools():
    """Return the tool contracts that a future orchestrator may expose to an LLM."""
    return {"tools": tool_registry.schemas()}


@app.get("/agent/skills", response_model=list[SkillRead])
def list_agent_skills():
    return skill_registry.list()


@app.get("/agent/skills/{skill_name}", response_model=SkillRead)
def get_agent_skill(skill_name: str):
    try:
        return skill_registry.get(skill_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/jobs/{job_id}/skill-route", response_model=SkillRouteRead)
def route_job_skill(job_id: int, db: Session = Depends(get_db)):
    # The route is intentionally deterministic and does not invoke an LLM.
    job = get_job_or_404(db, job_id)
    equipment = job.equipment
    result = skill_router.route(
        notes=job.technician_notes,
        error_code=job.error_code,
        equipment_type=equipment.equipment_type if equipment else None,
        equipment_model=job.equipment_model,
    )
    return result


@app.post("/jobs/{job_id}/agent-runs", response_model=AgentRunRead, status_code=201)
def create_agent_run(
    job_id: int,
    payload: AgentRunCreate,
    db: Session = Depends(get_db),
):
    job = get_job_or_404(db, job_id)
    if job.status == "closed":
        raise HTTPException(status_code=409, detail="Closed jobs cannot start an agent run")
    selected_skill_name = payload.skill_name
    if selected_skill_name is None:
        equipment = job.equipment
        selected_skill_name = skill_router.route(
            notes=job.technician_notes,
            error_code=job.error_code,
            equipment_type=equipment.equipment_type if equipment else None,
            equipment_model=job.equipment_model,
        ).selected_skill
    if selected_skill_name is not None:
        try:
            skill = skill_registry.get(selected_skill_name)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        max_steps = min(payload.max_steps, skill.max_steps)
    else:
        max_steps = payload.max_steps
    diagnostic_run = DiagnosticRun(
        job_id=job.id,
        status="running",
        model_name=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    )
    db.add(diagnostic_run)
    db.flush()
    run = AgentRun(
        job_id=job.id,
        diagnostic_run_id=diagnostic_run.id,
        max_steps=max_steps,
        skill_name=selected_skill_name,
        status="tool_testing",
    )
    db.add(run)
    job.status = "diagnosing"
    job.updated_at = utcnow()
    db.flush()
    add_event(db, job.id, "agent_run_created", {"agent_run_id": run.id, "diagnostic_run_id": diagnostic_run.id})
    db.commit()
    db.refresh(run)
    return run


@app.get("/agent-runs/{run_id}", response_model=AgentRunRead)
def get_agent_run(run_id: int, db: Session = Depends(get_db)):
    return get_agent_run_or_404(db, run_id)


@app.get("/agent-runs/{run_id}/messages", response_model=list[AgentMessageRead])
def list_agent_messages(run_id: int, db: Session = Depends(get_db)):
    get_agent_run_or_404(db, run_id)
    return db.query(AgentMessage).filter(AgentMessage.agent_run_id == run_id).order_by(AgentMessage.id.asc()).all()


@app.post("/agent-runs/{run_id}/tools", response_model=ToolExecuteResponse)
def execute_agent_tool(
    run_id: int,
    payload: ToolExecuteRequest,
    db: Session = Depends(get_db),
):
    get_agent_run_or_404(db, run_id)
    if payload.approved:
        raise HTTPException(status_code=403, detail="Use the approval endpoint to authorize a protected tool")
    try:
        execution = tool_executor.execute(
            db=db,
            agent_run_id=run_id,
            tool_name=payload.tool_name,
            arguments=payload.arguments,
            approved=payload.approved,
        )
        if payload.tool_name == "record_measurement":
            # Replace the earlier "pending" tool observation with the actual
            # technician result before the orchestrator resumes the model.
            run = get_agent_run_or_404(db, run_id)
            try:
                pending_input = json.loads(run.pending_input or "[]")
            except json.JSONDecodeError:
                pending_input = []
            if isinstance(pending_input, list) and pending_input:
                observation = {
                    "status": "completed",
                    "measurement_request_id": payload.arguments["request_id"],
                    "value": payload.arguments["value"],
                    "unit": payload.arguments.get("unit"),
                    "notes": payload.arguments.get("notes"),
                }
                pending_input[-1]["output"] = json.dumps(observation)
                run.pending_input = json.dumps(pending_input)
                db.add(JobEvent(
                    job_id=run.job_id,
                    event_type="technician_measurement_observed",
                    event_data=json.dumps(observation),
                ))
                db.commit()
        return execution
    except ToolExecutionError as exc:
        message = str(exc)
        status_code = 409 if "not allowed" in message or "approval" in message or "maximum" in message else 400
        raise HTTPException(status_code=status_code, detail=message) from exc


@app.get("/agent-runs/{run_id}/tool-calls", response_model=list[ToolCallRead])
def list_agent_tool_calls(run_id: int, db: Session = Depends(get_db)):
    get_agent_run_or_404(db, run_id)
    return db.query(ToolCall).filter(ToolCall.agent_run_id == run_id).order_by(ToolCall.id.asc()).all()


@app.get("/approvals/pending", response_model=list[ApprovalRequestRead])
def list_pending_approvals(db: Session = Depends(get_db)):
    now = utcnow()
    requests = db.query(ApprovalRequest).filter(ApprovalRequest.status == "pending").order_by(ApprovalRequest.created_at.asc()).all()
    for request in requests:
        if approval_expired(request):
            request.status = "expired"
            request.decided_at = now
            request.decision_reason = "Approval request expired"
            request.agent_run.status = "failed"
            request.agent_run.error_message = "Approval request expired"
    db.commit()
    return [request for request in requests if request.status == "pending"]


def _get_pending_approval(db: Session, approval_id: int) -> ApprovalRequest:
    request = db.get(ApprovalRequest, approval_id)
    if request is None:
        raise HTTPException(status_code=404, detail="Approval request not found")
    if request.status != "pending":
        raise HTTPException(status_code=409, detail=f"Approval request is {request.status}")
    if approval_expired(request):
        request.status = "expired"
        request.decided_at = utcnow()
        request.agent_run.status = "failed"
        request.agent_run.error_message = "Approval request expired"
        db.commit()
        raise HTTPException(status_code=409, detail="Approval request expired")
    return request


@app.post("/approvals/{approval_id}/approve", response_model=ApprovalDecisionRead)
def approve_agent_tool(approval_id: int, payload: ApprovalDecisionCreate, db: Session = Depends(get_db)):
    request = _get_pending_approval(db, approval_id)
    if request.risk_level == "critical":
        raise HTTPException(status_code=403, detail="Critical actions are prohibited and cannot be approved")
    decision = ApprovalDecision(approval_request_id=request.id, decision="approved", decided_by=payload.decided_by, reason=payload.reason)
    request.status = "approved"
    request.decided_by = payload.decided_by
    request.decision_reason = payload.reason
    request.decided_at = utcnow()
    db.add(decision)
    db.flush()
    run = request.agent_run
    try:
        tool_result = tool_executor.execute(db=db, agent_run_id=run.id, tool_name=request.tool_name, arguments=json.loads(request.arguments), approved=True)
    except ToolExecutionError as exc:
        run.status = "failed"
        run.error_message = str(exc)
        run.completed_at = utcnow()
        db.commit()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    run = db.get(AgentRun, run.id)
    pending_call = json.loads(run.pending_approval_call or "{}")
    run.pending_approval_call = None
    if run.status not in {"completed", "escalated"}:
        run.status = "waiting_for_technician" if run.job.status == "waiting_for_technician" else "running"
        run.pending_input = json.dumps([{
            "type": "function_call_output",
            "call_id": pending_call.get("call_id"),
            "output": json.dumps(tool_result["result"]),
        }])
    db.add(JobEvent(job_id=run.job_id, event_type="approval_granted", event_data=json.dumps({"approval_id": request.id, "tool_name": request.tool_name, "decided_by": payload.decided_by})))
    db.commit()
    return ApprovalDecisionRead(approval_request_id=request.id, status="approved", message="Protected tool executed after approval", agent_run_id=run.id, tool_result=tool_result)


@app.post("/approvals/{approval_id}/reject", response_model=ApprovalDecisionRead)
def reject_agent_tool(approval_id: int, payload: ApprovalDecisionCreate, db: Session = Depends(get_db)):
    request = _get_pending_approval(db, approval_id)
    decision = ApprovalDecision(approval_request_id=request.id, decision="rejected", decided_by=payload.decided_by, reason=payload.reason)
    request.status = "rejected"
    request.decided_by = payload.decided_by
    request.decision_reason = payload.reason
    request.decided_at = utcnow()
    run = request.agent_run
    run.status = "escalated"
    run.error_message = payload.reason or "Protected tool request was rejected"
    run.completed_at = utcnow()
    run.pending_approval_call = None
    run.job.status = "escalated"
    db.add(decision)
    db.add(JobEvent(job_id=run.job_id, event_type="approval_rejected", event_data=json.dumps({"approval_id": request.id, "tool_name": request.tool_name, "decided_by": payload.decided_by, "reason": payload.reason})))
    db.commit()
    return ApprovalDecisionRead(approval_request_id=request.id, status="rejected", message="Protected tool was not executed", agent_run_id=run.id)


def _agent_result_message(result: dict) -> str:
    """Turn an orchestrator result dict into a useful persisted chat entry."""
    if result.get("message"):
        return str(result["message"])
    pending = result.get("pending_action") or {}
    if pending.get("status") == "pending":
        return "A field measurement is needed before the diagnosis can continue."
    return f"Agent status: {result.get('status', 'unknown')}"


@app.post("/agent-runs/{run_id}/start", response_model=AgentRunResult)
def start_agent_run(run_id: int, db: Session = Depends(get_db)):
    try:
        result = agent_orchestrator.run(db, run_id)
        db.add(AgentMessage(agent_run_id=run_id, role="assistant", content=_agent_result_message(result)))
        db.commit()
        return result
    except AgentOrchestrationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/agent-runs/{run_id}/continue", response_model=AgentRunResult)
def continue_agent_run(run_id: int, db: Session = Depends(get_db)):
    try:
        result = agent_orchestrator.run(db, run_id)
        db.add(AgentMessage(agent_run_id=run_id, role="assistant", content=_agent_result_message(result)))
        db.commit()
        return result
    except AgentOrchestrationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/agent-runs/{run_id}/messages", response_model=AgentMessageRead)
def chat_with_assistant(run_id: int, payload: AgentMessageCreate, db: Session = Depends(get_db)):
    """Natural-language technician chat, kept separate from guided tools."""
    run = get_agent_run_or_404(db, run_id)
    job = run.job
    equipment = job.equipment
    prior_messages = (
        db.query(AgentMessage)
        .filter(AgentMessage.agent_run_id == run_id)
        .order_by(AgentMessage.id.desc())
        .limit(8)
        .all()
    )
    history = [
        {"role": "user" if item.role == "technician" else "assistant", "content": item.content}
        for item in reversed(prior_messages)
        if item.role in {"technician", "assistant"}
    ]
    technician_message = AgentMessage(agent_run_id=run_id, role="technician", content=payload.message)
    db.add(technician_message)
    db.commit()
    try:
        reply = answer_field_question(
            payload.message,
            equipment_model=job.equipment_model,
            manufacturer=equipment.manufacturer if equipment else None,
            equipment_type=equipment.equipment_type if equipment else None,
            error_code=job.error_code,
            job_notes=job.technician_notes,
            history=history,
        )
        assistant_message = AgentMessage(agent_run_id=run_id, role="assistant", content=reply)
        db.add(assistant_message)
        db.commit()
        db.refresh(assistant_message)
        return assistant_message
    except AIServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/agent-runs/{run_id}/guided-messages", response_model=AgentRunResult)
def send_guided_agent_message(run_id: int, payload: AgentMessageCreate, db: Session = Depends(get_db)):
    """Reserved for structured workflow continuation and its audit trail."""
    try:
        db.add(AgentMessage(agent_run_id=run_id, role="technician", content=payload.message))
        db.commit()
        result = agent_orchestrator.run(db, run_id, operator_message=payload.message)
        db.add(AgentMessage(agent_run_id=run_id, role="assistant", content=_agent_result_message(result)))
        db.commit()
        return result
    except AgentOrchestrationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/manuals/ingest", response_model=ManualIngestResponse)
def ingest_manual_library(force: bool = False, db: Session = Depends(get_db)):
    try:
        return ingest_manuals(db=db, force=force)
    except ManualIndexError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/manuals/search", response_model=list[ManualSearchResult])
def search_manual_library(payload: ManualSearchRequest, db: Session = Depends(get_db)):
    try:
        return hybrid_search(
            payload.query,
            top_k=payload.top_k,
            manufacturer=payload.manufacturer,
            equipment_model=payload.equipment_model,
            equipment_type=payload.equipment_type,
            db=db,
        )
    except (ManualIndexError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/retrieval/health")
def retrieval_health(db: Session = Depends(get_db)):
    from app.models import ManualChunk, ManualDocument

    return {
        "database_backend": db.bind.dialect.name,
        "vector_backend": "pgvector" if db.bind.dialect.name == "postgresql" else "stored-vector fallback",
        "documents": db.query(ManualDocument).count(),
        "chunks": db.query(ManualChunk).count(),
        "reranker_enabled": os.getenv("RERANK_ENABLED", "true").lower() == "true",
    }
