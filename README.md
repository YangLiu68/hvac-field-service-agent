# HVAC Field Service Diagnostic Agent

An evidence-grounded, safety-gated LLM agent for HVAC field-service workflows. The project combines a FastAPI backend, React/TypeScript console, durable tool-calling orchestration, hybrid manual retrieval, technician-in-the-loop measurements, human approval gates, verified case memory, and agent evaluation.

> Portfolio MVP: this repository demonstrates system architecture and evaluation workflows. It is not a certified diagnostic product and must not replace qualified HVAC technicians or manufacturer safety procedures.

The FastAPI service supports the full lifecycle of an HVAC work order: equipment registration, evidence-grounded diagnosis, technician/customer feedback, verified repair outcomes, and an auditable event timeline.

The retrieval layer replaces a transient FAISS-only index with persistent manual documents/chunks and a hybrid retrieval pipeline:

```text
manufacturer/model/equipment metadata filter
→ keyword search (BM25 locally, PostgreSQL full-text search in production)
→ dense semantic search (FastEmbed; pgvector in production)
→ Reciprocal Rank Fusion (RRF)
→ TinyBERT cross-encoder reranking
→ cited manual evidence
```

## Current workflow

1. `POST /equipment` registers an HVAC asset.
2. `POST /jobs` creates a work order linked to the equipment.
3. `POST /jobs/{job_id}/diagnose` retrieves manual evidence and stores a structured diagnostic run and hypothesis.
4. Technician and customer feedback are recorded against the work order.
5. `POST /jobs/{job_id}/close` records the verified cause and repair outcome.
6. `GET /jobs/{job_id}/timeline` returns an auditable lifecycle history.

The legacy `POST /jobs/{job_id}/analyze` route is retained as a deprecated alias. Diagnostic actions now run through the audited Tool Registry, Skill Registry, and Agent Orchestrator described below.

## Field-service intake and dispatch

The backend also supports a minimal customer-to-technician service workflow:

1. Create role-scoped users (`customer`, `technician`, `dispatcher`, or `admin`) through `POST /users`.
2. Create a technician profile using `POST /technicians`, including supported equipment types, certifications, service postal codes, availability, and on-call status.
3. A customer submits `POST /service-requests` with equipment, symptoms, address, preferred time, and urgency.
4. `GET /service-requests/{id}/matches` returns eligible technicians using deterministic, explainable criteria: service area, equipment skill, availability, required safety certification, and on-call coverage.
5. A dispatcher explicitly calls `POST /service-requests/{id}/assignments`; the server creates the existing diagnostic `Job` only after assignment.
6. The assigned technician accepts through `POST /assignments/{id}/accept?technician_profile_id={id}`, then uses the existing Agent workflow on the linked job.

This is deliberately not autonomous dispatch. The matching result is a recommendation; a dispatcher remains responsible for assignment.

## Role-based web console

`frontend/` is a React + TypeScript + Vite console designed for four distinct workflows:

- **Customer**: submit a service request with symptoms, address, urgency, and availability.
- **Dispatch**: review requests, inspect deterministic technician matches, and explicitly assign work.
- **Technician**: accept assigned work, start the safety-gated HVAC Agent, submit measurements, and report field observations to the Agent.
- **Admin**: view operational volume, accepted assignments, diagnostic jobs, and safety posture.

The browser has a local signed-session login. On a fresh database, choose **“First local setup? Create admin”** once, then create dispatcher and technician accounts in Admin. Customers may still submit a first service request without an account. The UI derives the dispatcher and technician identity from the signed-in account; it does not expose ID fields for those actions. For deployment, set a strong `APP_SECRET_KEY` and replace this lightweight local-session layer with your organization’s identity provider.

Run the API and web console in separate terminals:

```bash
# Terminal 1
uvicorn app.main:app --reload

# Terminal 2
cd frontend
npm install
npm run dev
```

Open the Vite URL (normally `http://localhost:5173`). The frontend reads `VITE_API_BASE_URL`, defaulting to `http://127.0.0.1:8000`; copy `frontend/.env.example` to `frontend/.env` only when the API runs elsewhere.

## Stage 4 Tool foundation

The tool layer is deliberately usable without an LLM so each action can be tested independently before adding an orchestrator:

- Read-only tools: `get_job_context`, `get_equipment_profile`, `get_equipment_history`, `search_service_manuals`, `search_error_codes`, and `search_verified_repairs`.
- Workflow tools: `request_technician_measurement`, `record_measurement`, `update_job_status`, `escalate_to_human`, `create_diagnostic_hypothesis`, and `complete_diagnosis`.
- Every tool has Pydantic input/output contracts, an allow-list of job states, and a public JSON schema exposed by `GET /agent/tools`.
- `ToolExecutor` validates arguments and state transitions, enforces the step limit, captures duration and failures, and persists every successful execution in `tool_calls` plus `agent_steps`.
- `AgentRun` and `DiagnosticRun` provide a durable execution context. Measurement requests pause a job in `waiting_for_technician`; recording a result returns it to `diagnosing`.

Create a tool-test run and execute a read-only tool:

```bash
curl -X POST http://127.0.0.1:8000/jobs/1/agent-runs \
  -H 'Content-Type: application/json' -d '{"max_steps":10}'
curl -X POST http://127.0.0.1:8000/agent-runs/1/tools \
  -H 'Content-Type: application/json' \
  -d '{"tool_name":"get_job_context","arguments":{}}'
curl http://127.0.0.1:8000/agent-runs/1/tool-calls
```

These endpoints also allow each tool to be exercised deterministically without invoking the LLM, making tool behavior independently testable.

## Stage 5 HVAC Skills

The deterministic, versioned Skill Registry constrains which tools the Agent Orchestrator may use for each diagnostic workflow.

Included definitions:

- `no_cooling`
- `low_airflow`
- `refrigerant_issue`
- `electrical_fault`
- `thermostat_issue`
- `general_hvac_triage` fallback

Each YAML definition contains triggers, supported equipment types, required context, required checks, allowed tools, human-approval-required actions, escalation conditions, completion criteria, and a maximum step budget. The loader validates every definition against the Stage 4 Tool Registry and rejects unknown tool names at startup.

The deterministic `SkillRouter` uses symptom phrases, error codes, and equipment type to select a workflow. Its confidence is routing confidence only; it is not a diagnosis confidence. A case with no matching signal is routed to `general_hvac_triage` rather than guessed into a specialized workflow.

Skill APIs:

```bash
curl http://127.0.0.1:8000/agent/skills
curl http://127.0.0.1:8000/agent/skills/no_cooling
curl -X POST http://127.0.0.1:8000/jobs/1/skill-route
```

An Agent Run may bind to a `skill_name`; the Stage 4 executor then rejects any Tool outside that Skill's allow-list and caps the run at the Skill's `max_steps`. Stage 6 will use this registry to drive the LLM plan/act/observe loop.

## Stage 6 Agent Orchestrator

Stage 6 connects the Skill Registry and Tool Executor through the OpenAI Responses API function-calling loop. The model receives only the tools allowed by the selected Skill, then the server executes each call, stores the result, and sends a structured `function_call_output` observation back to the next model turn. The server—not the model—remains responsible for argument validation, job-state rules, duplicate-call blocking, step limits, audit records, and terminal states.

Agent state is durable in `agent_runs`: `skill_name`, `current_step`, `last_response_id`, `pending_input`, `final_message`, and failure/completion timestamps. If a measurement request changes the job to `waiting_for_technician`, the loop pauses and can later continue using the saved Responses `previous_response_id` and pending tool output. An escalation pauses the run in `escalated`.

Start or resume an Agent Run:

```bash
curl -X POST http://127.0.0.1:8000/jobs/1/agent-runs \
  -H 'Content-Type: application/json' \
  -d '{"skill_name":"no_cooling","max_steps":20}'
curl -X POST http://127.0.0.1:8000/agent-runs/1/start
curl -X POST http://127.0.0.1:8000/agent-runs/1/continue
```

The orchestrator requires the model to call `complete_diagnosis`; a plain text response without that tool call is treated as an incomplete/failed run rather than a completed repair recommendation. Repeated identical tool calls and runs that exceed their step budget are rejected and audited. High-risk operations are handled by the explicit approval workflow below.

## Stage 7 Human approval and safety gates

Stage 7 adds independent `approval_requests` and `approval_decisions` records. The Tool Executor classifies hazardous operations before execution. Refrigerant service/pressure measurements, live-voltage or electrical-panel access, component replacement, and similar future tools create a high-risk approval request and move the Agent Run to `waiting_for_approval`. A critical safety-bypass action is always prohibited and cannot be approved.

Approval lifecycle:

```text
Agent requests protected Tool
→ approval_requests(status=pending)
→ Agent Run waits
→ supervisor approves or rejects
→ approval_decisions is recorded
→ only an approval endpoint may execute the protected Tool
→ result is stored as the next function_call_output
→ Agent can continue
```

Approval requests expire after 30 minutes. Pending requests can be reviewed and expired requests are marked failed so stale permissions cannot be reused. Direct Tool execution rejects `approved=true`; approval must go through the approval endpoint, preventing callers from self-authorizing a protected action.

```bash
curl http://127.0.0.1:8000/approvals/pending
curl -X POST http://127.0.0.1:8000/approvals/1/approve \
  -H 'Content-Type: application/json' \
  -d '{"decided_by":"Supervisor","reason":"Qualified technician assigned"}'
curl -X POST http://127.0.0.1:8000/approvals/1/reject \
  -H 'Content-Type: application/json' \
  -d '{"decided_by":"Supervisor","reason":"Unsafe site conditions"}'
```

Both approvals and rejections are added to the job Timeline. Approval is an authorization gate, not a diagnosis; the Tool, Skill allow-list, job-state validation, argument validation, and audit logging still run after approval.

## Stage 8 Case memory and feedback loop

Closing a work order now creates a `DiagnosticEvaluation` comparing the latest diagnostic hypotheses with the verified cause. The evaluation records top-1/top-3 cause correctness, technician recommendation acceptance, first-time-fix, return-visit status, and auditable details. Aggregate metrics are available at `GET /evaluation/metrics`.

Only a verified outcome with `approved_for_retrieval=true` is materialized into `case_memories`. The memory stores equipment metadata, sanitized symptoms, verified cause, repair action, and outcome flags; customer names, email addresses, and phone numbers are removed before persistence. Changing approval to false removes the corresponding memory on the next indexing operation.

`search_verified_repairs` now queries this approved memory layer and excludes the current job, so an in-progress case cannot leak its own answer. Results are labeled `verified_historical_case` and are separate from manufacturer-manual evidence; historical cases support similarity but never replace the service manual.

Useful endpoints:

```bash
curl -X POST http://127.0.0.1:8000/jobs/1/case-memory/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"warm air clogged filter","limit":5}'
curl http://127.0.0.1:8000/jobs/1/diagnostic-evaluation
curl http://127.0.0.1:8000/evaluation/metrics
```

The metrics are internal evaluation signals, not production accuracy claims. A larger, reviewed case set is still required before reporting them on a resume.

## Stage 9 Agent Evaluation

`evaluation/agent_cases.jsonl` contains 60 labelled HVAC cases spanning no-cooling, low-airflow, refrigerant, electrical, thermostat, and ambiguous triage workflows. `evaluation/agent_eval.py` reports Skill routing accuracy, Tool sequence accuracy, diagnosis top-1/top-3 accuracy, unsafe recommendation rate, escalation recall/precision, mean/p95 steps, latency, estimated tokens, and optional cost.

Run the offline policy regression:

```bash
python evaluation/agent_eval.py
```

The output is explicitly labelled `offline_policy_proxy`: it does not call an LLM and its latency/token values are local-policy estimates. The current report is a safety and routing regression baseline, not a claim about production Agent quality. For real Agent accuracy, record one JSON line per run using the trace schema below and pass it with `--traces`:

```json
{"case_id":"nc-001","selected_skill":"no_cooling","tool_calls":[{"name":"get_job_context"}],"predicted_causes":["clogged return air filter"],"escalated":false,"latency_ms":120,"usage":{"input_tokens":1000,"output_tokens":200}}
```

When real token usage and current provider rates are available, provide `--input-cost-per-million` and `--output-cost-per-million`; otherwise cost is intentionally reported as `null` instead of inventing a price. OpenAI's evaluation guidance likewise recommends testing the final response separately from tool/program output and comparing calls, turns, tokens, latency, and cost on representative tasks; this evaluator keeps those dimensions separate. [OpenAI Evals](https://developers.openai.com/api/reference/java/resources/evals/methods/create)

## Data model

- `equipment`: manufacturer/model/serial and equipment metadata
- `jobs`: work-order state and technician intake
- `diagnostic_runs`: every AI attempt, including failures and source evidence
- `diagnostic_hypotheses`: ranked causes and next tests
- `tool_calls`: reserved audit records for the later tool-using agent
- `technician_feedback`: acceptance, corrections, and actions taken
- `customer_feedback`: resolution status, rating, and comments
- `verified_outcomes`: final cause/repair and first-time-fix result
- `job_events`: chronological audit trail
- `ai_analyses`: retained for Stage 1 backward compatibility
- `manual_documents`: source metadata, model families, versions, and content hashes
- `manual_chunks`: page-level chunks and persistent embeddings

## Manual metadata

`app/data/manuals/manifest.json` is the authority for manufacturer, equipment type, model family, refrigerant, and document version. A PDF without a manifest record is not indexed. Update the manifest when adding a manual; do not infer production metadata from filenames.

Manufacturer service-manual PDFs are intentionally excluded from this public repository. Add manuals you are authorized to use under `app/data/manuals/`, then update the manifest before ingestion.

## Retrieval API

Index or refresh manuals:

```bash
curl -X POST 'http://127.0.0.1:8000/manuals/ingest?force=true'
```

Search with optional equipment filters:

```bash
curl -X POST http://127.0.0.1:8000/manuals/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"low airflow during cooling","top_k":5,"manufacturer":"Carrier","equipment_model":"FK4B-001","equipment_type":"fan_coil"}'
```

Inspect retrieval status:

```bash
curl http://127.0.0.1:8000/retrieval/health
```

Search responses expose keyword, vector, RRF, and reranker scores for debugging and evaluation.

## PostgreSQL + pgvector

SQLite remains a development fallback. Production retrieval is configured for PostgreSQL full-text search and pgvector with HNSW indexing.

After installing Docker Desktop:

```bash
docker compose up -d postgres
```

Set this in `.env`:

```bash
DATABASE_URL=postgresql+psycopg://hvac:hvac_dev_password@localhost:5432/hvac_agent
```

Restart the API and ingest the manuals. Startup creates the `vector` extension, the `vector(384)` column, a pgvector HNSW index, and a GIN full-text index.

Do not use the development password in a deployed environment.

## Setup

Python 3.11 or 3.12 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Add a valid `OPENAI_API_KEY` to `.env`. Never commit `.env`.

## Run

Run commands from the repository root:

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

Example lifecycle:

```bash
curl -X POST http://127.0.0.1:8000/equipment \
  -H 'Content-Type: application/json' \
  -d '{"manufacturer":"Carrier","model_number":"MODEL-123","equipment_type":"air_handler","serial_number":"SERIAL-001","refrigerant":"R-410A"}'

curl -X POST http://127.0.0.1:8000/jobs \
  -H 'Content-Type: application/json' \
  -d '{"customer_name":"Demo Customer","equipment_id":1,"technician_notes":"AC is not cooling and suction pressure is low","error_code":"E101"}'

curl -X POST http://127.0.0.1:8000/jobs/1/diagnose

curl -X POST http://127.0.0.1:8000/jobs/1/technician-feedback \
  -H 'Content-Type: application/json' \
  -d '{"diagnostic_run_id":1,"technician_name":"Alex Tech","recommendation_accepted":true,"action_taken":"Replaced clogged filter"}'

curl -X POST http://127.0.0.1:8000/jobs/1/customer-feedback \
  -H 'Content-Type: application/json' \
  -d '{"rating":5,"issue_resolved":true,"comments":"Cooling restored"}'

curl -X POST http://127.0.0.1:8000/jobs/1/close \
  -H 'Content-Type: application/json' \
  -d '{"actual_cause":"Clogged return-air filter","repair_action":"Replaced filter and verified airflow","first_time_fix":true,"verified_by":"Alex Tech"}'

curl http://127.0.0.1:8000/jobs/1/timeline
```

## Test

The automated tests mock the paid LLM call:

```bash
pytest -q
```

To test manual retrieval locally:

```bash
python test_rag.py
```

Run the labeled retrieval evaluation:

```bash
python evaluate_retrieval.py --top-k 5
```

The initial five-case smoke benchmark is intentionally small and must not be presented as production accuracy. Expand `evaluation/retrieval_cases.jsonl` to at least 50–100 reviewed cases before using the metric on a resume.

## Error behavior

Malformed PDFs are parsed in an isolated subprocess and skipped so they cannot crash the API. The API returns a clear `503` response when no usable manuals remain, evidence cannot be retrieved, the API key is missing, or the LLM request fails. A missing work order returns `404`.

## Existing database compatibility

On startup, `initialize_database()` creates the Stage 2 tables and adds only missing Stage 2 columns to an existing Stage 1 `jobs` table. Existing jobs and AI analyses are preserved. A pre-migration backup is stored in `backups/`.
