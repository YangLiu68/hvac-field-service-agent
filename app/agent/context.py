from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.models import AgentRun, Job


@dataclass
class ToolContext:
    db: Session
    job: Job
    agent_run: AgentRun
