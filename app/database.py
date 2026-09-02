import os

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./field_agent.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)

if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def enable_sqlite_foreign_keys(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(
    autocommit = False,
    autoflush = False,
    bind = engine
)

Base = declarative_base()


def initialize_database():
    """Create Stage 2 tables and safely upgrade the existing Stage 1 jobs table."""
    # Import here to avoid a circular import while models imports Base.
    from app import models  # noqa: F401

    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    if "jobs" not in inspector.get_table_names():
        return

    existing = {column["name"] for column in inspector.get_columns("jobs")}
    additions = {
        "equipment_id": "INTEGER REFERENCES equipment(id)",
        "error_code": "VARCHAR",
        "created_at": "DATETIME",
        "updated_at": "DATETIME",
        "skill_name": "VARCHAR",
    }
    with engine.begin() as connection:
        for name, sql_type in additions.items():
            if name not in existing:
                connection.execute(text(f"ALTER TABLE jobs ADD COLUMN {name} {sql_type}"))
        connection.execute(
            text("UPDATE jobs SET created_at = COALESCE(created_at, CURRENT_TIMESTAMP), "
                 "updated_at = COALESCE(updated_at, CURRENT_TIMESTAMP)")
        )

        if "users" in inspector.get_table_names():
            user_columns = {column["name"] for column in inspector.get_columns("users")}
            if "password_hash" not in user_columns:
                connection.execute(text("ALTER TABLE users ADD COLUMN password_hash VARCHAR NOT NULL DEFAULT ''"))

        if "tool_calls" in inspector.get_table_names():
            tool_call_columns = {column["name"] for column in inspector.get_columns("tool_calls")}
            tool_call_additions = {
                "agent_run_id": "INTEGER REFERENCES agent_runs(id)",
                "agent_step_id": "INTEGER REFERENCES agent_steps(id)",
            }
            for name, sql_type in tool_call_additions.items():
                if name not in tool_call_columns:
                    connection.execute(text(f"ALTER TABLE tool_calls ADD COLUMN {name} {sql_type}"))
        if "agent_runs" in inspector.get_table_names():
            agent_run_columns = {column["name"] for column in inspector.get_columns("agent_runs")}
            agent_run_additions = {
                "skill_name": "VARCHAR",
                "last_response_id": "VARCHAR",
                "pending_input": "TEXT",
                "final_message": "TEXT",
                "pending_approval_call": "TEXT",
            }
            for name, sql_type in agent_run_additions.items():
                if name not in agent_run_columns:
                    connection.execute(text(f"ALTER TABLE agent_runs ADD COLUMN {name} {sql_type}"))

        if engine.dialect.name == "postgresql":
            connection.execute(
                text("ALTER TABLE manual_chunks ADD COLUMN IF NOT EXISTS embedding_vector vector(384)")
            )
            connection.execute(
                text("CREATE INDEX IF NOT EXISTS manual_chunks_embedding_hnsw "
                     "ON manual_chunks USING hnsw (embedding_vector vector_cosine_ops)")
            )
            connection.execute(
                text("CREATE INDEX IF NOT EXISTS manual_chunks_content_fts "
                     "ON manual_chunks USING gin (to_tsvector('english', content))")
            )
