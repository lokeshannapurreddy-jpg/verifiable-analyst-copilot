"""Privacy-conscious query telemetry and basic index analytics.

Uses DATABASE_URL when configured; defaults to a local SQLite database.
Raw questions and answer text are deliberately not persisted.
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from sqlalchemy import (
    Column, DateTime, Integer, MetaData, String, Table, create_engine,
    func, select, text,
)
from sqlalchemy.engine import Engine

_METADATA = MetaData()
_EVENTS = Table(
    "query_events",
    _METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("session_hash", String(64), nullable=False),
    Column("question_hash", String(64), nullable=False),
    Column("question_length", Integer, nullable=False),
    Column("result_count", Integer, nullable=False),
    Column("latency_ms", Integer, nullable=False),
    Column("status", String(20), nullable=False),
)
_LOCK = Lock()
_ENGINE: Engine | None = None


def get_engine() -> Engine:
    """Create one engine, using SQLite locally or DATABASE_URL in deployments."""
    global _ENGINE
    if _ENGINE is None:
        url = os.getenv("DATABASE_URL")
        if not url:
            data_dir = Path(os.getenv("APP_DATA_DIR", "data"))
            data_dir.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{(data_dir / 'analytics.db').resolve().as_posix()}"
        _ENGINE = create_engine(url, pool_pre_ping=True, future=True)
        _METADATA.create_all(_ENGINE)
    return _ENGINE


def hash_question(question: str) -> str:
    """Return the stored digest for checking telemetry without retaining text."""
    return hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()


def record_query(session_id: str, question: str, result_count: int,
                 latency_ms: int, status: str = "ok") -> None:
    """Persist minimal metadata; never persist raw questions or answers."""
    digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    row = {
        "created_at": datetime.now(timezone.utc),
        "session_hash": digest(session_id or "anonymous"),
        "question_hash": hash_question(question),
        "question_length": len(question),
        "result_count": int(result_count),
        "latency_ms": max(0, int(latency_ms)),
        "status": status[:20],
    }
    with _LOCK, get_engine().begin() as connection:
        connection.execute(_EVENTS.insert().values(**row))


def get_analytics() -> dict:
    """Return aggregate service usage, not raw user content."""
    with get_engine().connect() as connection:
        row = connection.execute(
            select(
                func.count(_EVENTS.c.id),
                func.coalesce(func.avg(_EVENTS.c.latency_ms), 0),
                func.coalesce(func.avg(_EVENTS.c.result_count), 0),
            )
        ).one()
        status_rows = connection.execute(
            select(_EVENTS.c.status, func.count(_EVENTS.c.id))
            .group_by(_EVENTS.c.status)
        ).all()
    return {
        "total_queries": int(row[0] or 0),
        "average_latency_ms": round(float(row[1] or 0), 2),
        "average_evidence_passages": round(float(row[2] or 0), 2),
        "queries_by_status": {str(status): int(count) for status, count in status_rows},
        "stored_fields": [
            "timestamp", "hashed session id", "hashed question",
            "question length", "evidence count", "latency", "status"
        ],
        "privacy_note": "Raw questions and generated answers are not stored.",
    }
