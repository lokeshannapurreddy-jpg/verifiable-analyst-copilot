from src import analytics
import os
import uuid

import pytest
from sqlalchemy import inspect, select


def test_analytics_records_only_aggregate_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(analytics, "_ENGINE", None)
    analytics.record_query("session-a", "sensitive question text", 2, 17)
    summary = analytics.get_analytics()
    assert summary["total_queries"] == 1
    assert summary["average_latency_ms"] == 17
    assert summary["average_evidence_passages"] == 2
    assert "Raw questions" in summary["privacy_note"]
    assert "sensitive question text" not in str(summary)


@pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"),
    reason="Set TEST_DATABASE_URL to run the PostgreSQL integration check",
)
def test_postgresql_analytics_integration(monkeypatch):
    """Exercise schema creation and telemetry round-trip against PostgreSQL."""
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    monkeypatch.setattr(analytics, "_ENGINE", None)
    marker = f"postgres-integration-{uuid.uuid4()}"
    analytics.record_query("integration-session", marker, 1, 5)
    engine = analytics.get_engine()
    assert inspect(engine).has_table("query_events")
    with engine.connect() as connection:
        row = connection.execute(
            select(analytics._EVENTS.c.question_hash).where(
                analytics._EVENTS.c.question_hash
                == analytics.hash_question(marker)
            )
        ).first()
        assert row is not None
    assert "question" not in {column["name"] for column in inspect(engine).get_columns("query_events")}
