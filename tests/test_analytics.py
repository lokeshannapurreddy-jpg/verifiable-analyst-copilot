from src import analytics
import os
import uuid

import pytest
from sqlalchemy import delete, inspect, select
from sqlalchemy.engine import make_url


_SAFE_TEST_DATABASE_HOSTS = {"localhost", "127.0.0.1", "::1", "database"}
_SAFE_TEST_DATABASE_NAME = "analyst_test"


def _validated_test_database_url(value: str):
    """Reject non-local or non-disposable URLs before any database connection."""
    message = (
        "TEST_DATABASE_URL must target a local PostgreSQL host and the "
        "dedicated analyst_test database"
    )
    try:
        url = make_url(value)
    except Exception:
        raise ValueError(message) from None
    if (
        url.get_backend_name() != "postgresql"
        or url.host not in _SAFE_TEST_DATABASE_HOSTS
        or url.database != _SAFE_TEST_DATABASE_NAME
        or bool(url.query)
    ):
        raise ValueError(message)
    return url


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


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+psycopg://tester:unused@db.example/analyst_test",
        "postgresql+psycopg://tester:unused@database/analyst",
        "postgresql+psycopg://tester:unused@database/analyst_test?hostaddr=203.0.113.1",
        "sqlite:///analyst_test.db",
    ],
)
def test_postgresql_integration_rejects_unsafe_database_urls(url):
    with pytest.raises(ValueError, match="local PostgreSQL host"):
        _validated_test_database_url(url)


@pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"),
    reason="Set TEST_DATABASE_URL to run the PostgreSQL integration check",
)
def test_postgresql_analytics_integration(monkeypatch):
    """Exercise schema creation and telemetry round-trip against PostgreSQL."""
    test_url = _validated_test_database_url(os.environ["TEST_DATABASE_URL"])
    monkeypatch.setenv(
        "DATABASE_URL", test_url.render_as_string(hide_password=False)
    )
    monkeypatch.setattr(analytics, "_ENGINE", None)
    marker = f"postgres-integration-{uuid.uuid4()}"
    try:
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
        assert "question" not in {
            column["name"] for column in inspect(engine).get_columns("query_events")
        }
    finally:
        engine = analytics._ENGINE
        if engine is not None:
            try:
                with engine.begin() as connection:
                    connection.execute(
                        delete(analytics._EVENTS).where(
                            analytics._EVENTS.c.question_hash
                            == analytics.hash_question(marker)
                        )
                    )
            except Exception:
                pytest.fail("Could not clean up the PostgreSQL test marker row")
            finally:
                engine.dispose()
                analytics._ENGINE = None
