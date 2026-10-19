from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import api


def test_ask_request_rejects_invalid_bounds():
    with pytest.raises(ValidationError):
        api.AskRequest(question="x")
    with pytest.raises(ValidationError):
        api.AskRequest(question="A valid question", top_k=11)
    with pytest.raises(ValidationError):
        api.AskRequest(question="A valid question", session_id="")


def test_ask_request_generates_isolated_default_session_ids():
    first = api.AskRequest(question="A valid research question")
    second = api.AskRequest(question="A valid research question")

    assert first.session_id != second.session_id
    assert len(first.session_id) == 32


def test_ask_returns_answer_when_analytics_fails(monkeypatch):
    monkeypatch.setattr(api, "retriever", SimpleNamespace(chunks=[object()]))
    monkeypatch.setattr(api, "_get_memory", lambda session_id: None)
    monkeypatch.setattr(
        api,
        "run_agent",
        lambda *args, **kwargs: {
            "answer": "Evidence answer [1].",
            "evidence": [{"page": 2}],
            "tool": "evidence_retrieval",
            "trace": ["retrieval"],
        },
    )

    def fail_analytics(*args, **kwargs):
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr(api, "record_query", fail_analytics)
    response = api.ask(api.AskRequest(question="A valid research question"))

    assert response.answer == "Evidence answer [1]."
    assert response.evidence_count == 1


def test_ask_reports_unavailable_index(monkeypatch):
    monkeypatch.setattr(api, "retriever", SimpleNamespace(chunks=[]))
    monkeypatch.setattr(api, "record_query", lambda *args, **kwargs: None)

    with pytest.raises(HTTPException) as error:
        api.ask(api.AskRequest(question="A valid research question"))

    assert error.value.status_code == 503


def test_reload_requires_admin_key_and_checks_supplied_key(monkeypatch):
    monkeypatch.delenv("RELOAD_ADMIN_KEY", raising=False)
    with pytest.raises(HTTPException) as error:
        api.reload_documents()
    assert error.value.status_code == 503

    monkeypatch.setenv("RELOAD_ADMIN_KEY", "test-only-key")
    with pytest.raises(HTTPException) as error:
        api.reload_documents(x_admin_key="wrong-key")
    assert error.value.status_code == 403

    monkeypatch.setattr(api, "load_documents", lambda: 2)
    assert api.reload_documents(x_admin_key="test-only-key") == {
        "status": "success",
        "indexed_chunks": 2,
    }


def test_analytics_endpoint_returns_safe_error_on_storage_failure(monkeypatch):
    monkeypatch.setattr(
        api, "get_analytics", lambda: (_ for _ in ()).throw(RuntimeError())
    )

    with pytest.raises(HTTPException) as error:
        api.analytics()

    assert error.value.status_code == 503


def test_health_does_not_expose_startup_exception_details(monkeypatch):
    monkeypatch.setattr(api, "retriever", SimpleNamespace(chunks=[]))

    def fail_loading():
        raise FileNotFoundError("C:\\private\\reports\\secret.pdf")

    monkeypatch.setattr(api, "index_error", None)
    monkeypatch.setattr(api, "load_documents", fail_loading)
    api.startup_index()

    response = api.health()

    assert response["index_error"] == api._SAFE_INDEX_ERROR
    assert "C:\\private" not in str(response)


def test_reload_does_not_expose_local_paths(monkeypatch):
    monkeypatch.setenv("RELOAD_ADMIN_KEY", "test-only-key")

    def fail_loading():
        raise FileNotFoundError("C:\\private\\reports\\secret.pdf")

    monkeypatch.setattr(api, "load_documents", fail_loading)

    with pytest.raises(HTTPException) as error:
        api.reload_documents(x_admin_key="test-only-key")

    assert error.value.status_code == 503
    assert "C:\\private" not in error.value.detail


def test_session_memory_is_isolated_by_session_id():
    first_id = f"test-{uuid4()}"
    second_id = f"test-{uuid4()}"

    assert api._get_memory(first_id) is not api._get_memory(second_id)
