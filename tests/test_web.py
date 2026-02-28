from fastapi.testclient import TestClient
from unittest.mock import patch

from src.web_app import app

client = TestClient(app)

def test_get_documents():
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert isinstance(data["documents"], list)

@patch("src.backend.services._resolve_backend_and_key")
def test_query_no_docs_error(mock_resolve_backend):
    response = client.post(
        "/api/query",
        json={
            "documents": [],
            "query": "Hello world"
        }
    )
    assert response.status_code == 400
    assert "No documents selected" in response.json()["detail"]
    mock_resolve_backend.assert_not_called()

@patch("src.backend.services._resolve_backend_and_key")
def test_query_missing_doc_error(mock_resolve_backend):
    response = client.post(
        "/api/query",
        json={
            "documents": ["non_existent_doc_123.pdf"],
            "query": "Hello world"
        }
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
    mock_resolve_backend.assert_not_called()

@patch("src.backend.services._resolve_backend_and_key")
def test_query_stream_no_docs_error(mock_resolve_backend):
    response = client.post(
        "/api/query/stream",
        json={
            "documents": [],
            "query": "Hello stream"
        }
    )
    assert response.status_code == 400
    assert "No documents selected" in response.json()["detail"]
    mock_resolve_backend.assert_not_called()

@patch("src.backend.services._resolve_backend_and_key")
def test_query_stream_missing_doc_error(mock_resolve_backend):
    with client.stream("POST", "/api/query/stream", json={
        "documents": ["non_existent_doc_456.pdf"],
        "query": "Hello stream"
    }) as response:
        assert response.status_code == 200
        content = response.iter_bytes()
        first_chunk = next(content).decode("utf-8")
        assert "data:" in first_chunk
        assert "error" in first_chunk
        assert "not found" in first_chunk
    mock_resolve_backend.assert_not_called()


def test_query_stream_headers():
    with client.stream(
        "POST",
        "/api/query/stream",
        json={"documents": ["non_existent_header_doc.pdf"], "query": "headers"},
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        assert response.headers.get("cache-control") == "no-cache"
        assert "keep-alive" in response.headers.get("connection", "").lower()
        assert response.headers.get("x-accel-buffering") == "no"
