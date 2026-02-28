import asyncio
import json
import time
from types import SimpleNamespace

from src.backend import services
from src.backend.schemas import QueryRequest


def _parse_payload(chunk: str) -> dict | None:
    if not chunk.startswith("data: "):
        return None
    return json.loads(chunk[len("data: ") :].strip())


def _collect_stream(req: QueryRequest) -> list[str]:
    """Run the async generator synchronously and collect all chunks."""
    async def _run():
        chunks = []
        async for chunk in services.stream_query_events(req):
            chunks.append(chunk)
        return chunks
    return asyncio.run(_run())


def test_stream_query_events_emits_live_iteration_and_subcall_events(monkeypatch, tmp_path):
    doc_path = tmp_path / "sample.pdf"
    doc_path.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(services, "DOCUMENT_DIR", str(tmp_path))
    monkeypatch.setattr(services, "load_document", lambda _path: "extracted text")
    monkeypatch.setattr(services, "_resolve_backend_credentials", lambda: ("openai", "sk-test"))

    def fake_query(self, **kwargs):
        """Simulate a query that invokes the callbacks and verbose_override."""
        on_iteration_start = kwargs.get("on_iteration_start")
        on_subcall_start = kwargs.get("on_subcall_start")
        on_subcall_complete = kwargs.get("on_subcall_complete")

        if on_iteration_start:
            on_iteration_start(0, 1)
        if on_subcall_start:
            on_subcall_start(1, "openai/gpt-4.1-mini", "sub-prompt")
        if on_subcall_complete:
            on_subcall_complete(1, "openai/gpt-4.1-mini", 1.23, None)

        verbose_override = kwargs.get("verbose_override")
        if verbose_override:
            verbose_override.print_iteration_start(1)
            verbose_override.print_completion("model response", 0.5)
            verbose_override.print_final_answer("final answer")

        return "final answer"

    monkeypatch.setattr(services.RLMHandler, "query", fake_query, raising=True)

    req = QueryRequest(documents=["sample.pdf"], query="what is inside?")
    chunks = _collect_stream(req)
    payloads = [payload for payload in (_parse_payload(chunk) for chunk in chunks) if payload]

    event_types = [payload.get("type") for payload in payloads]
    verbose_actions = [
        payload.get("action")
        for payload in payloads
        if payload.get("type") == "verbose"
    ]

    assert "load_doc" in event_types
    assert "load_complete" in event_types
    assert "model_resolved" in event_types
    assert "done" in event_types
    assert "stream_open" in event_types
    assert "iteration_start_live" in verbose_actions
    assert "subcall_start" in verbose_actions
    assert "subcall_complete" in verbose_actions
    assert "completion" in verbose_actions
    assert "final_answer" in verbose_actions

    done_payload = next(payload for payload in payloads if payload.get("type") == "done")
    assert done_payload.get("response") == "final answer"


def test_stream_query_events_emits_load_timing(monkeypatch, tmp_path):
    """PDF load events should include load_time field."""
    doc_path = tmp_path / "timed.pdf"
    doc_path.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(services, "DOCUMENT_DIR", str(tmp_path))
    monkeypatch.setattr(services, "load_document", lambda _path: "extracted text")
    monkeypatch.setattr(services, "_resolve_backend_credentials", lambda: ("openai", "sk-test"))

    def fake_query(self, **kwargs):
        return "ok"

    monkeypatch.setattr(services.RLMHandler, "query", fake_query, raising=True)

    req = QueryRequest(documents=["timed.pdf"], query="timing test")
    chunks = _collect_stream(req)
    payloads = [payload for payload in (_parse_payload(chunk) for chunk in chunks) if payload]

    load_doc_payload = next(p for p in payloads if p.get("type") == "load_doc")
    assert "load_time" in load_doc_payload
    assert isinstance(load_doc_payload["load_time"], (int, float))
    assert load_doc_payload["load_time"] >= 0


def test_stream_query_events_uses_default_model_when_missing(monkeypatch, tmp_path):
    doc_path = tmp_path / "default-model.pdf"
    doc_path.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(services, "DOCUMENT_DIR", str(tmp_path))
    monkeypatch.setattr(services, "load_document", lambda _path: "extracted text")
    monkeypatch.setattr(services, "_resolve_backend_credentials", lambda: ("openai", "sk-test"))

    observed = {}

    def fake_query(self, **kwargs):
        observed["model"] = kwargs.get("model")
        return "ok"

    monkeypatch.setattr(services.RLMHandler, "query", fake_query, raising=True)

    req = QueryRequest(documents=["default-model.pdf"], query="ping", model_name=None)
    chunks = _collect_stream(req)
    payloads = [payload for payload in (_parse_payload(chunk) for chunk in chunks) if payload]

    assert observed.get("model") == "gpt-4.1-mini"
    model_payload = next(payload for payload in payloads if payload.get("type") == "model_resolved")
    assert model_payload.get("backend") == "openai"
    assert model_payload.get("model") == "gpt-4.1-mini"


def test_stream_query_events_emits_keepalive_while_waiting(monkeypatch, tmp_path):
    doc_path = tmp_path / "slow.pdf"
    doc_path.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(services, "DOCUMENT_DIR", str(tmp_path))
    monkeypatch.setattr(services, "load_document", lambda _path: "extracted text")
    monkeypatch.setattr(services, "_resolve_backend_credentials", lambda: ("openai", "sk-test"))

    def slow_query(self, **kwargs):
        time.sleep(1.8)
        return "slow done"

    monkeypatch.setattr(services.RLMHandler, "query", slow_query, raising=True)

    req = QueryRequest(documents=["slow.pdf"], query="wait")
    chunks = _collect_stream(req)
    keepalives = [chunk for chunk in chunks if chunk.startswith(": keepalive")]
    payloads = [payload for payload in (_parse_payload(chunk) for chunk in chunks) if payload]

    assert keepalives, "Expected at least one keepalive SSE comment while query is running."
    done_payload = next(payload for payload in payloads if payload.get("type") == "done")
    assert done_payload.get("response") == "slow done"


def test_stream_query_events_no_deadlock_without_lock(monkeypatch, tmp_path):
    """Verify concurrent queries complete without deadlock."""
    doc_path = tmp_path / "concurrent.pdf"
    doc_path.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(services, "DOCUMENT_DIR", str(tmp_path))
    monkeypatch.setattr(services, "load_document", lambda _path: "extracted text")
    monkeypatch.setattr(services, "_resolve_backend_credentials", lambda: ("openai", "sk-test"))

    call_count = {"value": 0}
    lock = asyncio.Lock()

    def counting_query(self, **kwargs):
        call_count["value"] += 1
        time.sleep(0.1)
        return "concurrent ok"

    monkeypatch.setattr(services.RLMHandler, "query", counting_query, raising=True)

    req = QueryRequest(documents=["concurrent.pdf"], query="concurrent test")

    async def run_concurrent():
        async def collect():
            chunks = []
            async for chunk in services.stream_query_events(req):
                chunks.append(chunk)
            return chunks

        results = await asyncio.gather(collect(), collect())
        return results

    results = asyncio.run(run_concurrent())

    assert call_count["value"] == 2, f"Expected 2 queries, got {call_count['value']}"

    for i, result in enumerate(results):
        payloads = [p for p in (_parse_payload(c) for c in result) if p]
        done_payload = next((p for p in payloads if p.get("type") == "done"), None)
        assert done_payload is not None, f"Stream {i} did not emit done event"
        assert done_payload.get("response") == "concurrent ok"
