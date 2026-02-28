import asyncio
import json
import os
import time
import threading
from queue import Empty, Queue
from collections.abc import AsyncGenerator

from fastapi import HTTPException

from src.cli_app import DOCUMENT_DIR, _default_model_for_backend, _resolve_backend_and_key
from src.pdf_utils import list_documents, load_document
from src.rlm_handler import RLMHandler

from .schemas import QueryRequest

RLM_LOG_DIR = "rlm_logs"
os.makedirs(DOCUMENT_DIR, exist_ok=True)
os.makedirs(RLM_LOG_DIR, exist_ok=True)


def list_available_documents() -> list[str]:
    return list_documents(DOCUMENT_DIR)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


def _build_document_context(document_names: list[str]) -> str:
    if not document_names:
        raise HTTPException(status_code=400, detail="No documents selected.")

    combined_sections: list[str] = []
    for document_name in document_names:
        document_path = os.path.join(DOCUMENT_DIR, document_name)
        if not os.path.isfile(document_path):
            raise HTTPException(status_code=404, detail=f"Document {document_name} not found.")

        text = load_document(document_path)
        if text:
            combined_sections.append(
                f"===== BEGIN FILE: {document_name} =====\n{text}\n===== END FILE: {document_name} ====="
            )

    if not combined_sections:
        raise HTTPException(status_code=400, detail="Failed to extract text from selected documents.")

    return "\n\n".join(combined_sections)


def _resolve_backend_credentials() -> tuple[str, str | None]:
    try:
        return _resolve_backend_and_key()
    except ValueError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


def run_query(req: QueryRequest) -> tuple[str, dict]:
    document_text = _build_document_context(req.documents)
    backend, api_key = _resolve_backend_credentials()
    effective_model = (req.model_name or "").strip() or _default_model_for_backend(backend)

    handler = RLMHandler(backend=backend, api_key=api_key, verbose=False, log_dir=RLM_LOG_DIR)
    response = handler.query(
        prompt=req.query,
        context=document_text,
        model=effective_model,
        use_subagents=req.use_subagents,
        system_prompt=req.system_prompt,
        direct_chunking_enabled=req.direct_chunking_enabled,
        direct_chunk_overlap_tokens=req.direct_chunk_overlap_tokens,
        direct_chunk_max_chunks=req.direct_chunk_max_chunks,
        openrouter_middle_out_fallback=req.openrouter_middle_out_fallback,
        subagent_root_compaction_enabled=req.subagent_root_compaction_enabled,
        subagent_compaction_threshold_pct=req.subagent_compaction_threshold_pct,
        subagent_backend=req.subagent_backend,
        subagent_model=req.subagent_model,
        max_iterations=req.max_iterations,
        max_subagent_calls=req.max_subagent_calls,
    )
    return str(response), handler.last_metrics


def _preview(value: object, max_chars: int = 320) -> str:
    text = str(value)
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


class _StreamVerbosePrinter:
    """Drop-in replacement for VerbosePrinter that pushes events to a Queue.

    The library calls methods like print_iteration, print_final_answer, etc.
    on self.verbose in rlm.RLM.completion(). By replacing self.verbose on the
    agent *after* construction, we intercept all verbose output and send it to
    the SSE queue without any monkey-patching.
    """

    def __init__(self, log_queue: Queue) -> None:
        self.enabled = True
        self._queue = log_queue
        self._iteration_count = 0
        self.last_final_answer: str | None = None

    # ── Called by rlm.RLM.completion() ────────────────────────────────

    def print_metadata(self, metadata) -> None:
        payload = metadata.to_dict() if hasattr(metadata, "to_dict") else metadata
        self._queue.put({"type": "verbose", "action": "metadata", "data": payload})

    def print_iteration_start(self, iteration: int) -> None:
        self._queue.put(
            {"type": "verbose", "action": "iteration_start_live", "iteration": iteration}
        )

    def print_completion(self, response, iteration_time=None) -> None:
        self._queue.put(
            {
                "type": "verbose",
                "action": "completion",
                "response": _preview(response, 2000),
            }
        )

    def print_code_execution(self, code_block) -> None:
        result = getattr(code_block, "result", None)
        self._queue.put(
            {
                "type": "verbose",
                "action": "code_execution",
                "code": str(code_block.code),
                "stdout": str(result.stdout if result else ""),
                "stderr": str(result.stderr if result else ""),
            }
        )
        for call in getattr(result, "rlm_calls", []) or []:
            self._queue.put(
                {
                    "type": "verbose",
                    "action": "subcall",
                    "model": str(getattr(call, "root_model", "unknown")),
                    "prompt": _preview(getattr(call, "prompt", "")),
                    "response": _preview(getattr(call, "response", "")),
                    "execution_time": getattr(call, "execution_time", None),
                }
            )

    def print_subcall(self, model, prompt_preview, response_preview,
                      execution_time=None, metadata=None) -> None:
        self._queue.put(
            {
                "type": "verbose",
                "action": "subcall",
                "model": model,
                "prompt": prompt_preview,
                "response": response_preview,
                "execution_time": execution_time,
                "metadata": metadata,
            }
        )

    def print_iteration(self, iteration, iteration_num) -> None:
        self._iteration_count += 1
        self.print_iteration_start(self._iteration_count)
        self.print_completion(
            getattr(iteration, "response", ""),
            getattr(iteration, "iteration_time", None),
        )
        for code_block in getattr(iteration, "code_blocks", []) or []:
            self.print_code_execution(code_block)

    def print_final_answer(self, answer) -> None:
        self.last_final_answer = str(answer)
        self._queue.put(
            {"type": "verbose", "action": "final_answer", "answer": str(answer)}
        )

    def print_summary(self, total_iterations, total_time, usage_summary=None) -> None:
        self._queue.put(
            {
                "type": "verbose",
                "action": "summary",
                "iterations": total_iterations,
                "total_time": total_time,
                "usage": usage_summary,
            }
        )

    # ── No-op stubs for methods we don't need ─────────────────────────

    def print_header(self, *args, **kwargs) -> None:
        return

    def print_budget_exceeded(self, *args, **kwargs) -> None:
        return

    def print_limit_exceeded(self, *args, **kwargs) -> None:
        return

    def print_compaction_status(self, *args, **kwargs) -> None:
        return

    def print_compaction(self, *args, **kwargs) -> None:
        return


async def stream_query_events(req: QueryRequest) -> AsyncGenerator[str, None]:
    """Async generator that yields SSE events for real-time streaming.

    Uses an asyncio event loop to read from a threading.Queue populated by
    a background thread running the RLM query.  This ensures each SSE chunk
    is flushed to the client immediately rather than being buffered.
    """
    yield _sse({"type": "stream_open"})

    if not req.documents:
        yield _sse({"type": "error", "detail": "No documents selected."})
        return

    requested_docs = len(req.documents)
    yield _sse({"type": "load_start", "requested_docs": requested_docs})

    combined_sections: list[str] = []
    for document_name in req.documents:
        document_path = os.path.join(DOCUMENT_DIR, document_name)
        if not os.path.isfile(document_path):
            yield _sse({"type": "error", "detail": f"Document {document_name} not found."})
            return

        load_start = time.perf_counter()
        text = await asyncio.get_event_loop().run_in_executor(
            None, load_document, document_path
        )
        load_time = time.perf_counter() - load_start

        if text:
            combined_sections.append(
                f"===== BEGIN FILE: {document_name} =====\n{text}\n===== END FILE: {document_name} ====="
            )
            yield _sse(
                {
                    "type": "load_doc",
                    "doc": document_name,
                    "loaded_docs": len(combined_sections),
                    "requested_docs": requested_docs,
                    "load_time": round(load_time, 3),
                }
            )
        else:
            yield _sse(
                {
                    "type": "load_doc_empty",
                    "doc": document_name,
                    "loaded_docs": len(combined_sections),
                    "requested_docs": requested_docs,
                    "load_time": round(load_time, 3),
                }
            )

    if not combined_sections:
        yield _sse({"type": "error", "detail": "Failed to extract text from selected documents."})
        return

    yield _sse(
        {
            "type": "load_complete",
            "num_docs": len(combined_sections),
            "requested_docs": requested_docs,
        }
    )

    try:
        backend, api_key = _resolve_backend_credentials()
    except HTTPException as error:
        yield _sse({"type": "error", "detail": str(error.detail)})
        return
    effective_model = (req.model_name or "").strip() or _default_model_for_backend(backend)
    yield _sse({"type": "model_resolved", "backend": backend, "model": effective_model})

    handler = RLMHandler(backend=backend, api_key=api_key, verbose=False, log_dir=RLM_LOG_DIR)
    log_queue: Queue = Queue()

    def run_query_in_background() -> None:
        try:
            def on_subcall_start(depth: int, model: str, prompt_preview: str) -> None:
                log_queue.put(
                    {
                        "type": "verbose",
                        "action": "subcall_start",
                        "model": model,
                        "prompt": _preview(prompt_preview),
                        "depth": depth,
                    }
                )

            def on_subcall_complete(depth: int, model: str, duration: float, error: str | None) -> None:
                payload = {
                    "type": "verbose",
                    "action": "subcall_complete",
                    "model": model,
                    "execution_time": duration,
                    "depth": depth,
                }
                if error:
                    payload["error"] = error
                else:
                    payload["response"] = "(completed)"
                log_queue.put(payload)

            def on_iteration_start(depth: int, iteration_num: int) -> None:
                log_queue.put(
                    {
                        "type": "verbose",
                        "action": "iteration_start_live",
                        "iteration": iteration_num,
                        "depth": depth,
                    }
                )

            def on_iteration_complete(depth: int, iteration_num: int, duration: float) -> None:
                log_queue.put(
                    {
                        "type": "verbose",
                        "action": "iteration_complete",
                        "iteration": iteration_num,
                        "duration": duration,
                        "depth": depth,
                    }
                )

            stream_verbose = _StreamVerbosePrinter(log_queue)

            response = handler.query(
                prompt=req.query,
                context="\n\n".join(combined_sections),
                model=effective_model,
                use_subagents=req.use_subagents,
                system_prompt=req.system_prompt,
                direct_chunking_enabled=req.direct_chunking_enabled,
                direct_chunk_overlap_tokens=req.direct_chunk_overlap_tokens,
                direct_chunk_max_chunks=req.direct_chunk_max_chunks,
                openrouter_middle_out_fallback=req.openrouter_middle_out_fallback,
                subagent_root_compaction_enabled=req.subagent_root_compaction_enabled,
                subagent_compaction_threshold_pct=req.subagent_compaction_threshold_pct,
                subagent_backend=req.subagent_backend,
                subagent_model=req.subagent_model,
                max_iterations=req.max_iterations,
                max_subagent_calls=req.max_subagent_calls,
                on_subcall_start=on_subcall_start,
                on_subcall_complete=on_subcall_complete,
                on_iteration_start=on_iteration_start,
                on_iteration_complete=on_iteration_complete,
                verbose_override=stream_verbose,
            )

            response_text = str(response) if response is not None else ""
            if (
                not response_text.strip()
                or response_text.strip().lower() == "verbose final_answer"
            ) and stream_verbose.last_final_answer:
                response_text = stream_verbose.last_final_answer

            log_queue.put(
                {"type": "done", "response": response_text, "metrics": handler.last_metrics}
            )
        except Exception as error:
            log_queue.put({"type": "error", "detail": str(error)})

    thread = threading.Thread(target=run_query_in_background, daemon=True)
    thread.start()

    # Read from the threading.Queue using run_in_executor so the async
    # event loop is never blocked.  Each yielded chunk is sent to the
    # client immediately by the ASGI server.
    loop = asyncio.get_event_loop()
    while True:
        try:
            message = await loop.run_in_executor(
                None, lambda: log_queue.get(timeout=1.5)
            )
        except Empty:
            yield ": keepalive\n\n"
            continue
        yield _sse(message)
        if message["type"] in {"done", "error"}:
            break
