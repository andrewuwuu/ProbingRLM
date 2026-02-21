import os
import json
import threading
from queue import Queue
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from src.pdf_utils import list_documents, load_document
from src.rlm_handler import RLMHandler
from src.cli_app import _resolve_backend_and_key, DOCUMENT_DIR

load_dotenv()

app = FastAPI(title="ProbingRLM Web Server")

os.makedirs(DOCUMENT_DIR, exist_ok=True)
RLM_LOG_DIR = "rlm_logs"
os.makedirs(RLM_LOG_DIR, exist_ok=True)
_RLM_PATCH_LOCK = threading.Lock()


class QueryRequest(BaseModel):
    documents: List[str]
    model_name: Optional[str] = None
    use_subagents: bool = False
    system_prompt: Optional[str] = None
    query: str
    max_iterations: Optional[int] = None
    max_subagent_calls: Optional[int] = None
    subagent_backend: Optional[str] = None
    subagent_model: Optional[str] = None


class QueryResponse(BaseModel):
    response: str
    metrics: dict


@app.get("/api/documents")
def get_documents():
    docs = list_documents(DOCUMENT_DIR)
    return {"documents": docs}


@app.post("/api/query", response_model=QueryResponse)
def run_query(req: QueryRequest):
    """Old direct completion endpoint without streaming."""
    try:
        backend, api_key = _resolve_backend_and_key()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not req.documents:
        raise HTTPException(status_code=400, detail="No documents selected.")

    combined_sections = []
    for doc in req.documents:
        doc_path = os.path.join(DOCUMENT_DIR, doc)
        if not os.path.isfile(doc_path):
            raise HTTPException(status_code=404, detail=f"Document {doc} not found.")
        text = load_document(doc_path)
        if text:
            combined_sections.append(
                f"===== BEGIN FILE: {doc} =====\n{text}\n===== END FILE: {doc} ====="
            )

    if not combined_sections:
        raise HTTPException(status_code=400, detail="Failed to extract text from selected documents.")

    document_text = "\n\n".join(combined_sections)
    handler = RLMHandler(backend=backend, api_key=api_key, verbose=False, log_dir=RLM_LOG_DIR)
    response = handler.query(
        prompt=req.query,
        context=document_text,
        model=req.model_name,
        use_subagents=req.use_subagents,
        system_prompt=req.system_prompt,
        subagent_backend=req.subagent_backend,
        subagent_model=req.subagent_model,
        max_iterations=req.max_iterations,
        max_subagent_calls=req.max_subagent_calls,
    )
    return QueryResponse(response=response, metrics=handler.last_metrics)


@app.post("/api/query/stream")
def run_query_stream(req: QueryRequest):
    """SSE Streaming endpoint providing live document loading and interaction details."""
    try:
        backend, api_key = _resolve_backend_and_key()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not req.documents:
        raise HTTPException(status_code=400, detail="No documents selected.")

    log_queue = Queue()

    def event_generator():
        combined_sections = []
        for doc in req.documents:
            doc_path = os.path.join(DOCUMENT_DIR, doc)
            if not os.path.isfile(doc_path):
                yield f"data: {json.dumps({'type': 'error', 'detail': f'Document {doc} not found.'})}\n\n"
                return
            
            yield f"data: {json.dumps({'type': 'load_doc', 'doc': doc})}\n\n"
            
            text = load_document(doc_path)
            if text:
                combined_sections.append(
                    f"===== BEGIN FILE: {doc} =====\n{text}\n===== END FILE: {doc} ====="
                )

        if not combined_sections:
            yield f"data: {json.dumps({'type': 'error', 'detail': 'Failed to extract text from selected documents.'})}\n\n"
            return

        document_text = "\n\n".join(combined_sections)
        yield f"data: {json.dumps({'type': 'load_complete', 'num_docs': len(req.documents)})}\n\n"

        handler = RLMHandler(backend=backend, api_key=api_key, verbose=True, log_dir=RLM_LOG_DIR)

        def run_query_in_background():
            final_answer_holder = {"value": None}
            try:
                import src.rlm_handler as rlm_h
                import rlm.core.rlm as rlm_core

                OriginalLogger = rlm_h.RLMLogger
                OriginalVerbose = rlm_core.VerbosePrinter

                class FakeRLMLoggerWrapper(OriginalLogger):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self._web_iteration_count = 0

                    def log_metadata(self, metadata):
                        super().log_metadata(metadata)
                        log_queue.put({"type": "metadata", "data": metadata.to_dict() if hasattr(metadata, "to_dict") else metadata})

                    def log(self, iteration):
                        super().log(iteration)
                        self._web_iteration_count += 1
                        payload = iteration.to_dict() if hasattr(iteration, "to_dict") else iteration
                        if isinstance(payload, dict):
                            payload["iteration"] = self._web_iteration_count
                        log_queue.put({"type": "iteration", "data": payload})

                class FakeVerbosePrinter(OriginalVerbose):
                    def print_metadata(self, metadata):
                        # Keep metadata flow on the logger path; avoid console rendering.
                        return

                    def print_iteration_start(self, iteration):
                        log_queue.put({"type": "verbose", "action": "iteration_start", "iteration": iteration})

                    def print_completion(self, response, iteration_time=None):
                        log_queue.put({"type": "verbose", "action": "completion", "response": str(response)})

                    def print_code_execution(self, code_block):
                        log_queue.put({
                            "type": "verbose", 
                            "action": "code_execution", 
                            "code": str(code_block.code), 
                            "stdout": str(code_block.result.stdout if code_block.result else ""), 
                            "stderr": str(code_block.result.stderr if code_block.result else "")
                        })

                    def print_subcall(self, model, prompt_preview, response_preview, execution_time=None, metadata=None):
                        log_queue.put({
                            "type": "verbose", 
                            "action": "subcall", 
                            "model": model, 
                            "prompt": prompt_preview, 
                            "response": response_preview,
                            "metadata": metadata,
                        })

                    def print_iteration(self, iteration, iteration_num):
                        self.print_iteration_start(iteration_num)
                        self.print_completion(
                            getattr(iteration, "response", ""),
                            getattr(iteration, "iteration_time", None),
                        )
                        for code_block in getattr(iteration, "code_blocks", []) or []:
                            self.print_code_execution(code_block)
                            result = getattr(code_block, "result", None)
                            for call in getattr(result, "rlm_calls", []) or []:
                                self.print_subcall(
                                    model=getattr(call, "root_model", "unknown"),
                                    prompt_preview=str(getattr(call, "prompt", "")),
                                    response_preview=str(getattr(call, "response", "")),
                                    execution_time=getattr(call, "execution_time", None),
                                )

                    def print_final_answer(self, answer):
                        final_answer_holder["value"] = str(answer)
                        log_queue.put(
                            {"type": "verbose", "action": "final_answer", "answer": str(answer)}
                        )

                    def print_summary(self, iterations, total_time, usage):
                        log_queue.put(
                            {
                                "type": "verbose",
                                "action": "summary",
                                "iterations": iterations,
                                "total_time": total_time,
                                "usage": usage,
                            }
                        )

                # rlms does not expose streaming callbacks, so we temporarily patch
                # logger/verbose classes to forward granular events over SSE.
                # Patches are process-global; keep them inside a lock to avoid cross-request races.
                with _RLM_PATCH_LOCK:
                    rlm_h.RLMLogger = FakeRLMLoggerWrapper
                    rlm_core.VerbosePrinter = FakeVerbosePrinter

                    response = handler.query(
                        prompt=req.query,
                        context=document_text,
                        model=req.model_name,
                        use_subagents=req.use_subagents,
                        system_prompt=req.system_prompt,
                        subagent_backend=req.subagent_backend,
                        subagent_model=req.subagent_model,
                        max_iterations=req.max_iterations,
                        max_subagent_calls=req.max_subagent_calls,
                    )

                response_text = str(response) if response is not None else ""
                if (
                    not response_text.strip()
                    or response_text.strip().lower() == "verbose final_answer"
                ) and final_answer_holder["value"]:
                    response_text = final_answer_holder["value"]

                log_queue.put({"type": "done", "response": response_text, "metrics": handler.last_metrics})
            except Exception as e:
                log_queue.put({"type": "error", "detail": str(e)})
            finally:
                if 'OriginalLogger' in locals():
                    rlm_h.RLMLogger = OriginalLogger
                if 'OriginalVerbose' in locals():
                    rlm_core.VerbosePrinter = OriginalVerbose

        # Daemon thread prevents long-running query workers from blocking process shutdown.
        thread = threading.Thread(target=run_query_in_background, daemon=True)
        thread.start()

        while True:
            msg = log_queue.get()
            yield f"data: {json.dumps(msg)}\n\n"
            if msg["type"] in ["done", "error"]:
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")


frontend_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(frontend_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")
