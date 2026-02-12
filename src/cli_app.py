import multiprocessing as mp
import os
import queue
import time
from typing import Any, Callable

from dotenv import load_dotenv

from src.cli_config import (
    _load_tools_from_env,
    _parse_bool_env,
    _parse_json_object_env,
    _parse_positive_float_env,
    _parse_positive_int_env,
    _read_env,
    _resolve_backend_and_key,
    _resolve_root_model_default,
    _resolve_subagent_config as _resolve_subagent_config_impl,
)
from src.cli_metrics import (
    _merge_run_metrics_with_fallback,
    _merge_run_metrics_with_retry,
    _parse_context_limit_error,
    _print_query_metrics as _print_query_metrics_impl,
    _response_has_pdf_page_na_citation,
    _should_use_research_direct_fallback,
    _with_openrouter_middle_out,
)
from src.cli_prompting import (
    _build_runtime_prompt_vars,
    _get_prompt_section,
    _normalize_output_mode,
    _output_template_for_mode,
    _prompt_output_mode,
    _render_prompt_section,
    _resolve_default_output_mode as _resolve_default_output_mode_impl,
    _select_documents,
    default_prompt_query as _default_prompt_query_impl,
    default_should_continue_session,
)
from src.output_utils import save_markdown_response, save_pdf_response
from src.pdf_utils import list_documents, load_document
from src.prompt_loader import load_prompts, parse_prompt_variables
from src.rlm_handler import RLMHandler

DOCUMENT_DIR = "embed-docs"
PROMPTS_FILE = "prompts.md"
RESPONSE_DIR = "response"
FORCE_SUBAGENT_QUERY_INSTRUCTION = (
    "Subagent requirement: you must call llm_query(...) or llm_query_batched(...) at least "
    "once before SUBMIT(...). If no subagent call is made, the answer is invalid."
)


def default_prompt_yes_no(message: str, default: bool = False) -> bool:
    raw = input(message).strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes"}


def _resolve_default_output_mode(prompt_vars: dict[str, str]) -> str:
    return _resolve_default_output_mode_impl(
        prompt_vars=prompt_vars,
        env_output_mode=_read_env("DSPY_OUTPUT_MODE"),
    )


def default_prompt_query(
    default_query: str | None,
    prompt_yes_no: Callable[[str, bool], bool] = default_prompt_yes_no,
) -> str:
    return _default_prompt_query_impl(default_query, prompt_yes_no)


def _resolve_subagent_config(
    use_subagents: bool,
    root_backend: str,
    root_model: str,
    prompt_yes_no: Callable[[str, bool], bool] = default_prompt_yes_no,
) -> tuple[str | None, str | None]:
    return _resolve_subagent_config_impl(
        use_subagents=use_subagents,
        root_backend=root_backend,
        root_model=root_model,
        prompt_yes_no=prompt_yes_no,
    )


def _preferred_mp_start_method() -> str:
    available = mp.get_all_start_methods()
    if "fork" in available:
        return "fork"
    return "spawn"


def _query_worker(
    backend: str,
    api_key: str | None,
    verbose: bool,
    query_kwargs: dict,
    out_queue,
) -> None:
    worker_handler = RLMHandler(backend=backend, api_key=api_key, verbose=verbose)
    response = worker_handler.query(**query_kwargs)
    payload = {
        "response": response,
        "metrics": worker_handler.last_metrics,
    }
    out_queue.put(payload)


def _run_query_with_live_status(
    handler: RLMHandler,
    query_kwargs: dict,
    timeout_seconds: float | None = None,
) -> str:
    result: dict[str, dict | str] = {}
    timed_out = False
    ctx = mp.get_context(_preferred_mp_start_method())
    out_queue: mp.Queue = ctx.Queue()

    process = ctx.Process(
        target=_query_worker,
        args=(
            handler.backend,
            handler.api_key,
            handler.verbose,
            query_kwargs,
            out_queue,
        ),
        daemon=True,
    )

    spinner = ["|", "/", "-", "\\"]
    start_time = time.perf_counter()
    spinner_index = 0

    process.start()
    try:
        while True:
            elapsed = time.perf_counter() - start_time
            if timeout_seconds is not None and elapsed >= timeout_seconds:
                timed_out = True
                break

            try:
                result = out_queue.get(timeout=0.2)
                break
            except queue.Empty:
                pass

            if not process.is_alive():
                grace_deadline = time.perf_counter() + 1.5
                while time.perf_counter() < grace_deadline:
                    try:
                        result = out_queue.get(timeout=0.1)
                        break
                    except queue.Empty:
                        pass
                break

            elapsed = time.perf_counter() - start_time
            print(
                f"\rProcessing {spinner[spinner_index % len(spinner)]}  elapsed: {elapsed:5.1f}s",
                end="",
                flush=True,
            )
            spinner_index += 1
    except KeyboardInterrupt:
        print("\nCancelling query...")
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        handler.last_metrics = {"cancelled": True}
        return "Query cancelled by user."
    finally:
        if process.is_alive():
            if timed_out:
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
            else:
                process.join(timeout=1)

    if timed_out:
        handler.last_metrics = {
            "error": "query_timeout",
            "timeout_seconds": timeout_seconds,
        }
        try:
            out_queue.close()
        except Exception:
            pass
        return f"Error querying DSPy: timed out after {timeout_seconds:.1f}s"

    if process.is_alive():
        process.terminate()
        process.join(timeout=2)

    if process.exitcode is None:
        process.join(timeout=1)

    try:
        if not result:
            try:
                result = out_queue.get_nowait()
            except queue.Empty:
                pass
    finally:
        try:
            out_queue.close()
        except Exception:
            pass

    elapsed = time.perf_counter() - start_time
    print(f"\rProcessing complete. elapsed: {elapsed:5.1f}s{' ' * 12}")

    if process.exitcode != 0:
        handler.last_metrics = {"error": f"worker_exit_code_{process.exitcode}"}
        return f"Error querying DSPy: worker exited with code {process.exitcode}"

    if not result:
        handler.last_metrics = {"error": "missing_worker_output"}
        return "Error querying DSPy: worker produced no output"

    response = str(result.get("response", ""))
    metrics = result.get("metrics", {})
    if isinstance(metrics, dict):
        handler.last_metrics = metrics
    else:
        handler.last_metrics = {}
    return response


def _print_query_metrics(metrics: dict) -> None:
    verbose_metrics = bool(_parse_bool_env("DSPY_VERBOSE_METRICS"))
    _print_query_metrics_impl(metrics=metrics, verbose_metrics=verbose_metrics)


def main(
    prompt_yes_no: Callable[[str, bool], bool] = default_prompt_yes_no,
    prompt_query: Callable[
        [str | None, Callable[[str, bool], bool]], str
    ] = default_prompt_query,
    should_continue_session: Callable[[bool, int], bool] = default_should_continue_session,
) -> None:
    load_dotenv()

    try:
        backend, api_key = _resolve_backend_and_key()
    except ValueError as error:
        print(f"Error: {error}")
        print("You can copy .env.example to .env and fill in your keys.")
        return

    print(f"=== DSPy Document Retrieval (Backend: {backend}) ===")

    created_document_dir = False
    if not os.path.isdir(DOCUMENT_DIR):
        os.makedirs(DOCUMENT_DIR, exist_ok=True)
        created_document_dir = True

    docs = list_documents(DOCUMENT_DIR)
    if not docs:
        if created_document_dir:
            print(f"Created '{DOCUMENT_DIR}/'.")
        print(f"No supported documents found in '{DOCUMENT_DIR}/'.")
        print("Place files like .pdf, .docx, .txt, .md (or other text files) there and try again.")
        return

    print(f"\nAvailable documents in '{DOCUMENT_DIR}/':")
    for i, doc in enumerate(docs, start=1):
        print(f"{i}. {doc}")
    print("a. All documents")

    try:
        selected_docs = _select_documents(
            docs,
            "\nSelect a document number, 'a' for all, or 'q' to quit: ",
        )
    except KeyboardInterrupt:
        return

    default_model = _resolve_root_model_default(backend)
    model_name = input(f"Enter model name (default: {default_model}): ").strip() or default_model
    model_name = RLMHandler.canonical_model_for_backend(backend, model_name)

    max_iterations = _parse_positive_int_env("DSPY_RLM_MAX_ITERATIONS")
    max_llm_calls = _parse_positive_int_env("DSPY_RLM_MAX_LLM_CALLS")
    max_output_chars = _parse_positive_int_env("DSPY_RLM_MAX_OUTPUT_CHARS")
    max_depth = _parse_positive_int_env("DSPY_RLM_MAX_DEPTH")
    subagent_prefetch_calls = _parse_positive_int_env("DSPY_SUBAGENT_PREFETCH_CALLS") or 0
    rlm_signature = _read_env("DSPY_RLM_SIGNATURE") or None
    custom_prompt = (os.getenv("DSPY_CUSTOM_PROMPT") or "").strip() or None
    query_timeout_seconds = _parse_positive_float_env("DSPY_QUERY_TIMEOUT_SECONDS")

    require_subagent_call = bool(_parse_bool_env("DSPY_REQUIRE_SUBAGENT_CALL"))
    require_subagent_call_retry_once = bool(
        _parse_bool_env("DSPY_REQUIRE_SUBAGENT_CALL_RETRY_ONCE")
    )
    enforce_pdf_page_citations = _parse_bool_env("DSPY_ENFORCE_PDF_PAGE_CITATIONS")

    allow_research_direct_fallback = _parse_bool_env("DSPY_RESEARCH_ALLOW_DIRECT_FALLBACK")
    if allow_research_direct_fallback is None:
        allow_research_direct_fallback = True

    citation_repair_direct_mode = _parse_bool_env("DSPY_CITATION_REPAIR_DIRECT_MODE")
    if citation_repair_direct_mode is None:
        citation_repair_direct_mode = False

    openrouter_auto_middle_out = _parse_bool_env("DSPY_OPENROUTER_AUTO_MIDDLE_OUT")
    if openrouter_auto_middle_out is None:
        openrouter_auto_middle_out = True

    root_lm_kwargs = _parse_json_object_env("DSPY_LM_KWARGS")
    subagent_lm_kwargs = _parse_json_object_env("DSPY_SUBAGENT_LM_KWARGS")
    rlm_interpreter = _parse_bool_env("DSPY_RLM_INTERPRETER")
    rlm_tools = _load_tools_from_env("DSPY_RLM_TOOLS")

    use_subagents = prompt_yes_no("Use DSPy RLM subagents? (y/N): ", default=False)
    subagent_backend: str | None = None
    subagent_model: str | None = None
    if use_subagents:
        subagent_backend, subagent_model = _resolve_subagent_config(
            use_subagents=use_subagents,
            root_backend=backend,
            root_model=model_name,
            prompt_yes_no=prompt_yes_no,
        )
    if require_subagent_call and not (subagent_backend and subagent_model):
        print(
            "Warning: strict subagent-call checks require a dedicated subagent backend/model. "
            "Disabling strict subagent-call requirement for this run."
        )
        require_subagent_call = False

    selected_pdf_docs = {doc.lower() for doc in selected_docs if doc.lower().endswith(".pdf")}

    combined_sections: list[str] = []
    failed_docs: list[str] = []
    for selected_doc in selected_docs:
        doc_path = os.path.join(DOCUMENT_DIR, selected_doc)
        print(f"\nLoading {selected_doc}...")
        doc_text = load_document(doc_path)
        if not doc_text:
            failed_docs.append(selected_doc)
            continue
        combined_sections.append(
            f"===== BEGIN FILE: {selected_doc} =====\n{doc_text}\n===== END FILE: {selected_doc} ====="
        )

    if not combined_sections:
        print("Failed to extract text from selected documents.")
        if failed_docs:
            print(f"Failed files: {', '.join(failed_docs)}")
        return

    if failed_docs:
        print(f"Warning: failed to read {len(failed_docs)} file(s): {', '.join(failed_docs)}")

    document_text = "\n\n".join(combined_sections)
    print(
        f"Loaded {len(combined_sections)} document(s) into context. Total extracted characters: {len(document_text)}."
    )

    prompts_cache: dict[str, str] = {}
    prompt_vars: dict[str, str] = {}
    file_custom_prompt_template: str | None = None
    file_rlm_signature_template: str | None = None
    if os.path.exists(PROMPTS_FILE) and prompt_yes_no(
        f"\nLoad prompts from {PROMPTS_FILE}? (y/N): ", default=False
    ):
        prompts_cache = load_prompts(PROMPTS_FILE)
        variables_section = _get_prompt_section(prompts_cache, "Variables")
        file_custom_prompt_template = (
            _get_prompt_section(prompts_cache, "Custom Prompt")
            or _get_prompt_section(prompts_cache, "Guidance")
        )
        file_rlm_signature_template = (
            _get_prompt_section(prompts_cache, "RLM Signature")
            or _get_prompt_section(prompts_cache, "Signature")
        )
        prompt_vars.update(parse_prompt_variables(variables_section))
        print(f"Loaded prompts from {PROMPTS_FILE}.")

    allow_follow_ups = prompt_yes_no(
        "\nAllow follow-up queries in this run? (Y/n): ",
        default=True,
    )
    selected_output_mode = _prompt_output_mode(_resolve_default_output_mode(prompt_vars))
    output_template = prompt_vars.get("output_template") or _output_template_for_mode(
        selected_output_mode
    )
    prompt_vars["output_mode"] = selected_output_mode
    prompt_vars["output_template"] = output_template

    handler = RLMHandler(backend=backend, api_key=api_key, verbose=False)

    answered_queries = 0
    while True:
        if not should_continue_session(allow_follow_ups, answered_queries):
            print("\nOne-time session complete. Exiting.")
            break

        runtime_prompt_vars = _build_runtime_prompt_vars(
            backend=backend,
            model_name=model_name,
            selected_docs=selected_docs,
            document_text=document_text,
            query_index=answered_queries + 1,
            output_mode=selected_output_mode,
            output_template=output_template,
        )
        default_query_template = _get_prompt_section(prompts_cache, "Query")
        default_query = _render_prompt_section(
            section_text=default_query_template,
            section_name="Query",
            runtime_vars=runtime_prompt_vars,
            prompt_vars=prompt_vars,
        )
        query = prompt_query(default_query, prompt_yes_no)
        if query.lower() == "q":
            break

        runtime_prompt_vars["query"] = query
        system_prompt = _render_prompt_section(
            section_text=_get_prompt_section(prompts_cache, "System"),
            section_name="System",
            runtime_vars=runtime_prompt_vars,
            prompt_vars=prompt_vars,
        )
        effective_custom_prompt = custom_prompt or _render_prompt_section(
            section_text=file_custom_prompt_template,
            section_name="Custom Prompt",
            runtime_vars=runtime_prompt_vars,
            prompt_vars=prompt_vars,
        )
        effective_rlm_signature = rlm_signature or _render_prompt_section(
            section_text=file_rlm_signature_template,
            section_name="RLM Signature",
            runtime_vars=runtime_prompt_vars,
            prompt_vars=prompt_vars,
        )

        print("\nProcessing...")

        retry_attempted = False
        middle_out_retry_attempted = False
        middle_out_primary_metrics: dict | None = None
        retry_suffix = FORCE_SUBAGENT_QUERY_INSTRUCTION
        response = ""
        run_custom_prompt = effective_custom_prompt
        run_root_lm_kwargs = dict(root_lm_kwargs) if root_lm_kwargs else None
        run_subagent_lm_kwargs = dict(subagent_lm_kwargs) if subagent_lm_kwargs else None

        while True:
            query_kwargs = dict(
                prompt=query,
                context=document_text,
                model=model_name,
                use_subagents=use_subagents,
                system_prompt=system_prompt,
                subagent_backend=subagent_backend,
                subagent_model=subagent_model,
                max_iterations=max_iterations,
                max_subagent_calls=max_llm_calls,
                max_llm_calls=max_llm_calls,
                max_output_chars=max_output_chars,
                max_depth=max_depth,
                custom_prompt=run_custom_prompt,
                rlm_signature=effective_rlm_signature,
                root_lm_kwargs=run_root_lm_kwargs,
                subagent_lm_kwargs=run_subagent_lm_kwargs,
                rlm_tools=rlm_tools,
                rlm_interpreter=rlm_interpreter,
                require_subagent_call=require_subagent_call,
                subagent_prefetch_calls=(
                    subagent_prefetch_calls
                    if use_subagents and subagent_backend and subagent_model
                    else 0
                ),
            )
            response = _run_query_with_live_status(
                handler,
                query_kwargs,
                timeout_seconds=query_timeout_seconds,
            )

            if not (
                require_subagent_call
                and require_subagent_call_retry_once
                and not retry_attempted
                and handler.last_metrics.get("error") == "required_subagent_call_missing"
            ):
                break

            retry_attempted = True
            print(
                "Strict subagent-call check failed. Retrying once with forced llm_query instruction."
            )
            if run_custom_prompt:
                run_custom_prompt = f"{run_custom_prompt}\n\n{retry_suffix}"
            else:
                run_custom_prompt = retry_suffix

        while True:
            context_limit = _parse_context_limit_error(response)
            if not (
                openrouter_auto_middle_out
                and not middle_out_retry_attempted
                and context_limit is not None
                and (
                    backend == "openrouter"
                    or (use_subagents and subagent_backend == "openrouter")
                )
            ):
                break

            middle_out_retry_attempted = True
            middle_out_primary_metrics = dict(handler.last_metrics)
            max_tokens, requested_tokens = context_limit
            print(
                "Context overflow detected "
                f"({requested_tokens} > {max_tokens} tokens). "
                "Retrying once with OpenRouter middle-out transform."
            )
            if backend == "openrouter":
                run_root_lm_kwargs = _with_openrouter_middle_out(run_root_lm_kwargs)
            if use_subagents and subagent_backend == "openrouter":
                run_subagent_lm_kwargs = _with_openrouter_middle_out(run_subagent_lm_kwargs)

            response = _run_query_with_live_status(
                handler,
                dict(
                    query_kwargs,
                    root_lm_kwargs=run_root_lm_kwargs,
                    subagent_lm_kwargs=run_subagent_lm_kwargs,
                ),
                timeout_seconds=query_timeout_seconds,
            )

        if middle_out_primary_metrics is not None:
            handler.last_metrics = _merge_run_metrics_with_retry(
                primary=middle_out_primary_metrics,
                retry=dict(handler.last_metrics),
                retry_label="middle_out",
            )

        primary_metrics = dict(handler.last_metrics)
        if allow_research_direct_fallback and _should_use_research_direct_fallback(
            output_mode=selected_output_mode,
            use_subagents=use_subagents,
            response=response,
            metrics=primary_metrics,
        ):
            print("Research fallback: retrying with direct mode.")
            direct_kwargs = dict(query_kwargs)
            direct_kwargs["use_subagents"] = False
            direct_kwargs["subagent_backend"] = None
            direct_kwargs["subagent_model"] = None
            direct_kwargs["subagent_lm_kwargs"] = None
            direct_kwargs["rlm_tools"] = None
            direct_kwargs["rlm_interpreter"] = None
            direct_kwargs["require_subagent_call"] = False
            direct_kwargs["custom_prompt"] = (
                (run_custom_prompt or "")
                + "\n\nReliability mode: provide the best research answer directly from context."
            ).strip()
            response = _run_query_with_live_status(
                handler,
                direct_kwargs,
                timeout_seconds=query_timeout_seconds,
            )
            handler.last_metrics = _merge_run_metrics_with_fallback(
                primary=primary_metrics,
                fallback=dict(handler.last_metrics),
            )

        should_enforce_pdf_citations = (
            enforce_pdf_page_citations
            if enforce_pdf_page_citations is not None
            else (selected_output_mode == "research")
        )
        if (
            should_enforce_pdf_citations
            and selected_pdf_docs
            and not response.startswith("Error querying DSPy:")
            and _response_has_pdf_page_na_citation(response, selected_pdf_docs)
        ):
            print("Citation enforcement: retrying once to resolve PDF page citations.")
            citation_retry_kwargs = dict(query_kwargs)
            if citation_repair_direct_mode:
                citation_retry_kwargs["use_subagents"] = False
                citation_retry_kwargs["subagent_backend"] = None
                citation_retry_kwargs["subagent_model"] = None
                citation_retry_kwargs["subagent_lm_kwargs"] = None
                citation_retry_kwargs["rlm_tools"] = None
                citation_retry_kwargs["rlm_interpreter"] = None
                citation_retry_kwargs["require_subagent_call"] = False
            citation_retry_kwargs["custom_prompt"] = (
                (citation_retry_kwargs.get("custom_prompt") or "")
                + "\n\nCitation strictness: for PDF sources, every citation must include a "
                "numeric page from [Page N] markers. Do not output page: n/a for PDF sources."
            ).strip()
            citation_primary_metrics = dict(handler.last_metrics)
            response = _run_query_with_live_status(
                handler,
                citation_retry_kwargs,
                timeout_seconds=query_timeout_seconds,
            )
            handler.last_metrics = _merge_run_metrics_with_retry(
                primary=citation_primary_metrics,
                retry=dict(handler.last_metrics),
                retry_label="citation_repair",
            )

        print(f"\nResponse:\n{response}")
        _print_query_metrics(handler.last_metrics)

        if prompt_yes_no("\nSave response? (y/N): ", default=False):
            filename = input("Enter filename (default: response): ").strip() or "response"
            base_filename = os.path.splitext(os.path.basename(filename))[0]
            os.makedirs(RESPONSE_DIR, exist_ok=True)

            format_opt = input("Format? (md/pdf/both) [default: md]: ").strip().lower() or "md"
            if format_opt not in {"md", "pdf", "both"}:
                print("Unknown format option. Saving as markdown.")
                format_opt = "md"

            try:
                if format_opt in {"md", "both"}:
                    md_path = os.path.join(RESPONSE_DIR, f"{base_filename}.md")
                    save_markdown_response(md_path, query, response)
                    print(f"Saved to {md_path}")

                if format_opt in {"pdf", "both"}:
                    output_pdf_path = os.path.join(RESPONSE_DIR, f"{base_filename}.pdf")
                    save_pdf_response(output_pdf_path, query, response)
                    print(f"Saved to {output_pdf_path}")
            except Exception as error:
                print(f"Error saving file: {error}")

        answered_queries += 1


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
