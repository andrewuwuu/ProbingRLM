import os
import time
import multiprocessing as mp
import queue
from dotenv import load_dotenv

from src.output_utils import save_markdown_response, save_pdf_response
from src.pdf_utils import list_documents, load_document
from src.prompt_loader import load_prompts
from src.rlm_handler import RLMHandler

DOCUMENT_DIR = "embed-docs"
PROMPTS_FILE = "prompts.md"
RESPONSE_DIR = "response"
SUPPORTED_BACKENDS = {"openai", "openrouter"}


def _prompt_yes_no(message: str, default: bool = False) -> bool:
    raw = input(message).strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes"}


def _select_documents(items: list[str], prompt: str) -> list[str]:
    selected_index = -1
    while selected_index < 0 or selected_index >= len(items):
        selection = input(prompt).strip().lower()
        if selection == "q":
            raise KeyboardInterrupt
        if selection in {"a", "all"}:
            return items
        try:
            selected_index = int(selection) - 1
        except ValueError:
            print("Please enter a valid number, 'a' for all, or 'q' to quit.")

    return [items[selected_index]]


def _resolve_backend_and_key() -> tuple[str, str]:
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    configured_backend = os.getenv("RLM_BACKEND")

    if configured_backend and configured_backend not in {"openai", "openrouter"}:
        raise ValueError("RLM_BACKEND must be either 'openai' or 'openrouter'.")

    if configured_backend == "openai":
        if not openai_key:
            raise ValueError("RLM_BACKEND=openai but OPENAI_API_KEY is missing.")
        return "openai", openai_key

    if configured_backend == "openrouter":
        if not openrouter_key:
            raise ValueError("RLM_BACKEND=openrouter but OPENROUTER_API_KEY is missing.")
        return "openrouter", openrouter_key

    if openrouter_key:
        return "openrouter", openrouter_key
    if openai_key:
        return "openai", openai_key

    raise ValueError(
        "No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env."
    )


def _has_backend_key(backend: str) -> bool:
    if backend == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    if backend == "openrouter":
        return bool(os.getenv("OPENROUTER_API_KEY"))
    return False


def _default_model_for_backend(backend: str) -> str:
    if backend == "openrouter":
        return "openai/gpt-5-mini"
    return "gpt-4.1-mini"


def _resolve_subagent_config(
    use_subagents: bool, root_backend: str, root_model: str
) -> tuple[str | None, str | None]:
    if not use_subagents:
        return None, None

    env_backend = (os.getenv("RLM_SUBAGENT_BACKEND") or "").strip().lower()
    env_model = (os.getenv("RLM_SUBAGENT_MODEL") or "").strip()
    default_use_different = bool(env_backend and env_model)

    use_different = _prompt_yes_no(
        "Use a different backend/model for subagents? (y/N): ",
        default=default_use_different,
    )
    if not use_different:
        return None, None

    default_backend = env_backend or ("openai" if root_backend == "openrouter" else "openrouter")
    if default_backend not in SUPPORTED_BACKENDS:
        default_backend = "openai" if root_backend == "openrouter" else "openrouter"

    while True:
        sub_backend = (
            input(
                f"Subagent backend (openai/openrouter) [default: {default_backend}]: "
            ).strip().lower()
            or default_backend
        )
        if sub_backend in SUPPORTED_BACKENDS:
            break
        print("Unsupported backend. Choose 'openai' or 'openrouter'.")

    if not _has_backend_key(sub_backend):
        print(f"No API key configured for backend '{sub_backend}'.")
        print("Falling back to root backend/model for subagents.")
        return None, None

    default_model = env_model or _default_model_for_backend(sub_backend)
    sub_model = input(f"Subagent model (default: {default_model}): ").strip() or default_model

    if sub_backend == root_backend and sub_model == root_model:
        print("Subagent backend/model matches root model. Using root settings.")
        return None, None

    return sub_backend, sub_model


def _get_prompt_section(prompts: dict[str, str], section_name: str) -> str | None:
    lower_name = section_name.lower()
    for key, value in prompts.items():
        if key.strip().lower() == lower_name:
            return value
    return None


def _prompt_query(default_query: str | None) -> str:
    if not default_query:
        return input("\nEnter your query (or 'q' to quit): ").strip()

    preview = " ".join(default_query.split())
    if len(preview) > 120:
        preview = preview[:117] + "..."
    print(f"\nDefault query loaded: {preview}")

    if _prompt_yes_no("Use loaded default query? (Y/n): ", default=True):
        return default_query

    query = input("\nEnter your query (or 'q' to quit): ").strip()
    if query.lower() in {"y", "yes", "n", "no"}:
        print(
            "That looks like a yes/no answer. Enter your query text, or press Enter to use the loaded default."
        )
        retry = input("Query text [default: loaded from file]: ").strip()
        if not retry:
            return default_query
        return retry

    return query


def _should_continue_session(allow_follow_ups: bool, answered_queries: int) -> bool:
    if allow_follow_ups:
        return True
    return answered_queries < 1


def _run_query_with_live_status(handler: RLMHandler, query_kwargs: dict) -> str:
    result: dict[str, dict | str] = {}
    ctx = mp.get_context("spawn")
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
        while process.is_alive():
            elapsed = time.perf_counter() - start_time
            print(
                f"\rProcessing {spinner[spinner_index % len(spinner)]}  elapsed: {elapsed:5.1f}s",
                end="",
                flush=True,
            )
            spinner_index += 1
            time.sleep(0.2)
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

    process.join()
    elapsed = time.perf_counter() - start_time
    print(f"\rProcessing complete. elapsed: {elapsed:5.1f}s{' ' * 12}")

    if process.exitcode != 0:
        handler.last_metrics = {"error": f"worker_exit_code_{process.exitcode}"}
        return f"Error querying RLM: worker exited with code {process.exitcode}"

    try:
        result = out_queue.get_nowait()
    except queue.Empty:
        handler.last_metrics = {"error": "missing_worker_output"}
        return "Error querying RLM: worker produced no output"

    response = str(result.get("response", ""))
    metrics = result.get("metrics", {})
    if isinstance(metrics, dict):
        handler.last_metrics = metrics
    else:
        handler.last_metrics = {}
    return response


def _query_worker(
    backend: str,
    api_key: str | None,
    verbose: bool,
    query_kwargs: dict,
    out_queue,
) -> None:
    worker_handler = RLMHandler(backend=backend, api_key=api_key, verbose=verbose)
    response = worker_handler.query(**query_kwargs)
    out_queue.put(
        {
            "response": response,
            "metrics": worker_handler.last_metrics,
        }
    )


def _print_query_metrics(metrics: dict) -> None:
    if not metrics:
        print("\nRun Metrics: unavailable")
        return

    print("\nRun Metrics:")
    print(f"- Mode: {metrics.get('mode', 'unknown')}")
    if metrics.get("cancelled"):
        print("- Status: cancelled by user")
        return
    if metrics.get("error"):
        print(f"- Error: {metrics.get('error')}")
        return

    execution_time = metrics.get("execution_time")
    if execution_time is not None:
        print(f"- Execution Time: {execution_time:.2f}s")

    print(f"- Iterations: {metrics.get('iterations', 0)}")
    print(f"- Subagent Calls: {metrics.get('subagent_calls', 0)}")
    print(f"- Input Tokens: {metrics.get('total_input_tokens', 0)}")
    print(f"- Output Tokens: {metrics.get('total_output_tokens', 0)}")
    print(f"- Total Tokens: {metrics.get('total_tokens', 0)}")
    if metrics.get("mode") == "rlm_subagents":
        configured_subagent_model = metrics.get("configured_subagent_model")
        subagent_calls = int(metrics.get("subagent_calls", 0))
        if configured_subagent_model and subagent_calls == 0:
            print(
                "- Warning: no subagent calls were made, so only the root model executed."
            )
        if metrics.get("retry_attempted"):
            print("- Recovery: retried once after FINAL_VAR variable error")
        if metrics.get("fallback_used"):
            print("- Recovery: used direct completion fallback after retry failure")

    model_usage = metrics.get("model_usage", {})
    if model_usage:
        print("- Per-Model Usage:")
        for model_name, usage in model_usage.items():
            print(
                f"  {model_name}: calls={usage.get('total_calls', 0)}, "
                f"in={usage.get('total_input_tokens', 0)}, "
                f"out={usage.get('total_output_tokens', 0)}"
            )


def main() -> None:
    load_dotenv()

    try:
        backend, api_key = _resolve_backend_and_key()
    except ValueError as error:
        print(f"Error: {error}")
        print("You can copy .env.example to .env and fill in your keys.")
        return

    print(f"=== RLM Document Retrieval (Backend: {backend}) ===")

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
            docs, "\nSelect a document number, 'a' for all, or 'q' to quit: "
        )
    except KeyboardInterrupt:
        return

    default_model = _default_model_for_backend(backend)
    model_name = input(f"Enter model name (default: {default_model}): ").strip() or default_model

    use_subagents = _prompt_yes_no("Use RLM subagents? (y/N): ", default=False)
    subagent_backend: str | None = None
    subagent_model: str | None = None
    if use_subagents:
        print("Subagent mode enabled (rlms currently supports max_depth=1).")
        subagent_backend, subagent_model = _resolve_subagent_config(
            use_subagents=use_subagents,
            root_backend=backend,
            root_model=model_name,
        )
        if subagent_backend and subagent_model:
            print(
                f"Subagents configured to use backend='{subagent_backend}', model='{subagent_model}'."
            )
        else:
            print(
                "Subagents will use recursive depth-1 llm_query calls on the same root backend/model."
            )

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
    if os.path.exists(PROMPTS_FILE) and _prompt_yes_no(
        f"\nLoad prompts from {PROMPTS_FILE}? (y/N): ", default=False
    ):
        prompts_cache = load_prompts(PROMPTS_FILE)
        system_prompt = _get_prompt_section(prompts_cache, "System")
        default_query = _get_prompt_section(prompts_cache, "Query")
        if system_prompt:
            print(f"Loaded System Prompt: {system_prompt[:50]}...")
        if default_query:
            print(f"Loaded Default Query: {default_query[:50]}...")

    allow_follow_ups = _prompt_yes_no(
        "\nAllow follow-up queries in this run? (Y/n): ",
        default=True,
    )

    handler = RLMHandler(backend=backend, api_key=api_key, verbose=False)

    answered_queries = 0
    while True:
        if not _should_continue_session(allow_follow_ups, answered_queries):
            print("\nOne-time session complete. Exiting.")
            break

        default_query = _get_prompt_section(prompts_cache, "Query")
        query = _prompt_query(default_query)

        if query.lower() == "q":
            break

        system_prompt = _get_prompt_section(prompts_cache, "System")

        if system_prompt:
            print("\nProcessing with System Instructions...")

        query_kwargs = dict(
            prompt=query,
            context=document_text,
            model=model_name,
            use_subagents=use_subagents,
            system_prompt=system_prompt,
            subagent_backend=subagent_backend,
            subagent_model=subagent_model,
        )
        response = _run_query_with_live_status(handler, query_kwargs)
        print(f"\nResponse:\n{response}")
        _print_query_metrics(handler.last_metrics)

        if _prompt_yes_no("\nSave response? (y/N): ", default=False):
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
