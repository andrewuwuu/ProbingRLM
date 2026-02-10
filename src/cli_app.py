import os
import time
import multiprocessing as mp
import queue
from typing import Callable
from dotenv import load_dotenv

from src.output_utils import save_markdown_response, save_pdf_response
from src.pdf_utils import list_documents, load_document
from src.prompt_loader import load_prompts
from src.rlm_handler import RLMHandler

DOCUMENT_DIR = "embed-docs"
PROMPTS_FILE = "prompts.md"
RESPONSE_DIR = "response"
SUPPORTED_BACKENDS = (
    "openai",
    "openrouter",
    "anthropic",
    "portkey",
    "litellm",
    "vllm",
    "vercel",
    "gemini",
    "azure_openai",
)
REQUIRED_API_KEY_ENV_BY_BACKEND = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "portkey": "PORTKEY_API_KEY",
    "vercel": "AI_GATEWAY_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "azure_openai": "AZURE_OPENAI_API_KEY",
}


def default_prompt_yes_no(message: str, default: bool = False) -> bool:
    raw = input(message).strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes"}


def _parse_positive_int_env(var_name: str) -> int | None:
    raw = (os.getenv(var_name) or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        print(f"Ignoring {var_name}: expected integer but got '{raw}'.")
        return None
    if value <= 0:
        print(f"Ignoring {var_name}: expected > 0 but got '{value}'.")
        return None
    return value


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


def _backend_configuration_error(backend: str) -> str | None:
    api_key_env = REQUIRED_API_KEY_ENV_BY_BACKEND.get(backend)
    if api_key_env and not os.getenv(api_key_env):
        return f"{api_key_env} is missing."

    if backend == "vllm" and not (os.getenv("RLM_VLLM_BASE_URL") or "").strip():
        return "RLM_VLLM_BASE_URL is missing."

    if backend == "azure_openai" and not (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip():
        return "AZURE_OPENAI_ENDPOINT is missing."

    return None


def _resolve_backend_and_key() -> tuple[str, str | None]:
    configured_backend = (os.getenv("RLM_BACKEND") or "").strip().lower()

    if configured_backend and configured_backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            "RLM_BACKEND must be one of: " + ", ".join(SUPPORTED_BACKENDS) + "."
        )

    if configured_backend:
        config_error = _backend_configuration_error(configured_backend)
        if config_error:
            raise ValueError(f"RLM_BACKEND={configured_backend} but {config_error}")
        api_key_env = REQUIRED_API_KEY_ENV_BY_BACKEND.get(configured_backend)
        return configured_backend, (os.getenv(api_key_env) if api_key_env else None)

    autodetect_order = [
        "openrouter",
        "openai",
        "anthropic",
        "gemini",
        "portkey",
        "vercel",
        "azure_openai",
    ]
    for backend in autodetect_order:
        if not _backend_configuration_error(backend):
            api_key_env = REQUIRED_API_KEY_ENV_BY_BACKEND.get(backend)
            return backend, (os.getenv(api_key_env) if api_key_env else None)

    raise ValueError(
        "No usable backend config found. Set RLM_BACKEND and required credentials in .env "
        "(e.g., OPENROUTER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, "
        "PORTKEY_API_KEY, AI_GATEWAY_API_KEY, or AZURE_OPENAI_* settings)."
    )


def _default_model_for_backend(backend: str) -> str:
    defaults = {
        "openrouter": "openai/gpt-5-mini",
        "openai": "gpt-4.1-mini",
        "anthropic": "claude-3-5-haiku-latest",
        "gemini": "gemini-2.5-flash",
        "portkey": "openai/gpt-4.1-mini",
        "litellm": "openai/gpt-4.1-mini",
        "vllm": "meta-llama/Llama-3.1-8B-Instruct",
        "vercel": "openai/gpt-4.1-mini",
        "azure_openai": "gpt-4o-mini",
    }
    return defaults.get(backend, "gpt-4.1-mini")


def _resolve_subagent_config(
    use_subagents: bool,
    root_backend: str,
    root_model: str,
    prompt_yes_no: Callable[[str, bool], bool] = default_prompt_yes_no,
) -> tuple[str | None, str | None]:
    if not use_subagents:
        return None, None

    env_backend = (os.getenv("RLM_SUBAGENT_BACKEND") or "").strip().lower()
    env_model = (os.getenv("RLM_SUBAGENT_MODEL") or "").strip()
    default_use_different = bool(env_backend and env_model)

    use_different = prompt_yes_no(
        "Use a different backend/model for subagents? (y/N): ",
        default=default_use_different,
    )
    if not use_different:
        return None, None

    default_backend = env_backend or root_backend
    if default_backend not in SUPPORTED_BACKENDS:
        default_backend = root_backend

    while True:
        sub_backend = (
            input(
                f"Subagent backend ({'/'.join(SUPPORTED_BACKENDS)}) [default: {default_backend}]: "
            ).strip().lower()
            or default_backend
        )
        if sub_backend in SUPPORTED_BACKENDS:
            break
        print(f"Unsupported backend. Choose one of: {', '.join(SUPPORTED_BACKENDS)}.")

    config_error = _backend_configuration_error(sub_backend)
    if config_error:
        print(f"Backend '{sub_backend}' is not configured: {config_error}")
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


def default_prompt_query(
    default_query: str | None,
    prompt_yes_no: Callable[[str, bool], bool] = default_prompt_yes_no,
) -> str:
    if not default_query:
        return input("\nEnter your query (or 'q' to quit): ").strip()

    preview = " ".join(default_query.split())
    if len(preview) > 120:
        preview = preview[:117] + "..."
    print(f"\nDefault query loaded: {preview}")

    if prompt_yes_no("Use loaded default query? (Y/n): ", default=True):
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


def default_should_continue_session(allow_follow_ups: bool, answered_queries: int) -> bool:
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
        if metrics.get("max_subagent_calls_exceeded"):
            print(
                "- Guard: max subagent calls exceeded; run aborted by configured safety limit."
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
    max_iterations = _parse_positive_int_env("RLM_MAX_ITERATIONS")
    max_subagent_calls = _parse_positive_int_env("RLM_MAX_SUBAGENT_CALLS")
    if max_iterations is not None:
        print(f"RLM max iterations cap active: {max_iterations}")
    if max_subagent_calls is not None:
        print(f"RLM max subagent calls cap active: {max_subagent_calls}")

    use_subagents = prompt_yes_no("Use RLM subagents? (y/N): ", default=False)
    subagent_backend: str | None = None
    subagent_model: str | None = None
    if use_subagents:
        print("Subagent mode enabled (rlms currently supports max_depth=1).")
        subagent_backend, subagent_model = _resolve_subagent_config(
            use_subagents=use_subagents,
            root_backend=backend,
            root_model=model_name,
            prompt_yes_no=prompt_yes_no,
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
    if os.path.exists(PROMPTS_FILE) and prompt_yes_no(
        f"\nLoad prompts from {PROMPTS_FILE}? (y/N): ", default=False
    ):
        prompts_cache = load_prompts(PROMPTS_FILE)
        system_prompt = _get_prompt_section(prompts_cache, "System")
        default_query = _get_prompt_section(prompts_cache, "Query")
        if system_prompt:
            print(f"Loaded System Prompt: {system_prompt[:50]}...")
        if default_query:
            print(f"Loaded Default Query: {default_query[:50]}...")

    allow_follow_ups = prompt_yes_no(
        "\nAllow follow-up queries in this run? (Y/n): ",
        default=True,
    )

    handler = RLMHandler(backend=backend, api_key=api_key, verbose=False)

    answered_queries = 0
    while True:
        if not should_continue_session(allow_follow_ups, answered_queries):
            print("\nOne-time session complete. Exiting.")
            break

        default_query = _get_prompt_section(prompts_cache, "Query")
        query = prompt_query(default_query, prompt_yes_no)

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
            max_iterations=max_iterations,
            max_subagent_calls=max_subagent_calls,
        )
        response = _run_query_with_live_status(handler, query_kwargs)
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
