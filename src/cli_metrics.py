import os
import re

CITATION_PATTERN = re.compile(
    r"\[source:\s*([^,\]]+)\s*,\s*page:\s*([^\]]+)\]",
    re.IGNORECASE,
)
CONTEXT_LIMIT_PATTERN = re.compile(
    r"maximum context length is\s+(\d+)\s+tokens.*?requested about\s+(\d+)\s+tokens",
    re.IGNORECASE | re.DOTALL,
)
OPENROUTER_MIDDLE_OUT_TRANSFORM = "middle-out"


def _should_use_research_direct_fallback(
    output_mode: str,
    use_subagents: bool,
    response: str,
    metrics: dict,
) -> bool:
    if output_mode != "research" or not use_subagents:
        return False

    error = str(metrics.get("error") or "")
    if error in {"query_timeout", "missing_worker_output"}:
        return True
    if error.startswith("worker_exit_code_"):
        return True
    if response.startswith("Error querying DSPy: timed out"):
        return True
    return False


def _merge_run_metrics_with_fallback(primary: dict, fallback: dict) -> dict:
    merged = dict(fallback)
    merged["fallback_used"] = True
    merged["primary_mode"] = primary.get("mode")
    merged["primary_error"] = primary.get("error")
    merged["mode"] = "research_direct_fallback"

    merged["execution_time"] = float(primary.get("execution_time") or 0.0) + float(
        fallback.get("execution_time") or 0.0
    )
    merged["iterations"] = int(primary.get("iterations") or 0) + int(
        fallback.get("iterations") or 0
    )
    merged["subagent_calls"] = int(primary.get("subagent_calls") or 0) + int(
        fallback.get("subagent_calls") or 0
    )
    merged["total_input_tokens"] = int(primary.get("total_input_tokens") or 0) + int(
        fallback.get("total_input_tokens") or 0
    )
    merged["total_output_tokens"] = int(primary.get("total_output_tokens") or 0) + int(
        fallback.get("total_output_tokens") or 0
    )
    merged["total_tokens"] = int(primary.get("total_tokens") or 0) + int(
        fallback.get("total_tokens") or 0
    )
    return merged


def _merge_run_metrics_with_retry(primary: dict, retry: dict, retry_label: str) -> dict:
    merged = dict(retry)
    merged["retry_used"] = retry_label
    merged["primary_mode"] = primary.get("mode")
    merged["primary_error"] = primary.get("error")
    merged["mode"] = f"{retry_label}_retry"

    if primary.get("fallback_used"):
        merged["fallback_used"] = True

    merged["execution_time"] = float(primary.get("execution_time") or 0.0) + float(
        retry.get("execution_time") or 0.0
    )
    merged["iterations"] = int(primary.get("iterations") or 0) + int(
        retry.get("iterations") or 0
    )
    merged["subagent_calls"] = int(primary.get("subagent_calls") or 0) + int(
        retry.get("subagent_calls") or 0
    )
    merged["total_input_tokens"] = int(primary.get("total_input_tokens") or 0) + int(
        retry.get("total_input_tokens") or 0
    )
    merged["total_output_tokens"] = int(primary.get("total_output_tokens") or 0) + int(
        retry.get("total_output_tokens") or 0
    )
    merged["total_tokens"] = int(primary.get("total_tokens") or 0) + int(
        retry.get("total_tokens") or 0
    )
    return merged


def _response_has_pdf_page_na_citation(response: str, pdf_doc_names: set[str]) -> bool:
    if not response or not pdf_doc_names:
        return False
    page_na_values = {"n/a", "na", "not available", "unknown"}
    for source_raw, page_raw in CITATION_PATTERN.findall(response):
        source = os.path.basename(source_raw.strip().strip("'\"")).lower()
        page = page_raw.strip().lower()
        if source in pdf_doc_names and page in page_na_values:
            return True
    return False


def _parse_context_limit_error(response: str) -> tuple[int, int] | None:
    if not response:
        return None
    match = CONTEXT_LIMIT_PATTERN.search(response)
    if not match:
        return None
    try:
        max_tokens = int(match.group(1))
        requested_tokens = int(match.group(2))
    except (TypeError, ValueError):
        return None
    return max_tokens, requested_tokens


def _with_openrouter_middle_out(
    lm_kwargs: dict | None,
) -> dict:
    updated = dict(lm_kwargs or {})
    raw_extra_body = updated.get("extra_body")
    extra_body = dict(raw_extra_body) if isinstance(raw_extra_body, dict) else {}

    transforms_raw = extra_body.get("transforms")
    transforms: list[str]
    if isinstance(transforms_raw, str):
        transforms = [transforms_raw]
    elif isinstance(transforms_raw, (list, tuple, set)):
        transforms = [str(item) for item in transforms_raw if str(item).strip()]
    else:
        transforms = []

    if OPENROUTER_MIDDLE_OUT_TRANSFORM not in transforms:
        transforms.append(OPENROUTER_MIDDLE_OUT_TRANSFORM)

    extra_body["transforms"] = transforms
    updated["extra_body"] = extra_body
    return updated


def _print_query_metrics(metrics: dict, verbose_metrics: bool = False) -> None:
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
    print(f"- Total Tokens: {metrics.get('total_tokens', 0)}")
    if metrics.get("fallback_used"):
        print("- Fallback: recursive run failed; direct research retry used.")
    if metrics.get("retry_used") == "middle_out":
        print("- Context Repair: re-ran once with OpenRouter middle-out transform.")
    if metrics.get("retry_used") == "citation_repair":
        print("- Citation Repair: re-ran once to resolve PDF page citations.")
    configured_subagent_model = metrics.get("configured_subagent_model")
    subagent_calls = int(metrics.get("subagent_calls", 0))
    if configured_subagent_model and subagent_calls == 0:
        print("- Warning: no subagent calls were made; only root LM calls were used.")

    if verbose_metrics:
        print(f"- Input Tokens: {metrics.get('total_input_tokens', 0)}")
        print(f"- Output Tokens: {metrics.get('total_output_tokens', 0)}")
        model_usage = metrics.get("model_usage", {})
        if model_usage:
            print("- Per-Model Usage:")
            for model_name, usage in model_usage.items():
                print(
                    f"  {model_name}: calls={usage.get('total_calls', 0)}, "
                    f"in={usage.get('total_input_tokens', 0)}, "
                    f"out={usage.get('total_output_tokens', 0)}"
                )
