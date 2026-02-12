from typing import Callable

from src.prompt_loader import extract_template_variables, render_template

OUTPUT_MODES = ("research", "concise")


def _normalize_output_mode(value: str | None) -> str:
    raw = (value or "").strip().lower()
    aliases = {
        "concise": "concise",
        "short": "concise",
        "brief": "concise",
        "research": "research",
        "full": "research",
        "long": "research",
        "detailed": "research",
    }
    return aliases.get(raw, "")


def _resolve_default_output_mode(
    prompt_vars: dict[str, str],
    env_output_mode: str | None,
) -> str:
    env_mode = _normalize_output_mode(env_output_mode)
    if env_mode:
        return env_mode

    prompt_mode_raw = (prompt_vars.get("output_mode") or "").strip()
    prompt_mode = _normalize_output_mode(prompt_mode_raw)
    if prompt_mode_raw and not prompt_mode:
        print(
            f"Ignoring prompt variable output_mode='{prompt_mode_raw}': "
            "use 'research' or 'concise'."
        )
    return prompt_mode or "concise"


def _prompt_output_mode(default_mode: str) -> str:
    effective_default = default_mode if default_mode in OUTPUT_MODES else "research"
    while True:
        raw = input(
            f"Output mode ({'/'.join(OUTPUT_MODES)}) [default: {effective_default}]: "
        ).strip()
        if not raw:
            return effective_default

        normalized = _normalize_output_mode(raw)
        if normalized:
            return normalized

        print("Unsupported output mode. Use 'research' or 'concise'.")


def _output_template_for_mode(mode: str) -> str:
    if mode == "concise":
        return (
            "1. Direct Answer (1 short paragraph)\n"
            "2. Key Evidence (2-4 bullets, each with a short quote and citation)\n"
            "3. Missing Evidence (1 line if anything required is absent)\n"
            "4. Confidence: High | Medium | Low"
        )
    return (
        "1. Thesis (1 dense paragraph)\n"
        "2. Evidence Map (6-12 bullets: short quote + citation + one-line relevance)\n"
        "3. Analytical Synthesis (3-6 paragraphs linking evidence to the research question)\n"
        "4. Theoretical / Comparative Lens (1-3 paragraphs; label inference explicitly)\n"
        "5. Gaps and Limits (explicit unsupported claims; use exact phrase "
        "'Not found in provided document context' per missing claim)\n"
        "6. Conclusion (1 paragraph)\n"
        "7. Confidence: High | Medium | Low (with brief reason)"
    )


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


def _get_prompt_section(prompts: dict[str, str], section_name: str) -> str | None:
    lower_name = section_name.lower()
    for key, value in prompts.items():
        if key.strip().lower() == lower_name:
            return value
    return None


def _build_runtime_prompt_vars(
    backend: str,
    model_name: str,
    selected_docs: list[str],
    document_text: str,
    query_index: int,
    output_mode: str,
    output_template: str,
) -> dict[str, str]:
    return {
        "backend": backend,
        "model": model_name,
        "doc_count": str(len(selected_docs)),
        "docs_csv": ", ".join(selected_docs),
        "docs_list": "\n".join(selected_docs),
        "context_chars": str(len(document_text)),
        "query_index": str(query_index),
        "output_mode": output_mode,
        "output_template": output_template,
    }


def _render_prompt_section(
    section_text: str | None,
    section_name: str,
    runtime_vars: dict[str, str],
    prompt_vars: dict[str, str],
) -> str | None:
    if section_text is None:
        return None

    merged_vars: dict[str, str] = dict(runtime_vars)
    merged_vars.update(prompt_vars)

    for var_name in extract_template_variables(section_text):
        if var_name in merged_vars:
            continue
        value = input(
            f"Template variable '{var_name}' for section '{section_name}': "
        ).strip()
        prompt_vars[var_name] = value
        merged_vars[var_name] = value

    rendered = render_template(section_text, merged_vars)
    return rendered.strip() if rendered is not None else None


def default_prompt_query(
    default_query: str | None,
    prompt_yes_no: Callable[[str, bool], bool],
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
