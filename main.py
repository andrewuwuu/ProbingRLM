from typing import Callable

from src.cli_app import (
    default_prompt_query,
    default_prompt_yes_no,
    default_should_continue_session,
    main as run_cli_main,
)


def _prompt_yes_no(message: str, default: bool = False) -> bool:
    return default_prompt_yes_no(message, default)


def _prompt_query(
    default_query: str | None,
    prompt_yes_no_func: Callable[[str, bool], bool] | None = None,
) -> str:
    prompt_yes_no_impl = prompt_yes_no_func or _prompt_yes_no
    return default_prompt_query(default_query, prompt_yes_no_impl)


def _should_continue_session(allow_follow_ups: bool, answered_queries: int) -> bool:
    return default_should_continue_session(allow_follow_ups, answered_queries)


def main() -> None:
    run_cli_main(
        prompt_yes_no=_prompt_yes_no,
        prompt_query=_prompt_query,
        should_continue_session=_should_continue_session,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
