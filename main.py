import argparse
import os
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
    parser = argparse.ArgumentParser(description="ProbingRLM: CLI and Web document retrieval.")
    parser.add_argument("mode", nargs="?", choices=["cli", "web"], default="cli", help="Mode to run (default: cli).")
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli_main(
            prompt_yes_no=_prompt_yes_no,
            prompt_query=_prompt_query,
            should_continue_session=_should_continue_session,
        )
    elif args.mode == "web":
        import uvicorn
        print("Starting web server on http://localhost:8000")
        reload_enabled = os.getenv("PROBINGRLM_WEB_RELOAD", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        uvicorn.run("src.web_app:app", host="127.0.0.1", port=8000, reload=reload_enabled)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
