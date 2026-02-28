import argparse
import os
import shutil
import subprocess
from typing import Callable

from src.cli_app import (
    default_prompt_query,
    default_prompt_yes_no,
    default_should_continue_session,
    main as run_cli_main,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")


def _is_truthy_env(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


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


def _build_frontend_assets() -> None:
    if _is_truthy_env(os.getenv("PROBINGRLM_SKIP_FRONTEND_BUILD")):
        print("Skipping frontend build (PROBINGRLM_SKIP_FRONTEND_BUILD is set).")
        return

    if not os.path.isdir(FRONTEND_DIR):
        raise RuntimeError(
            f"Frontend directory not found at {FRONTEND_DIR}. Cannot launch web mode."
        )

    npm_bin = shutil.which("npm")
    if not npm_bin:
        raise RuntimeError("npm is required for web mode. Install Node.js and npm first.")

    package_lock = os.path.join(FRONTEND_DIR, "package-lock.json")
    node_modules = os.path.join(FRONTEND_DIR, "node_modules")
    force_install = _is_truthy_env(os.getenv("PROBINGRLM_FRONTEND_FORCE_INSTALL"))
    should_install = force_install or not os.path.isdir(node_modules)

    try:
        if should_install:
            install_cmd = [npm_bin, "ci"] if os.path.isfile(package_lock) else [npm_bin, "install"]
            print("Installing frontend dependencies...")
            subprocess.run(install_cmd, cwd=FRONTEND_DIR, check=True)

        print("Building frontend assets...")
        subprocess.run([npm_bin, "run", "build"], cwd=FRONTEND_DIR, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Frontend build command failed ({error}).") from error


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
        try:
            _build_frontend_assets()
        except RuntimeError as error:
            print(f"Error: {error}")
            return

        print("Starting web server on http://localhost:8000")
        reload_enabled = os.getenv("PROBINGRLM_WEB_RELOAD", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        uvicorn.run("src.backend.app:app", host="127.0.0.1", port=8000, reload=reload_enabled)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
