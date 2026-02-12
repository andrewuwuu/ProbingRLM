import importlib
import json
import os
from typing import Any, Callable

from src.rlm_handler import RLMHandler

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


def _read_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return ""


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


def _parse_positive_float_env(var_name: str) -> float | None:
    raw = (os.getenv(var_name) or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        print(f"Ignoring {var_name}: expected number but got '{raw}'.")
        return None
    if value <= 0:
        print(f"Ignoring {var_name}: expected > 0 but got '{value}'.")
        return None
    return value


def _parse_json_object_env(var_name: str) -> dict[str, Any] | None:
    raw = (os.getenv(var_name) or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print(f"Ignoring {var_name}: expected valid JSON object but got '{raw}'.")
        return None
    if not isinstance(parsed, dict):
        print(f"Ignoring {var_name}: expected JSON object.")
        return None
    return parsed


def _parse_bool_env(var_name: str) -> bool | None:
    raw = (os.getenv(var_name) or "").strip().lower()
    if not raw:
        return None
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    print(f"Ignoring {var_name}: expected true/false but got '{raw}'.")
    return None


def _load_tools_from_env(var_name: str) -> list[Callable] | None:
    raw = (os.getenv(var_name) or "").strip()
    if not raw:
        return None

    tools: list[Callable] = []
    for spec in [item.strip() for item in raw.split(",") if item.strip()]:
        if ":" not in spec:
            print(f"Ignoring tool spec '{spec}': expected module.path:function_name.")
            continue
        module_name, attr_name = spec.rsplit(":", 1)
        try:
            module = importlib.import_module(module_name)
            tool = getattr(module, attr_name)
        except Exception as error:
            print(f"Ignoring tool spec '{spec}': {error}")
            continue
        if not callable(tool):
            print(f"Ignoring tool spec '{spec}': target is not callable.")
            continue
        tools.append(tool)
    return tools or None


def _backend_configuration_error(backend: str) -> str | None:
    api_key_env = REQUIRED_API_KEY_ENV_BY_BACKEND.get(backend)
    if api_key_env and not os.getenv(api_key_env):
        return f"{api_key_env} is missing."

    if backend == "vllm" and not _read_env("DSPY_VLLM_BASE_URL"):
        return "DSPY_VLLM_BASE_URL is missing."

    if backend == "azure_openai" and not (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip():
        return "AZURE_OPENAI_ENDPOINT is missing."

    return None


def _resolve_backend_and_key() -> tuple[str, str | None]:
    configured_backend = (os.getenv("DSPY_BACKEND") or "").strip().lower()

    if configured_backend and configured_backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            "DSPY_BACKEND must be one of: "
            + ", ".join(SUPPORTED_BACKENDS)
            + "."
        )

    if configured_backend:
        config_error = _backend_configuration_error(configured_backend)
        if config_error:
            raise ValueError(f"DSPY_BACKEND={configured_backend} but {config_error}")
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
        "No usable backend config found. Set DSPY_BACKEND and required "
        "credentials in .env "
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


def _resolve_root_model_default(backend: str) -> str:
    backend_model_env_by_backend = {
        "openrouter": ("OPENROUTER_MODEL",),
        "openai": ("OPENAI_MODEL",),
        "anthropic": ("ANTHROPIC_MODEL",),
        "gemini": ("GEMINI_MODEL",),
        "azure_openai": ("AZURE_OPENAI_DEPLOYMENT",),
    }
    backend_specific = backend_model_env_by_backend.get(backend, tuple())
    selected_model = (
        (os.getenv("DSPY_MODEL") or "").strip()
        or _read_env(*backend_specific)
        or _default_model_for_backend(backend)
    )
    return RLMHandler.canonical_model_for_backend(backend, selected_model)


def _resolve_subagent_config(
    use_subagents: bool,
    root_backend: str,
    root_model: str,
    prompt_yes_no: Callable[[str, bool], bool],
) -> tuple[str | None, str | None]:
    if not use_subagents:
        return None, None

    env_backend = (os.getenv("DSPY_SUBAGENT_BACKEND") or "").strip().lower()
    env_model = (os.getenv("DSPY_SUBAGENT_MODEL") or "").strip()
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
    sub_model = RLMHandler.canonical_model_for_backend(sub_backend, sub_model)

    if sub_backend == root_backend and sub_model == root_model:
        print("Subagent backend/model matches root model. Using root settings.")
        return None, None

    return sub_backend, sub_model
