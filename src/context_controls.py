import os
from dataclasses import dataclass

DEFAULT_DIRECT_CHUNKING_ENABLED = True
DEFAULT_DIRECT_CHUNK_OVERLAP_TOKENS = 256
DEFAULT_DIRECT_MAX_CHUNKS = 64
DEFAULT_OPENROUTER_MIDDLE_OUT_FALLBACK = True
DEFAULT_SUBAGENT_ROOT_COMPACTION_ENABLED = True
DEFAULT_SUBAGENT_COMPACTION_THRESHOLD_PCT = 0.75


@dataclass(frozen=True)
class ContextControls:
    direct_chunking_enabled: bool = DEFAULT_DIRECT_CHUNKING_ENABLED
    direct_chunk_overlap_tokens: int = DEFAULT_DIRECT_CHUNK_OVERLAP_TOKENS
    direct_chunk_max_chunks: int = DEFAULT_DIRECT_MAX_CHUNKS
    openrouter_middle_out_fallback: bool = DEFAULT_OPENROUTER_MIDDLE_OUT_FALLBACK
    subagent_root_compaction_enabled: bool = DEFAULT_SUBAGENT_ROOT_COMPACTION_ENABLED
    subagent_compaction_threshold_pct: float = DEFAULT_SUBAGENT_COMPACTION_THRESHOLD_PCT


def _parse_bool_env(var_name: str, default: bool) -> bool:
    raw = (os.getenv(var_name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on", "y"}


def _parse_int_env(var_name: str, default: int, *, min_value: int = 0) -> int:
    raw = (os.getenv(var_name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if value < min_value:
        return default
    return value


def _parse_float_env(
    var_name: str,
    default: float,
    *,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> float:
    raw = (os.getenv(var_name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if value < min_value or value > max_value:
        return default
    return value


def context_controls_from_env() -> ContextControls:
    return ContextControls(
        direct_chunking_enabled=_parse_bool_env(
            "RLM_DIRECT_CHUNKING_ENABLED", DEFAULT_DIRECT_CHUNKING_ENABLED
        ),
        direct_chunk_overlap_tokens=_parse_int_env(
            "RLM_DIRECT_CHUNK_OVERLAP_TOKENS",
            DEFAULT_DIRECT_CHUNK_OVERLAP_TOKENS,
            min_value=0,
        ),
        direct_chunk_max_chunks=_parse_int_env(
            "RLM_DIRECT_CHUNK_MAX_CHUNKS",
            DEFAULT_DIRECT_MAX_CHUNKS,
            min_value=1,
        ),
        openrouter_middle_out_fallback=_parse_bool_env(
            "RLM_OPENROUTER_MIDDLE_OUT_FALLBACK",
            DEFAULT_OPENROUTER_MIDDLE_OUT_FALLBACK,
        ),
        subagent_root_compaction_enabled=_parse_bool_env(
            "RLM_SUBAGENT_ROOT_COMPACTION_ENABLED",
            DEFAULT_SUBAGENT_ROOT_COMPACTION_ENABLED,
        ),
        subagent_compaction_threshold_pct=_parse_float_env(
            "RLM_SUBAGENT_COMPACTION_THRESHOLD_PCT",
            DEFAULT_SUBAGENT_COMPACTION_THRESHOLD_PCT,
            min_value=0.1,
            max_value=0.99,
        ),
    )
