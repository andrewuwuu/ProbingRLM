from typing import Optional

from pydantic import BaseModel, Field
from src.context_controls import (
    DEFAULT_DIRECT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_DIRECT_MAX_CHUNKS,
    DEFAULT_OPENROUTER_MIDDLE_OUT_FALLBACK,
    DEFAULT_SUBAGENT_ROOT_COMPACTION_ENABLED,
    DEFAULT_SUBAGENT_COMPACTION_THRESHOLD_PCT,
)


class QueryRequest(BaseModel):
    documents: list[str]
    model_name: Optional[str] = None
    use_subagents: bool = False
    system_prompt: Optional[str] = None
    direct_chunking_enabled: bool = True
    direct_chunk_overlap_tokens: int = Field(default=DEFAULT_DIRECT_CHUNK_OVERLAP_TOKENS, ge=0)
    direct_chunk_max_chunks: int = Field(default=DEFAULT_DIRECT_MAX_CHUNKS, gt=0)
    openrouter_middle_out_fallback: bool = DEFAULT_OPENROUTER_MIDDLE_OUT_FALLBACK
    subagent_root_compaction_enabled: bool = DEFAULT_SUBAGENT_ROOT_COMPACTION_ENABLED
    subagent_compaction_threshold_pct: float = Field(
        default=DEFAULT_SUBAGENT_COMPACTION_THRESHOLD_PCT,
        ge=0.1,
        le=0.99,
    )
    query: str
    max_iterations: Optional[int] = None
    max_subagent_calls: Optional[int] = None
    subagent_backend: Optional[str] = None
    subagent_model: Optional[str] = None


class QueryResponse(BaseModel):
    response: str
    metrics: dict
