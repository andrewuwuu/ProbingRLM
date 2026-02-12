# ProbingRLM

CLI document Q&A for local files using DSPy (`dspy.LM` + `dspy.RLM`).

## Quick Start

```bash
uv sync
cp .env.example .env
uv run main.py
```

Minimal `.env`:

```ini
DSPY_BACKEND=openrouter
OPENROUTER_API_KEY=your_openrouter_api_key_here
DSPY_MODEL=openrouter/free
```

## What It Supports

- Documents from `embed-docs/`: `.pdf`, `.docx`, `.doc/.docs`, `.md`, `.txt`, `.json`, `.yaml`, `.csv`, and other text-like files.
- Prompt configuration folder (`prompt-config/`) with a single `config.md` (plus legacy per-file support).
- Modes:
  - direct model call
  - DSPy RLM recursive mode (optional subagent backend/model)
  - Research safety fallback: if recursive research run times out/stalls, it retries once in direct mode
- Optional programmable prompt presets from `prompts.md` (`Variables`, `System`, `Custom Prompt`, `Query`, `RLM Signature`)
- Switchable answer style: `research` (full, citation-heavy) or `concise` (short summary with citations)
- Save response as Markdown and/or PDF
- Runtime metrics (tokens, iterations, model usage)

## Code Layout

- `main.py`: thin entrypoint.
- `src/cli_app.py`: runtime orchestration loop.
- `src/cli_config.py`: backend/env/tool parsing and model defaults.
- `src/cli_prompting.py`: prompt sections, template rendering, output-mode handling.
- `src/cli_metrics.py`: retry/fallback/citation/context-limit helpers and metrics printing.
- `src/rlm_handler.py`: DSPy LM/RLM engine integration.

## Optional Config

Subagent routing:

```ini
DSPY_SUBAGENT_BACKEND=openrouter
DSPY_SUBAGENT_MODEL=arcee-ai/trinity-large-preview:free
```

RLM limits:

```ini
DSPY_RLM_MAX_ITERATIONS=20
DSPY_RLM_MAX_LLM_CALLS=40
DSPY_RLM_MAX_OUTPUT_CHARS=10000
DSPY_RLM_MAX_DEPTH=3
DSPY_SUBAGENT_PREFETCH_CALLS=0
```

RLM customization:

```ini
DSPY_RLM_SIGNATURE=context, query -> answer
DSPY_CUSTOM_PROMPT=Prefer concise answers.
DSPY_OUTPUT_MODE=research
DSPY_USE_PROMPTS_FILE=false
DSPY_PROMPT_CONFIG_DIR=prompt-config
DSPY_QUERY_TIMEOUT_SECONDS=180
DSPY_REQUIRE_SUBAGENT_CALL=false
DSPY_REQUIRE_SUBAGENT_CALL_RETRY_ONCE=false
DSPY_ENFORCE_PDF_PAGE_CITATIONS=true
DSPY_RESEARCH_ALLOW_DIRECT_FALLBACK=true
DSPY_CITATION_REPAIR_DIRECT_MODE=false
DSPY_OPENROUTER_AUTO_MIDDLE_OUT=true
DSPY_RLM_VERBOSE=false
DSPY_LIVE_LM_LOGS=false
DSPY_DISABLE_JSON_ADAPTER_FALLBACK=true
DSPY_CONTEXT_MAX_CHARS=260000
DSPY_CONTEXT_CHUNK_CHARS=2200
DSPY_CONTEXT_MAX_CHUNKS=120
DSPY_LM_KWARGS={"temperature":0.2}
DSPY_SUBAGENT_LM_KWARGS={"temperature":0.0}
DSPY_RLM_INTERPRETER=true
DSPY_RLM_TOOLS=my_tools.web:search,my_tools.math:calculator
```

Endpoint overrides (only if needed):

```ini
DSPY_OPENROUTER_API_BASE=https://openrouter.ai/api/v1
DSPY_VLLM_BASE_URL=
DSPY_LITELLM_API_BASE=
DSPY_LITELLM_API_KEY=
DSPY_PORTKEY_API_BASE=
DSPY_VERCEL_API_BASE=
```

## Model Notes

- `DSPY_MODEL` controls the root model default shown in CLI.
- `DSPY_SUBAGENT_MODEL` controls subagent calls only.
- For OpenRouter:
  - `free` -> `openrouter/free`
  - `gpt-4.1-mini` -> `openrouter/openai/gpt-4.1-mini`
  - `arcee-ai/...:free` -> `openrouter/arcee-ai/...:free`

## Testing

```bash
uv run python3 -m pytest -q
```

## Common Issues

- AES PDF error:
  - run `uv sync` (requires `cryptography` for encrypted PDFs).
- Signature input mismatch:
  - use `DSPY_RLM_SIGNATURE` with `context, query` (and optional guidance-like field names).
- Prompt templates:
  - simple default workflow: use `prompt-config/config.md` sections (`# Question`, `# Scope`, `# System`, `# Custom Prompt`, `# Signature`, `# Output Template`).
  - `prompt-config/README.md` includes starter guidance and a single sample file.
  - set `DSPY_PROMPT_CONFIG_DIR` to use another folder path.
  - `prompts.md` is optional and disabled by default (`DSPY_USE_PROMPTS_FILE=false`).
  - if you enable `prompts.md`, the existing template behavior still applies.
  - `{{variable}}` placeholders are supported in prompt sections.
  - missing template variables are prompted interactively at runtime.
  - `output_mode` and `output_template` are injected automatically so you can switch between research and concise outputs.
  - in `prompts.md`, keep inline notes in `Variables` as plain text (not markdown headers) so defaults parse correctly.
- PDF citations in research mode:
  - with `DSPY_ENFORCE_PDF_PAGE_CITATIONS=true`, one automatic repair retry is performed if output includes `[source: <pdf>, page: n/a]`.
  - keep `DSPY_CITATION_REPAIR_DIRECT_MODE=false` to preserve subagent calls during citation repair retries.
- OpenRouter context overflow:
  - with `DSPY_OPENROUTER_AUTO_MIDDLE_OUT=true`, one automatic retry is attempted using OpenRouter `middle-out` transform when max-context errors are detected.
  - query-focused context selection is enabled by default; tune with `DSPY_CONTEXT_MAX_CHARS`, `DSPY_CONTEXT_CHUNK_CHARS`, and `DSPY_CONTEXT_MAX_CHUNKS`.
- Live progress logs:
  - set `DSPY_LIVE_LM_LOGS=true` to print ongoing root/sub LM call start/finish logs during a run.
  - set `DSPY_RLM_VERBOSE=true` to include DSPy RLM iteration/code logs.
- Provider compatibility:
  - DSPy may fallback to JSONAdapter (`response_format=json_object`) when chat parsing fails.
  - some provider endpoints (e.g. OpenRouter StepFun routes) reject `json_object`.
  - this app auto-disables JSONAdapter fallback for known incompatible models (currently StepFun model IDs).
  - you can still override manually with `DSPY_DISABLE_JSON_ADAPTER_FALLBACK=true|false`.
- Research fallback mode:
  - `DSPY_RESEARCH_ALLOW_DIRECT_FALLBACK=true` allows automatic direct retry on recursive timeout/stall.
  - set it to `false` if you want research mode to stay recursive/subagent-only.
