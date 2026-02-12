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
- Modes:
  - direct model call
  - DSPy RLM recursive mode (optional subagent backend/model)
  - Research safety fallback: if recursive research run times out/stalls, it retries once in direct mode
- Programmable prompt presets from `prompts.md` (`Variables`, `System`, `Custom Prompt`, `Query`, `RLM Signature`)
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
DSPY_QUERY_TIMEOUT_SECONDS=180
DSPY_REQUIRE_SUBAGENT_CALL=false
DSPY_REQUIRE_SUBAGENT_CALL_RETRY_ONCE=false
DSPY_ENFORCE_PDF_PAGE_CITATIONS=true
DSPY_RESEARCH_ALLOW_DIRECT_FALLBACK=true
DSPY_CITATION_REPAIR_DIRECT_MODE=false
DSPY_OPENROUTER_AUTO_MIDDLE_OUT=true
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
  - `{{variable}}` placeholders are supported in prompt sections.
  - missing template variables are prompted interactively at runtime.
  - `output_mode` and `output_template` are injected automatically so you can switch between research and concise outputs.
  - in `prompts.md`, keep inline notes in `Variables` as plain text (not markdown headers) so defaults parse correctly.
- PDF citations in research mode:
  - with `DSPY_ENFORCE_PDF_PAGE_CITATIONS=true`, one automatic repair retry is performed if output includes `[source: <pdf>, page: n/a]`.
  - keep `DSPY_CITATION_REPAIR_DIRECT_MODE=false` to preserve subagent calls during citation repair retries.
- OpenRouter context overflow:
  - with `DSPY_OPENROUTER_AUTO_MIDDLE_OUT=true`, one automatic retry is attempted using OpenRouter `middle-out` transform when max-context errors are detected.
- Research fallback mode:
  - `DSPY_RESEARCH_ALLOW_DIRECT_FALLBACK=true` allows automatic direct retry on recursive timeout/stall.
  - set it to `false` if you want research mode to stay recursive/subagent-only.
