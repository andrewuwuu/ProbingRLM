# ProbingRLM

## About
`ProbingRLM` is an elegant, local-first web and CLI application built on the `rlms` library. It enables you to securely load local technical documents (PDF, DOCX, TXT, MD) and interrogate them using Advanced Agentic Models (RLMs) or Direct LLM queries. It is a capable, transparent tool for research and technical handoff, providing granular real-time visibility into the reasoning process of Subagents and long-context processing (chunking, compaction, and provider fallbacks).

The project is designed to stay practical:
- **Minimal setup** with explicit backend/key handling.
- **CLI + Modern Web UI** (SolidJS) interactive workflows.
- **Real-time Streaming** visibility into iterative agent thought processes via Server-Sent Events (SSE).
- Test coverage around core agentic and chunking behavior.

## What It Does

- Scans `embed-docs/` for supported document files to extract their context.
- Sends your query plus document context to an LLM via `rlms`.
- **Live Streams Agent Activity:** Watch the underlying agent iterate, write code, and spin up subagents live in the Web UI.
- Supports both **Direct LM Mode** for fast extraction and **Recursive Agent Mode (RLM Subagents)** for complex reasoning over multiple turns.
- Automatically handles over-budget contexts by switching to **Adaptive Chunking** or provider-specific context tools like **OpenRouter Middle-Out Rescue**.
- In Subagent Mode, uses **Root Compaction Guard** to prevent the primary agent from overflowing its context limit during long chains of thought.
- Optionally loads reusable prompts from `prompts.md`.
- Saves final outputs as `.md` or `.pdf` for documentation and reporting.

## How It Works

The CLI entrypoint is `main.py`.

1. Environment variables are loaded from `.env`.
2. Backend and credentials are resolved from supported providers.
3. You pick document(s) and model.
4. You choose execution mode:
   - direct LM mode (`use_subagents = false`)
   - RLM subagent mode (`use_subagents = true`)
   - optional cross-provider/model subagent routing
5. The response is printed and can be saved to disk.

Implementation files:
- `main.py`: thin CLI entrypoint/wrapper
- `src/cli_app.py`: interactive flow and orchestration
- `src/rlm_handler.py`: backend/model calls and RLM behavior
- `src/pdf_utils.py`: document discovery + text extraction
- `src/prompt_loader.py`: markdown prompt section parsing
- `src/output_utils.py`: markdown/PDF output writing
- `src/backend/app.py`: FastAPI app factory + static frontend mount
- `src/backend/routes.py`: API routes (`/api/*`)
- `src/backend/services.py`: query orchestration + SSE streaming backend logic
- `src/web_app.py`: compatibility export for legacy imports
- `frontend/`: SolidJS web app source (built to `frontend/dist`)
- `tests/`: behavior checks for retrieval, prompts, and output generation

## Requirements

- Python `>=3.11`
- `uv` for dependency management
- Node.js + `npm` for web build pipeline
- Backend-specific credentials for the provider you use.

## Installation

```bash
uv sync
cp .env.example .env
```

Update `.env`:

```ini
OPENROUTER_API_KEY=your_key_here
# or
OPENAI_API_KEY=your_key_here

# Optional explicit backend selection:
# openai|openrouter|anthropic|gemini|portkey|vercel|azure_openai|litellm|vllm
RLM_BACKEND=openrouter

# Optional defaults for cross-provider/model subagents.
# Must be set together if used.
RLM_SUBAGENT_BACKEND=openai
RLM_SUBAGENT_MODEL=gpt-4.1-mini

# Optional runtime safety guards (positive integers):
RLM_MAX_ITERATIONS=20
RLM_MAX_SUBAGENT_CALLS=40

# Optional long-context controls:
RLM_DIRECT_CHUNKING_ENABLED=true
RLM_DIRECT_CHUNK_OVERLAP_TOKENS=256
RLM_DIRECT_CHUNK_MAX_CHUNKS=64
RLM_OPENROUTER_MIDDLE_OUT_FALLBACK=true
RLM_SUBAGENT_ROOT_COMPACTION_ENABLED=true
RLM_SUBAGENT_COMPACTION_THRESHOLD_PCT=0.75
```

Supported backends:
- `openai`
- `openrouter`
- `anthropic`
- `gemini`
- `portkey`
- `vercel`
- `azure_openai`
- `litellm`
- `vllm`

If `RLM_BACKEND` is omitted, the app auto-detects the first configured backend in this order:
`openrouter -> openai -> anthropic -> gemini -> portkey -> vercel -> azure_openai`.

## Quick Start

1. Put one or more documents in `embed-docs/`.
   Supported types include `.pdf`, `.docx`, `.txt`, `.md`, and other text-like files.
2. Run CLI:

```bash
uv run main.py
```

3. Or run web UI:

```bash
uv run main.py web
```

This command automatically builds `frontend/` before starting FastAPI.

Alternative one-command flow (tests + frontend build + backend start):

```bash
./scripts/start_fullstack.sh
```

4. Follow CLI prompts:
   - choose a document by number
   - or choose `a` to load all documents in `embed-docs/`
   - choose model (or accept default)
   - enable/disable subagents
   - optionally load `prompts.md`
   - choose one-time mode or allow follow-up queries
   - enter your question
   - optionally save response

While a query is running, the CLI shows a live spinner + elapsed time.
After each answer, it prints run metrics (tokens, iterations, subagent calls, and per-model usage).
If subagents are enabled but `Subagent Calls` is `0`, the run stayed on the root model only.
Progress updates run in an isolated worker process to keep elapsed-time UI responsive during long calls.
Press `Ctrl+C` during processing to cancel the active query cleanly without a traceback.

## Web UI

- Open `http://localhost:8000` after running `uv run main.py web`.
- Frontend assets are compiled from SolidJS (`frontend/`) on web startup.
- Query responses stream live from `/api/query/stream`.
- Sidebar includes per-run toggles for direct chunking, OpenRouter middle-out retry,
  and subagent root compaction behavior.
- Streaming logs are split into two panels:
  - `Agent Output`: document loading, iteration/completion, and code execution events.
  - `Subagent Output`: recursive subagent call prompt/response events.
- Final answer and metrics are shown in the main chat once the run completes.

## Prompt File (`prompts.md`)

You can preload a system instruction and default query from file.

Example:

```markdown
# System
You are a precise technical analyst. Answer from the provided document context.

# Query
Summarize the document, then list key risks and next actions.
```

Parser behavior:
- accepts `# Header` and `## Header`
- supports section names like `System` and `Query` (case-insensitive lookup in CLI)
- ignores missing files by returning an empty prompt map
- in subagent mode, your `System` text is appended to the built-in RLM system prompt

## Execution Modes

### Direct LM Mode (`use_subagents = false`)

- Sends a single completion request with prompt + context.
- Faster and cheaper.
- Good default for straightforward extraction/summarization tasks.
- If prompt+context exceeds the model's token budget, it automatically switches to
  map-reduce chunking (chunk summaries, then final synthesis) instead of failing.

Chunking knobs for `/api/query` and `/api/query/stream` request bodies:
- `direct_chunking_enabled` (default: `true`)
- `direct_chunk_overlap_tokens` (default: `256`)
- `direct_chunk_max_chunks` (default: `64`)
- `openrouter_middle_out_fallback` (default: `true`)
- `subagent_root_compaction_enabled` (default: `true`)
- `subagent_compaction_threshold_pct` (default: `0.75`)

How sizing works:
- The handler estimates each model's context limit automatically and derives a per-request token budget.
- `direct_chunk_max_chunks` is **not** token limit; it is only a safety cap on how many chunks may be generated.
- Different OpenRouter models can have different context sizes; budget calculation adapts per selected model.

### RLM Subagent Mode (`use_subagents = true`)

- Uses `rlm.RLM(..., max_depth=1)` with:
  - `prompt=context`
  - `root_prompt=user_question`
- Adds root-history compaction protection by default:
  - `compaction=True`
  - `compaction_threshold_pct=0.75` (configurable)
- Recursive `llm_query` calls still happen even when no alternate subagent model is configured.
  In that case, depth-1 calls run on the same root backend/model.
- Supports optional separate subagent backend/model via:
  - `other_backends=[subagent_backend]`
  - `other_backend_kwargs=[{\"model_name\": subagent_model}]`
- Can improve complex reasoning over long/technical material.
- Usually slower and more expensive.
- If the model returns a `FINAL_VAR` missing-variable error, the tool retries once automatically.
- If retry still fails, the tool falls back to direct completion and marks this in run metrics.

### Cross-Provider Subagents

When subagents are enabled, the CLI can route recursive calls to a different provider/model.
Example:
- root model: `openrouter` + `openai/gpt-5-mini`
- subagent model: `openai` + `gpt-4.1-mini`

Requirements:
- both `RLM_SUBAGENT_BACKEND` and `RLM_SUBAGENT_MODEL` can be pre-set in `.env`, or entered interactively
- the selected subagent backend must be fully configured in `.env` (API key and backend-specific extras when needed)
- same-provider subagents are supported (for example `openrouter -> openrouter` with a different model)
- optional guardrails:
  - `RLM_MAX_ITERATIONS`: caps root RLM loop iterations
  - `RLM_MAX_SUBAGENT_CALLS`: aborts runs that exceed subagent-call budget

## Output Saving

After each response, you can save in:
- `md`: saves `# Query: ...` + response text
- `pdf`: renders markdown content to PDF
- `both`: writes both file types

Filename defaults to `response` if left blank.
All saved files are written under the project `response/` directory.

## Testing

Run all current tests:

```bash
uv run python -m pytest -q
```

Test coverage includes:
- document listing and extraction behavior
- RLM/direct-LM call routing and payload format
- prompt file parsing
- markdown/PDF output generation
- runtime metrics aggregation for direct and subagent modes

## Troubleshooting

- `No API key found`:
  - Set `RLM_BACKEND` and required backend credentials in `.env`.
- `RLM_BACKEND=... but ..._API_KEY is missing`:
  - Either add the matching key or change `RLM_BACKEND`.
- `No supported documents found in 'embed-docs/'`:
  - Add files such as `.pdf`, `.docx`, `.txt`, or `.md` into `embed-docs/`.
- Empty or weak answers:
  - Try a stronger model, add a `System` prompt, or enable subagents.
- Very long-running subagent sessions:
  - Set `RLM_MAX_ITERATIONS` and/or `RLM_MAX_SUBAGENT_CALLS` in `.env`.
- `maximum context length` / OpenRouter 400 errors:
  - Keep `direct_chunking_enabled=true` (default).
  - Keep `openrouter_middle_out_fallback=true` to retry once with OpenRouter transform before chunking.
  - In subagent mode, keep `subagent_root_compaction_enabled=true` and lower
    `subagent_compaction_threshold_pct` (for example `0.65`) if root context still grows too large.
  - Lower selected document count, or raise `direct_chunk_max_chunks` for very large corpora.
  - For complex long-context reasoning, prefer `use_subagents=true` so RLM can work from REPL context instead of one giant prompt.

## Notes

- This tool is intentionally CLI-first and local-document focused.
- It does not persist conversation state between runs.
- For production-scale retrieval, consider adding chunking + vector search before RLM querying.
