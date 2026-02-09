# RLM PDF Retrieval Tool

`rlm-test` is a terminal-based PDF question-answering tool built on the `rlms` library.
It loads a local PDF, lets you ask natural-language questions, and can save responses to
Markdown or PDF for reporting and handoff.

The project is designed to stay practical:
- minimal setup
- explicit backend/key handling
- simple interactive workflow
- test coverage around core behavior

## What It Does

- Scans `embed-docs/` for PDF files.
- Extracts PDF text using `pypdf`.
- Sends your query plus document context to an LLM via `rlms`.
- Supports both direct LM mode and RLM subagent mode.
- Optionally loads reusable prompts from `prompts.md`.
- Saves model outputs as `.md`, `.pdf`, or both.

## How It Works

The CLI entrypoint is `main.py`.

1. Environment variables are loaded from `.env`.
2. Backend and API key are resolved (`openrouter` or `openai`).
3. You pick a PDF and model.
4. You choose execution mode:
   - direct LM mode (`use_subagents = false`)
   - RLM subagent mode (`use_subagents = true`)
   - optional cross-provider/model subagent routing
5. The response is printed and can be saved to disk.

Implementation files:
- `main.py`: interactive flow and orchestration
- `src/rlm_handler.py`: backend/model calls and RLM behavior
- `src/pdf_utils.py`: file discovery + PDF text extraction
- `src/prompt_loader.py`: markdown prompt section parsing
- `src/output_utils.py`: markdown/PDF output writing
- `tests/`: behavior checks for retrieval, prompts, and output generation

## Requirements

- Python `>=3.11`
- `uv` for dependency management
- At least one API key:
  - `OPENROUTER_API_KEY`, or
  - `OPENAI_API_KEY`

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

# Optional explicit backend selection: openrouter|openai
RLM_BACKEND=openrouter

# Optional defaults for cross-provider/model subagents.
# Must be set together if used.
RLM_SUBAGENT_BACKEND=openai
RLM_SUBAGENT_MODEL=gpt-4.1-mini
```

If `RLM_BACKEND` is omitted, the app prefers OpenRouter when both keys exist.

## Quick Start

1. Put one or more PDFs in `embed-docs/`.
2. Run:

```bash
uv run main.py
```

3. Follow prompts:
   - choose a PDF by number
   - or choose `a` to load all PDFs in `embed-docs/`
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

### RLM Subagent Mode (`use_subagents = true`)

- Uses `rlm.RLM(..., max_depth=1)` with:
  - `prompt=context`
  - `root_prompt=user_question`
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
- the selected subagent backend must have a valid API key in `.env`

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
uv run python tests/verify_retrieval.py
uv run python tests/test_prompt_loader.py
uv run python tests/test_pdf_gen.py
```

Test coverage includes:
- PDF listing and extraction behavior
- RLM/direct-LM call routing and payload format
- prompt file parsing
- markdown/PDF output generation
- runtime metrics aggregation for direct and subagent modes

## Troubleshooting

- `No API key found`:
  - Set `OPENROUTER_API_KEY` or `OPENAI_API_KEY` in `.env`.
- `RLM_BACKEND=... but ..._API_KEY is missing`:
  - Either add the matching key or change `RLM_BACKEND`.
- `No PDF files found in 'embed-docs/'`:
  - Add at least one `.pdf` into `embed-docs/`.
- Empty or weak answers:
  - Try a stronger model, add a `System` prompt, or enable subagents.

## Notes

- This tool is intentionally CLI-first and local-document focused.
- It does not persist conversation state between runs.
- For production-scale retrieval, consider adding chunking + vector search before RLM querying.
