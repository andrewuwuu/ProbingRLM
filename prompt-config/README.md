# Prompt Config

Use one file: `config.md`.

Copy `config.sample.md` to `config.md`, then edit only the sections you want:

- `# Question`
- `# Scope`
- `# System`
- `# Custom Prompt`
- `# Signature`
- `# Output Template`

Notes:

- Missing sections are ignored.
- `prompts.md` is still supported, but disabled by default (`DSPY_USE_PROMPTS_FILE=false`).
- Legacy per-file config (`question.md`, `scope.md`, `system.md`, `custom.md`, `signature.txt`, `output_template.md`) still works if you already use it.
