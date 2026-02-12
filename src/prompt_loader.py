import os
import re
from collections.abc import Mapping

TEMPLATE_PATTERN = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")

def load_prompts(file_path: str) -> dict[str, str]:
    """
    Parses a markdown file for sections formatted as headers (e.g., # System).
    Returns a dictionary mapping section names to their content.
    """
    if not os.path.exists(file_path):
        return {}

    prompts: dict[str, str] = {}
    current_section: str | None = None
    current_content: list[str] = []

    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped_line = line.strip()
                if stripped_line.startswith("#"):
                    header_text = stripped_line.lstrip("#").strip()
                    if not header_text:
                        continue

                    if current_section:
                        prompts[current_section] = "\n".join(current_content).strip()

                    current_section = header_text
                    current_content = []
                else:
                    if current_section:
                        current_content.append(line.rstrip())

            if current_section:
                prompts[current_section] = "\n".join(current_content).strip()

    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return {}

    return prompts


def extract_template_variables(template: str | None) -> list[str]:
    if not template:
        return []
    found = TEMPLATE_PATTERN.findall(template)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in found:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def render_template(template: str | None, variables: Mapping[str, str]) -> str | None:
    if template is None:
        return None

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        value = variables.get(key)
        if value is None:
            return match.group(0)
        return str(value)

    return TEMPLATE_PATTERN.sub(replace, template)


def parse_prompt_variables(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}

    variables: dict[str, str] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue
        variables[key] = value.strip()
    return variables
