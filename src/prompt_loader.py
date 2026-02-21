import os


def load_prompts(file_path: str) -> dict[str, str]:
    """
    Parses a markdown file for sections formatted as headers (e.g., # System).
    Returns a dictionary mapping section names to their content.
    """
    if not os.path.exists(file_path):
        return {}

    prompts = {}
    current_section = None
    current_content = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line.startswith('#'):
                    header_text = stripped_line.lstrip('#').strip()
                    if not header_text:
                        continue

                    if current_section:
                        prompts[current_section] = '\n'.join(current_content).strip()

                    current_section = header_text
                    current_content = []
                else:
                    if current_section:
                        current_content.append(line.rstrip())

            if current_section:
                prompts[current_section] = '\n'.join(current_content).strip()

    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return {}
        
    return prompts
