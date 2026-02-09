import os
from pypdf import PdfReader


def list_pdfs(directory: str) -> list[str]:
    """List all PDF files in the given directory."""
    if not os.path.exists(directory):
        return []
    return sorted(f for f in os.listdir(directory) if f.lower().endswith(".pdf"))


def load_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text_parts: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
        return "\n".join(text_parts).strip()
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ""
