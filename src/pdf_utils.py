import os
import shutil
import subprocess
from pypdf import PdfReader

TEXT_LIKE_EXTENSIONS = {
    ".doc",
    ".docs",
    ".txt",
    ".md",
    ".markdown",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".csv",
    ".tsv",
    ".xml",
    ".html",
    ".htm",
    ".ini",
    ".toml",
    ".log",
    ".rst",
    ".sql",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
}


def _is_probably_binary(data: bytes) -> bool:
    if not data:
        return False

    # UTF-16 often contains NUL bytes; allow if BOM exists.
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return False

    if b"\x00" in data:
        return True

    text_bytes = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(32, 127)))
    non_text = data.translate(None, text_bytes)
    return (len(non_text) / len(data)) > 0.30


def _decode_text_bytes(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _load_plain_text(file_path: str) -> str:
    try:
        with open(file_path, "rb") as handle:
            data = handle.read()
    except Exception as e:
        print(f"Error reading text file {file_path}: {e}")
        return ""

    if _is_probably_binary(data):
        print(f"Skipping binary/non-text file: {file_path}")
        return ""

    return _decode_text_bytes(data).strip()


def _load_docx(file_path: str) -> str:
    try:
        from docx import Document  # type: ignore
    except Exception:
        print(
            "python-docx is required to read .docx files. Install dependencies via `uv sync`."
        )
        return ""

    try:
        document = Document(file_path)
        paragraphs = [p.text for p in document.paragraphs if p.text and p.text.strip()]
        return "\n".join(paragraphs).strip()
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
        return ""


def _load_doc(file_path: str) -> str:
    """
    Try common CLI converters for legacy .doc first, then fall back to plain-text decode.
    """
    converters = (
        ["antiword", file_path],
        ["catdoc", file_path],
    )
    for cmd in converters:
        executable = cmd[0]
        if not shutil.which(executable):
            continue
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                check=False,
                timeout=10,
            )
        except Exception:
            continue
        if result.returncode == 0 and result.stdout:
            text = result.stdout.decode("utf-8", errors="ignore").strip()
            if text:
                return text

    fallback = _load_plain_text(file_path)
    if fallback:
        return fallback

    print(
        f"Could not extract .doc file {file_path}. "
        "Install antiword/catdoc or convert it to .docx."
    )
    return ""


def list_documents(directory: str) -> list[str]:
    """List supported docs from a directory (pdf/docx/text-like/any probable text file)."""
    if not os.path.isdir(directory):
        return []

    supported_by_ext = {".pdf", ".docx"} | TEXT_LIKE_EXTENSIONS
    docs: list[str] = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in supported_by_ext:
            docs.append(name)
            continue
        # Allow any text-like file regardless of extension.
        try:
            with open(path, "rb") as handle:
                sample = handle.read(2048)
        except Exception:
            continue
        if not _is_probably_binary(sample):
            docs.append(name)

    return sorted(docs, key=str.lower)


def load_document(file_path: str) -> str:
    """Extract text from supported docs or any text-like file."""
    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".pdf":
        return load_pdf(file_path)
    if extension == ".docx":
        return _load_docx(file_path)
    if extension in {".doc", ".docs"}:
        return _load_doc(file_path)
    return _load_plain_text(file_path)


def list_pdfs(directory: str) -> list[str]:
    """Backward-compatible helper for PDF-only listings."""
    return [
        name for name in list_documents(directory) if os.path.splitext(name)[1].lower() == ".pdf"
    ]


def load_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text_parts: list[str] = []
        for page_index, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            text_parts.append(f"[Page {page_index}]\n{page_text}".strip())
        return "\n".join(text_parts).strip()
    except Exception as e:
        message = str(e)
        if "cryptography>=3.1 is required for AES algorithm" in message:
            print(
                f"Error reading PDF {file_path}: {message}. "
                "Install dependencies with `uv sync` (or `uv add cryptography`)."
            )
            return ""
        print(f"Error reading PDF {file_path}: {message}")
        return ""
