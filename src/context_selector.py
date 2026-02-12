import re
import math
from dataclasses import dataclass


_PAGE_MARKER_PATTERN = re.compile(r"\[Page\s+(\d+)\]", re.IGNORECASE)
_WORD_PATTERN = re.compile(r"\w{2,}", re.UNICODE)


@dataclass
class _Chunk:
    file_name: str
    index: int
    text: str
    score: float


def _extract_query_terms(query: str, max_terms: int = 120) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for raw in _WORD_PATTERN.findall((query or "").lower()):
        if raw in seen:
            continue
        seen.add(raw)
        terms.append(raw)
        if len(terms) >= max_terms:
            break
    return terms


def _char_embedding(
    text: str,
    dims: int = 2048,
    min_n: int = 3,
    max_n: int = 5,
    max_ngrams: int = 12000,
) -> dict[int, float]:
    lowered = (text or "").lower()
    vec: dict[int, float] = {}
    ngrams_seen = 0
    for n in range(min_n, max_n + 1):
        if len(lowered) < n:
            continue
        for i in range(0, len(lowered) - n + 1):
            ng = lowered[i : i + n]
            if ng.isspace():
                continue
            if ngrams_seen >= max_ngrams:
                break
            idx = hash(ng) % dims
            vec[idx] = vec.get(idx, 0.0) + 1.0
            ngrams_seen += 1
        if ngrams_seen >= max_ngrams:
            break

    norm = math.sqrt(sum(value * value for value in vec.values()))
    if norm <= 0:
        return {}
    return {idx: value / norm for idx, value in vec.items()}


def _cosine_sparse(a: dict[int, float], b: dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    score = 0.0
    for idx, value in a.items():
        score += value * b.get(idx, 0.0)
    return score


def _split_text_chunks(
    file_name: str,
    text: str,
    chunk_chars: int,
) -> list[tuple[str, int]]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text or "") if item.strip()]
    chunks: list[tuple[str, int]] = []
    buf: list[str] = []
    buf_len = 0
    page_hint = 0

    def flush() -> None:
        nonlocal buf, buf_len, page_hint
        if not buf:
            return
        chunk_text = "\n\n".join(buf).strip()
        if chunk_text:
            chunks.append((chunk_text, page_hint))
        buf = []
        buf_len = 0

    for paragraph in paragraphs:
        match = _PAGE_MARKER_PATTERN.search(paragraph)
        if match:
            try:
                page_hint = int(match.group(1))
            except ValueError:
                pass

        if len(paragraph) > chunk_chars:
            flush()
            cursor = 0
            while cursor < len(paragraph):
                part = paragraph[cursor : cursor + chunk_chars]
                chunk_text = part.strip()
                if chunk_text:
                    chunks.append((chunk_text, page_hint))
                cursor += chunk_chars
            continue

        if buf_len and buf_len + len(paragraph) + 2 > chunk_chars:
            flush()
        buf.append(paragraph)
        buf_len += len(paragraph) + 2

    flush()
    return chunks


def _score_chunk(
    chunk_text: str,
    query_terms: list[str],
    query_vec: dict[int, float],
) -> float:
    if not query_vec:
        return 0.0
    emb_score = _cosine_sparse(query_vec, _char_embedding(chunk_text))
    lexical_hits = 0
    lowered = chunk_text.lower()
    for term in query_terms:
        if term and term in lowered:
            lexical_hits += 1
    lexical_bonus = min(0.6, lexical_hits * 0.04)
    return float(emb_score + lexical_bonus)


def build_query_context(
    documents: list[tuple[str, str]],
    query: str,
    max_chars: int,
    chunk_chars: int = 2200,
    max_chunks: int = 120,
) -> tuple[str, dict[str, int]]:
    """
    Build query-focused context from document texts with a character budget.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be > 0")
    if max_chunks <= 0:
        raise ValueError("max_chunks must be > 0")

    raw_blocks = [
        f"===== BEGIN FILE: {name} =====\n{text}\n===== END FILE: {name} ====="
        for name, text in documents
        if text and text.strip()
    ]
    raw_context = "\n\n".join(raw_blocks)
    if len(raw_context) <= max_chars:
        return raw_context, {
            "selected_chars": len(raw_context),
            "selected_chunks": 0,
            "available_chunks": 0,
            "truncated": 0,
        }

    query_terms = _extract_query_terms(query)
    query_vec = _char_embedding(query, max_ngrams=20000)
    all_chunks: list[_Chunk] = []
    for file_name, text in documents:
        split_chunks = _split_text_chunks(file_name=file_name, text=text, chunk_chars=chunk_chars)
        for idx, (chunk_text, page_hint) in enumerate(split_chunks, start=1):
            score = _score_chunk(chunk_text, query_terms, query_vec)
            if page_hint > 0:
                score += 0.1
            all_chunks.append(
                _Chunk(
                    file_name=file_name,
                    index=idx,
                    text=chunk_text,
                    score=score,
                )
            )

    if not all_chunks:
        return "", {
            "selected_chars": 0,
            "selected_chunks": 0,
            "available_chunks": 0,
            "truncated": 1,
        }

    ranked = sorted(
        all_chunks,
        key=lambda item: (item.score, len(item.text)),
        reverse=True,
    )
    if query_vec:
        ranked = [item for item in ranked if item.score > 0] + [item for item in ranked if item.score <= 0]

    selected: list[_Chunk] = []
    used = 0
    for chunk in ranked:
        if len(selected) >= max_chunks:
            break
        block_len = len(chunk.text) + len(chunk.file_name) + 64
        if selected and used + block_len > max_chars:
            continue
        if not selected and block_len > max_chars:
            selected.append(chunk)
            used = block_len
            break
        if used + block_len <= max_chars:
            selected.append(chunk)
            used += block_len
        if used >= max_chars:
            break

    if not selected:
        selected = ranked[:1]

    position_map = {(item.file_name, item.index): pos for pos, item in enumerate(all_chunks)}
    selected = sorted(selected, key=lambda item: position_map[(item.file_name, item.index)])

    grouped: dict[str, list[_Chunk]] = {}
    file_order: list[str] = []
    for chunk in selected:
        if chunk.file_name not in grouped:
            grouped[chunk.file_name] = []
            file_order.append(chunk.file_name)
        grouped[chunk.file_name].append(chunk)

    sections: list[str] = []
    for file_name in file_order:
        sections.append(f"===== BEGIN FILE: {file_name} =====")
        for chunk in grouped[file_name]:
            sections.append(
                f"[Excerpt {chunk.index} | score={chunk.score:.1f}]"
            )
            sections.append(chunk.text)
        sections.append(f"===== END FILE: {file_name} =====")

    context = "\n".join(sections).strip()
    return context, {
        "selected_chars": len(context),
        "selected_chunks": len(selected),
        "available_chunks": len(all_chunks),
        "truncated": 1,
    }
