"""Knowledge-base loading and section-aware chunking (no model dependencies, so it can be unit-tested)."""
import re
from pathlib import Path

from . import config


def load_docs(kb_dir: Path = config.KB_DIR) -> list[dict]:
    docs = []
    for p in sorted(kb_dir.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        m = re.search(r"^# (.+)$", text, flags=re.M)
        docs.append({"doc_id": p.stem, "title": m.group(1).strip() if m else p.stem, "text": text})
    return docs


def full_kb_text(kb_dir: Path = config.KB_DIR) -> str:
    return "\n\n".join(d["text"].strip() for d in load_docs(kb_dir))


def _sections(doc: dict) -> list[tuple[str, str]]:
    """Split a document on '## ' headings; returns (heading, body) pairs."""
    parts = re.split(r"^## (.+)$", doc["text"], flags=re.M)
    out = []
    intro = re.sub(r"^# .+$", "", parts[0], flags=re.M).strip()
    if intro:
        out.append(("Overview", intro))
    for i in range(1, len(parts), 2):
        out.append((parts[i].strip(), parts[i + 1].strip()))
    return out


def chunk_docs(docs: list[dict], size: int = config.CHUNK_WORDS, overlap: int = config.CHUNK_OVERLAP) -> list[dict]:
    """Section-aware chunks; long sections are split into overlapping word windows.

    Each chunk is prefixed with 'Document title > Section heading' so the chunk is self-describing.
    """
    chunks = []
    for doc in docs:
        for heading, body in _sections(doc):
            words = body.split()
            step = max(size - overlap, 1)
            starts = range(0, max(len(words) - overlap, 1), step) if len(words) > size else [0]
            for j, s in enumerate(starts):
                piece = " ".join(words[s:s + size])
                chunks.append({
                    "chunk_id": f"{doc['doc_id']}#{len(chunks)}",
                    "doc_id": doc["doc_id"],
                    "section": heading,
                    "text": f"{doc['title']} > {heading}\n{piece}",
                })
    return chunks
