#!/usr/bin/env python3
"""
updatedb_docs.py
===========
Walk a DOCS directory, extract text from PDF / DOCX / Markdown (and plain .txt),
chunk it with character-aware splits, embed, and store in a ChromaDB
collection (e.g. "all_documents"), avoiding duplicate IDs and oversize requests.

"""
from __future__ import annotations
import os
import sys
import re
from pypdf import PdfReader
import docx
from typing import List, Dict, Any, Iterable, Tuple
from config import (
    CHROMA_DB_DIR,
    GLOBAL_RAGDATA_MAP,
    RAGType,
)
from dotenv import load_dotenv
from path_utils import encode_path
from updatedb_helper import (
    uniquify_records,
    short_hash,
    get_existing_ids,
    embed_and_add,
    split_by_chars,
)
from chromadb_shim import chromadb

load_dotenv()

# Character-based chunk size for documents
MAX_CHUNK_CHARS = 2000  # Safe size for retrieval (approx 500 tokens)
OVERLAP_CHARS = 200     # 10% overlap

SUPPORTED_EXTS = {".pdf", ".docx", ".md", ".markdown", ".txt"}

CHROMA_DB_FULL_PATH = os.path.expanduser(CHROMA_DB_DIR)

# ------------- Hash / ID helpers -------------

# ------------- File readers -------------

def read_markdown(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path: str) -> List[Tuple[int, List[str]]]:
    """Return list of (page_number, [lines...])"""
    reader = PdfReader(path)
    out: List[Tuple[int, List[str]]] = []
    for i, page in enumerate(reader.pages):
        if i % 50 == 0:
            print(f"Processing pdf page {i}")
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        # split into lines
        lines = txt.splitlines()
        out.append((i + 1, lines))
    return out

def read_docx(path: str) -> str:
    d = docx.Document(path)
    parts = []
    for p in d.paragraphs:
        parts.append(p.text)
    return "\n".join(parts)

# ------------- Chunk builders -------------

def chunk_text_generic(
    text: str,
    path: str,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap_chars: int = OVERLAP_CHARS
) -> Iterable[Dict[str, Any]]:
    """Character-based sliding window for generic/plaintext content."""
    if not text.strip():
        return

    stride = max(1, max_chars - overlap_chars)

    for start in range(0, len(text), stride):
        piece = text[start:start + max_chars].strip()
        if not piece:
            continue
        end = start + len(piece)
        cid = f"{path}:char{start}-{end}-{short_hash(piece)}"
        yield {
            "id": cid,
            "text": piece,
            "metadata": {
                "file_path": encode_path(path),
                "start_char": start,
                "end_char": end,
                "source_type": "doc",
            },
        }


def chunk_markdown(
    text: str,
    path: str,
    max_chars: int = MAX_CHUNK_CHARS
) -> Iterable[Dict[str, Any]]:
    """Split on top-level headings (#, ##, ###) first, then char-split if needed."""
    blocks = re.split(r"^(?=#)", text, flags=re.MULTILINE)
    for idx, block in enumerate(blocks):
        block = block.strip()
        if not block:
            continue
        # further char-split if needed
        pieces = [block] if len(block) <= max_chars else split_by_chars(block, max_chars)
        for j, piece in enumerate(pieces):
            cid = f"{path}:md{idx}-{j}-{short_hash(piece)}"
            yield {
                "id": cid,
                "text": piece,
                "metadata": {
                    "file_path": encode_path(path),
                    "block_index": idx,
                    "piece_index": j,
                    "source_type": "doc",
                    "format": "markdown",
                },
            }


def chunk_pdf_pages(
    pages: List[Tuple[int, List[str]]],
    path: str,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> Iterable[Dict[str, Any]]:
    """Character-based PDF chunker with overlap to preserve context."""
    stride = max(1, max_chars - overlap_chars)

    for page_no, lines in pages:
        page_text = "\n".join(lines).strip()
        if not page_text:
            continue

        for start in range(0, len(page_text), stride):
            piece = page_text[start:start + max_chars].strip()
            if not piece:
                continue
            end = start + len(piece)
            cid = f"{path}:p{page_no}-char{start}-{end}-{short_hash(piece)}"
            yield {
                "id": cid,
                "text": piece,
                "metadata": {
                    "file_path": encode_path(path),
                    "source_type": "doc",
                    "format": "pdf",
                    "page_number": page_no,
                    "start_char": start,
                    "end_char": end,
                },
            }


# ------------- Main pipeline -------------

def build_records_for_file(path: str) -> List[Dict[str, Any]]:
    print(f"Building records for {path}")
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".md", ".markdown"):
            txt = read_markdown(path)
            return list(chunk_markdown(txt, path))
        elif ext == ".pdf":
            pages = read_pdf(path)
            return list(chunk_pdf_pages(pages, path))
        elif ext == ".docx":
            txt = read_docx(path)
            return list(chunk_text_generic(txt, path))
        elif ext == ".txt":
            txt = read_txt(path)
            return list(chunk_text_generic(txt, path))
        elif ext == ".doc":
            print(f"[WARN] .doc not supported natively ({path}), skipping or convert to .docx")
            return []
        else:
            return []
    except Exception as e:
        print(f"[ERROR] Failed to process {path}: {e}")
        return []


def walk_docs(root_dir: str) -> List[Dict[str, Any]]:
    all_records: List[Dict[str, Any]] = []
    for dirpath, _, files in os.walk(root_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in SUPPORTED_EXTS:
                p = os.path.join(dirpath, name)
                recs = build_records_for_file(p)
                print(f"Adding #{len(recs)} records of file {name}")
                all_records.extend(recs)
    return all_records


def update_docs_collection(db_client, name, full_path):
    print(f"Updating docs collection {name}")
    col = db_client.get_or_create_collection(
        name,
        metadata={"hnsw:space": "cosine"}
    )

    records = walk_docs(full_path)
    print(f"Found {len(records)} records in total")

    # De-dup vs existing
    existing = get_existing_ids(col)
    uniq_records = uniquify_records(records, already_seen=set(existing))
    print(f"After dedup: {len(uniq_records)} new chunks to embed")

    # Embed & store
    embed_and_add(uniq_records, col, verbose=True)
    print("Done.")

def main():
    valid_names = ", ".join(
        sorted(key for key, (_, entry_type) in GLOBAL_RAGDATA_MAP.items() if entry_type == RAGType.DOC)
    )

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python updatedb_docs.py <collection name>")
        print(f"  Valid names are: {valid_names}")
        cname='DOCS'
#        return
    else:
        cname = sys.argv[1]

    rag_entry = GLOBAL_RAGDATA_MAP.get(cname)
    if rag_entry is None:
        print(f"[ERROR] Unknown collection '{cname}'. Valid options are: {valid_names}")
        return
    doc_path, _ = rag_entry
    client = chromadb.PersistentClient(path=CHROMA_DB_FULL_PATH)
    update_docs_collection(client, cname, doc_path)

if __name__ == "__main__":
    main()
