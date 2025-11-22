#!/usr/bin/env python3
"""
updatedb_docs.py
===========
Walk a DOCS directory, extract text from PDF / DOCX / Markdown (and plain .txt),
chunk it with token-aware splits, embed with OpenAI, and store in a ChromaDB
collection (e.g. "all_documents"), avoiding duplicate IDs and oversize requests.

"""
from __future__ import annotations
import os
import sys
import re
import math
import json
import hashlib
import argparse
from pypdf import PdfReader
import docx
from typing import List, Dict, Any, Iterable, Tuple, Optional
from config import (
    USE_OPENAI,
    CHROMA_DB_DIR,
    EMBEDDING_MODEL,
    GLOBAL_RAGDATA_MAP,
    OLLAMA_BASE_URL,
    RAGType,
)
from dotenv import load_dotenv
from path_utils import encode_path
from tokenizer_utils import split_by_tokens, encode as encode_tokens, decode as decode_tokens
from updatedb_helper import embed_record_with_retry, uniquify_records, token_len, short_hash

# ------------- Third-party deps -------------
import chromadb
from embed_client import EmbedClientUni
load_dotenv()


if USE_OPENAI:
    MAX_ITEM_TOKENS     = 900    # tighter doc chunks keep grounding sharp for 4o-mini
    MAX_REQUEST_TOKENS  = 4500   # ~5 chunks per embed request with headroom
    DEFAULT_BATCH_LIMIT = 4500   # matches the per-request cap
else:
    # Optimized for Qwen3-4B (256K context window)
    # See TOKEN_LIMITS_GUIDE.md for rationale
    MAX_ITEM_TOKENS = 2048                   # Maximum tokens per chunk (balanced for RAG)
    MAX_REQUEST_TOKENS = 8192                # Maximum tokens per batch request
    DEFAULT_BATCH_LIMIT = 8192               # Sum of tokens per request

embed_client = EmbedClientUni(use_openai = USE_OPENAI)

SUPPORTED_EXTS = {".pdf", ".docx", ".md", ".markdown", ".txt"}  # add more if needed

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

def chunk_text_generic(text: str, path: str, max_tokens: int = MAX_ITEM_TOKENS, overlap_tokens: int = 100) -> Iterable[Dict[str, Any]]:
    """Token slide window for generic/plaintext content."""
    try:
        toks = encode_tokens(text)
    except RuntimeError:
        toks = None

    if toks is None:
        # naive char-based fallback
        step = max_tokens * 4
        for i in range(0, len(text), step - overlap_tokens * 4):
            piece = text[i:i+step]
            cid = f"{path}:char{i}-{i+len(piece)}-{short_hash(piece)}"
            yield {
                "id": cid,
                "text": piece,
                "metadata": {
                    "file_path": encode_path(path),
                    "start_char": i,
                    "end_char": i + len(piece),
                    "source_type": "doc",
                },
            }
        return

    step = max_tokens
    ov   = overlap_tokens
    for i in range(0, len(toks), step - ov):
        piece_toks = toks[i:i+step]
        try:
            piece = decode_tokens(piece_toks)
        except RuntimeError:
            # Fallback if decode fails
            piece = text[i*4:(i+len(piece_toks))*4]
        cid = f"{path}:tok{i}-{i+len(piece_toks)}-{short_hash(piece)}"
        yield {
            "id": cid,
            "text": piece,
            "metadata": {
                "file_path": encode_path(path),
                "start_token": i,
                "end_token": i + len(piece_toks),
                "source_type": "doc",
            },
        }

def chunk_markdown(text: str, path: str, max_tokens: int = MAX_ITEM_TOKENS) -> Iterable[Dict[str, Any]]:
    """Split on top-level headings (#, ##, ###) first, then token-split."""
    blocks = re.split(r"^(?=#)" , text, flags=re.MULTILINE)  # keep headings at block start
    for idx, block in enumerate(blocks):
        block = block.strip()
        if not block:
            continue
        # further token-split if needed
        pieces = [block] if token_len(block) <= max_tokens else split_by_tokens(block, max_tokens)
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
    max_tokens: int = MAX_ITEM_TOKENS,
    overlap_tokens: int = 100,
) -> Iterable[Dict[str, Any]]:
    """Token-based PDF chunker with overlap to preserve context."""

    token_stride = max(1, max_tokens - overlap_tokens)

    def _char_chunks(page_text: str) -> Iterable[Tuple[str, int, int]]:
        """Fallback slicer when tokenization fails."""
        char_window = max_tokens * 4
        char_stride = max(1, char_window - overlap_tokens * 4)
        for start in range(0, len(page_text), char_stride):
            piece = page_text[start:start + char_window].strip()
            if not piece:
                continue
            end = start + len(piece)
            yield piece, start, end

    for page_no, lines in pages:
        page_text = "\n".join(lines).strip()
        if not page_text:
            continue

        try:
            tokens = encode_tokens(page_text)
        except RuntimeError:
            for piece, start_char, end_char in _char_chunks(page_text):
                cid = f"{path}:p{page_no}-char{start_char}-{end_char}-{short_hash(piece)}"
                yield {
                    "id": cid,
                    "text": piece,
                    "metadata": {
                        "file_path": encode_path(path),
                        "source_type": "doc",
                        "format": "pdf",
                        "page_number": page_no,
                        "start_char": start_char,
                        "end_char": end_char,
                    },
                }
            continue

        if not tokens:
            continue

        for start in range(0, len(tokens), token_stride):
            window = tokens[start:start + max_tokens]
            if not window:
                continue
            try:
                chunk_text = decode_tokens(window)
            except RuntimeError:
                chunk_text = page_text
            end_token = start + len(window)
            cid = f"{path}:p{page_no}-tok{start}-{end_token}-{short_hash(chunk_text)}"
            yield {
                "id": cid,
                "text": chunk_text,
                "metadata": {
                    "file_path": encode_path(path),
                    "source_type": "doc",
                    "format": "pdf",
                    "page_number": page_no,
                    "start_token": start,
                    "end_token": end_token,
                },
            }
# ------------- De-dup helpers -------------

def get_all_ids(col) -> set[str]:
    total = col.count()
    if total == 0:
        return set()
    ids = []
    offset = 0
    LIMIT = 5000
    while offset < total:
        res = col.get(include=[], limit=min(LIMIT, total - offset), offset=offset)
        ids.extend(res["ids"])
        offset += LIMIT
    return set(ids)

# ------------- Batching / embedding -------------

def batch_by_token_budget(records: List[Dict[str, Any]], max_req_tokens: int = MAX_REQUEST_TOKENS) -> List[List[Dict[str, Any]]]:
    """Split records into batches that fit within token budget. Returns list of batches."""
    batches = []
    batch = []
    used = 0

    for r in records:
        tl = token_len(r["text"])
        if tl > MAX_ITEM_TOKENS:
            # Split oversized record into smaller pieces
            for piece in split_by_tokens(r["text"], MAX_ITEM_TOKENS):
                # Create new record with split text piece
                new_id = f"{r['id']}#r{short_hash(piece)}"
                new_record = r.copy()
                new_record["id"] = new_id
                new_record["text"] = piece
                # Recursively batch the split pieces
                batches.extend(batch_by_token_budget([new_record], max_req_tokens))
            continue

        if batch and used + tl > max_req_tokens:
            # Current batch is full, start new one
            batches.append(batch)
            batch, used = [r], tl
        else:
            # Add to current batch
            batch.append(r)
            used += tl

    if batch:
        batches.append(batch)

    return batches

def embed_and_add(records: List[Dict[str, Any]], col, verbose: bool = True):
    """
    Embed and add records to ChromaDB with retry logic and progress tracking.
    Uses embed_record_with_retry for robust error handling.
    """
    if not records:
        if verbose:
            print("Nothing to embed.")
        return

    from collections import deque

    already_seen = set()
    queue = deque(records)
    total_pending = len(queue)

    if verbose:
        print(f"Total new records to process: {total_pending}")

    batch_records = []
    batch_embeddings = []
    batch_tokens = 0

    def flush_batch():
        nonlocal total_pending, batch_tokens
        if not batch_records:
            return
        ids = [r["id"] for r in batch_records]
        texts = [r["text"] for r in batch_records]
        metas = [r["metadata"] for r in batch_records]
        col.add(ids=ids, documents=texts, metadatas=metas, embeddings=batch_embeddings)
        total_pending -= len(ids)
        if verbose:
            print(f"Added {len(ids)} records. {total_pending} to go")
        batch_records.clear()
        batch_embeddings.clear()
        batch_tokens = 0

    while queue:
        record = queue.popleft()
        text = record["text"]
        tl = token_len(text)

        # Check if record is too large
        if tl > MAX_ITEM_TOKENS:
            # Split oversized record
            pieces = split_by_tokens(text, MAX_ITEM_TOKENS // 2)
            for i, piece in enumerate(pieces):
                new_id = f"{record['id']}#split{i}_{short_hash(piece)}"
                if new_id in already_seen:
                    continue
                already_seen.add(new_id)
                new_record = record.copy()
                new_record["id"] = new_id
                new_record["text"] = piece
                queue.append(new_record)
                total_pending += 1
            total_pending -= 1
            continue

        # Flush batch if adding this record would exceed token limit
        if batch_records and batch_tokens + tl > MAX_REQUEST_TOKENS:
            flush_batch()

        # Embed with retry logic
        try:
            embedding, new_records = embed_record_with_retry(
                record, embed_client, already_seen, MAX_ITEM_TOKENS, verbose=verbose
            )
        except Exception:
            flush_batch()
            raise

        # If splitting was needed, add splits to queue
        if new_records:
            for nr in reversed(new_records):
                queue.appendleft(nr)
            total_pending += len(new_records) - 1
            continue

        # Add to batch
        batch_records.append(record)
        batch_embeddings.append(embedding)
        batch_tokens += tl

    # Flush remaining batch
    flush_batch()

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
    col = db_client.get_or_create_collection(name)

    records = walk_docs(full_path)
    print(f"Found {len(records)} records in total")

    # De-dup vs existing
    existing = get_all_ids(col)
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
        cname = "ARTICLES"
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
