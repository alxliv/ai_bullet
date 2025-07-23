#!/usr/bin/env python3
"""
updatedb.py
===========
Walk a DOCS directory, extract text from PDF / DOCX / Markdown (and plain .txt),
chunk it with token-aware splits, embed with OpenAI, and store in a ChromaDB
collection (e.g. "all_documents"), avoiding duplicate IDs and oversize requests.

This mirrors the token-budget logic you used for the cpp_code collection.

Dependencies (install what you need):
    pip install chromadb openai tiktoken pypdf python-docx
    # optional fallbacks:
    # pip install textract  (for legacy .doc)  OR skip .doc files

Example:
    python updatedb.py --docs_dir ./DOCS --store ./chroma_store --collection all_documents

"""
from __future__ import annotations

import os
import re
import math
import json
import hashlib
import argparse
import tiktoken
from pypdf import PdfReader
import docx
from typing import List, Dict, Any, Iterable, Tuple, Optional
from config import DOCUMENTS_PATH, SOURCES_PATH, CHROMA_DB_DIR, EMBEDDING_MODEL

# ------------- Third-party deps -------------
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from openai import OpenAI, BadRequestError

# ------------- Config defaults -------------
EMBED_MODEL = EMBEDDING_MODEL
MAX_ITEM_TOKENS = 7800                   # leave headroom under 8192
MAX_REQUEST_TOKENS = 7800
DEFAULT_BATCH_LIMIT = 7800               # sum of tokens per request
SUPPORTED_EXTS = {".pdf", ".docx", ".md", ".markdown", ".txt"}  # add more if needed

DOCUMENTS_FULL_PATH = os.path.expanduser(DOCUMENTS_PATH)
SOURCES_FULL_PATH = os.path.expanduser(SOURCES_PATH)
CHROMA_DB_FULL_PATH = os.path.expanduser(CHROMA_DB_DIR)


_ENC = tiktoken.get_encoding("cl100k_base")

def token_len(text: str) -> int:
    if _ENC is None:
        return math.ceil(len(text) / 4)  # rough fallback
    return len(_ENC.encode(text))

def split_by_tokens(text: str, max_tokens: int) -> List[str]:
    if _ENC is None:
        # naive split by chars
        step = max_tokens * 4
        return [text[i:i+step] for i in range(0, len(text), step)]
    toks = _ENC.encode(text)
    return [_ENC.decode(toks[i:i+max_tokens]) for i in range(0, len(toks), max_tokens)]

# ------------- Hash / ID helpers -------------

def short_hash(text: str, length: int = 8) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]

# ------------- File readers -------------

def read_markdown(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path: str) -> List[Tuple[int, str]]:
    """Return list[(page_number, text)]"""
    reader = PdfReader(path)
    out = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        out.append((i + 1, txt))
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
    toks = _ENC.encode(text) if _ENC else None
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
                    "file_path": path,
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
        piece = _ENC.decode(piece_toks)
        cid = f"{path}:tok{i}-{i+len(piece_toks)}-{short_hash(piece)}"
        yield {
            "id": cid,
            "text": piece,
            "metadata": {
                "file_path": path,
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
                    "file_path": path,
                    "block_index": idx,
                    "piece_index": j,
                    "source_type": "doc",
                    "format": "markdown",
                },
            }

def chunk_pdf_pages(pages: List[Tuple[int, str]], path: str, max_tokens: int = MAX_ITEM_TOKENS) -> Iterable[Dict[str, Any]]:
    for page_no, txt in pages:
        if not txt.strip():
            continue
        if token_len(txt) <= max_tokens:
            cid = f"{path}:p{page_no}-{short_hash(txt)}"
            yield {
                "id": cid,
                "text": txt,
                "metadata": {
                    "file_path": path,
                    "page": page_no,
                    "source_type": "doc",
                    "format": "pdf",
                },
            }
        else:
            pieces = split_by_tokens(txt, max_tokens)
            for idx, piece in enumerate(pieces):
                cid = f"{path}:p{page_no}-{idx}-{short_hash(piece)}"
                yield {
                    "id": cid,
                    "text": piece,
                    "metadata": {
                        "file_path": path,
                        "page": page_no,
                        "piece_index": idx,
                        "source_type": "doc",
                        "format": "pdf",
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

from collections import defaultdict

def uniquify_records(records: List[Dict[str, Any]], already_seen: set[str]) -> List[Dict[str, Any]]:
    seen_local = defaultdict(set)  # id -> set(text_hash)
    out = []
    for r in records:
        rid = r["id"]
        txt = r["text"]
        hsh = short_hash(txt, 12)
        if rid in already_seen:
            # skip entirely
            continue
        if hsh in seen_local[rid]:
            continue
        if rid in seen_local and seen_local[rid]:
            # collision: different text under same id -> rename
            new_id = f"{rid}#dup{len(seen_local[rid])}-{hsh[:6]}"
            r = {**r, "id": new_id}
        seen_local[rid].add(hsh)
        already_seen.add(r["id"])
        out.append(r)
    return out

# ------------- Batching / embedding -------------

def batch_by_token_budget(records: List[Dict[str, Any]], max_req_tokens: int = MAX_REQUEST_TOKENS) -> Iterable[List[Dict[str, Any]]]:
    batch = []
    used = 0
    for r in records:
        tl = token_len(r["text"])
        if tl > MAX_ITEM_TOKENS:
            # should not happen (we split), but double check
            for piece in split_by_tokens(r["text"], MAX_ITEM_TOKENS):
                nr = {**r, "id": f"{r['id']}#r{short_hash(piece)}", "text": piece}
                yield from batch_by_token_budget([nr], max_req_tokens)
            continue
        if batch and used + tl > max_req_tokens:
            yield batch
            batch, used = [r], tl
        else:
            batch.append(r); used += tl
    if batch:
        yield batch

def embed_and_add(records: List[Dict[str, Any]], col, client_oa: OpenAI, verbose: bool = True):
    if not records:
        if verbose:
            print("Nothing to embed.")
        return
    for batch in batch_by_token_budget(records):
        texts     = [r["text"] for r in batch]
        ids       = [r["id"] for r in batch]
        metadatas = [r["metadata"] for r in batch]
        try:
            resp = client_oa.embeddings.create(model=EMBED_MODEL, input=texts)
            embs = [d.embedding for d in resp.data]
        except BadRequestError as e:
            msg = str(e)
            if "maximum context length" in msg:
                print("Batch too big, resplitting...")
                # split items again aggressively
                smaller = []
                for r in batch:
                    for piece in split_by_tokens(r["text"], MAX_ITEM_TOKENS // 2):
                        smaller.append({**r, "id": f"{r['id']}#rs{short_hash(piece)}", "text": piece})
                embed_and_add(smaller, col, client_oa, verbose)
                continue
            else:
                raise
        col.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embs)
        if verbose:
            print(f"Added {len(ids)} records")

# ------------- Main pipeline -------------

def build_records_for_file(path: str) -> List[Dict[str, Any]]:
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
                all_records.extend(recs)
    return all_records


def main():

    client = chromadb.PersistentClient(path=CHROMA_DB_FULL_PATH)
    col = client.get_or_create_collection(name="bullet_docs")

    records = walk_docs(DOCUMENTS_FULL_PATH)
    print(f"Found {len(records)} candidate chunks")

    # De-dup vs existing
    existing = get_all_ids(col)
    uniq_records = uniquify_records(records, already_seen=set(existing))
    print(f"After dedup: {len(uniq_records)} new chunks to embed")

    # Embed & store
    oa = OpenAI()
    embed_and_add(uniq_records, col, oa, verbose=True)
    print("Done.")


if __name__ == "__main__":
    main()
