"""
updatedb_helper.py
==================

Shared utilities for updatedb_docs.py and updatedb_code.py.
Provides robust embedding with retry logic, exponential backoff, and automatic splitting.
"""

import time
import httpx
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import defaultdict
import hashlib
from config import USE_OPENAI
from embed_client import EmbedClientUni

# Retry configuration
MAX_EMBED_RETRIES = 3
RETRY_INITIAL_DELAY = 0.2  # seconds
RETRY_MAX_DELAY = 5.0      # seconds
MIN_RETRY_CHARS = 4000      # Minimum chars before we try splitting

# Character-based limits for embedding models
# mxbai-embed-large: 512 tokens - extremely restrictive (~1 char/token worst case for code)
# nomic-embed-text: 8192 tokens context - RECOMMENDED for RAG
# text-embedding-3-small: 8191 tokens context
# Using conservative limits to handle worst-case tokenization
if USE_OPENAI:
    MAX_EMBED_CHARS = 24000   # Safe limit for OpenAI embedding models
else:
    MAX_EMBED_CHARS = 4000

embed_client = EmbedClientUni(use_openai = USE_OPENAI)


def short_hash(text: str, length: int = 8) -> str:
    """Generate short hash for ID generation."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]


def uniquify_records(records: List[Dict[str, Any]], already_seen: Set[str]) -> List[Dict[str, Any]]:
    """
    Ensure every record has a unique ID.

    - If ID is free: keep it
    - If ID collides but text is identical: skip (already have it)
    - If ID collides and text differs: rename with '#dup{n}-{hash}'

    Args:
        records: List of records with 'id' and 'text' keys
        already_seen: Set of IDs already in ChromaDB (will be updated)

    Returns:
        List of unique records
    """
    seen_text_by_id = {}
    out = []
    dup_counters = defaultdict(int)

    for r in records:
        rid, txt = r["id"], r["text"]

        if rid in already_seen:
            # Already stored in ChromaDB - skip
            continue

        if rid in seen_text_by_id:
            if seen_text_by_id[rid] == txt:
                # Exact duplicate in this batch - skip
                continue
            # Different text but same ID - rename
            dup_counters[rid] += 1
            new_id = f"{rid}#dup{dup_counters[rid]}-{short_hash(txt)}"
            r = r.copy()
            r["id"] = new_id
            rid = new_id

        seen_text_by_id[rid] = txt
        already_seen.add(rid)
        out.append(r)

    return out


def split_by_chars(text: str, max_chars: int) -> List[str]:
    """
    Split text into chunks by character count.
    Tries to split at newlines for cleaner breaks.

    Args:
        text: Input text string
        max_chars: Maximum characters per chunk

    Returns:
        List of text chunks
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:])
            break
        # Try to find a newline to break at
        newline_pos = text.rfind('\n', start, end)
        if newline_pos > start + max_chars // 2:
            end = newline_pos + 1
        chunks.append(text[start:end])
        start = end
    return chunks


def make_retry_splits(
    record: Dict[str, Any],
    already_seen: Set[str],
    target_chars: int
) -> List[Dict[str, Any]]:
    """
    Split a record into smaller pieces for retry purposes.

    Args:
        record: The record to split
        already_seen: Set of IDs already used (to avoid duplicates)
        target_chars: Target character count per split

    Returns:
        List of split records
    """
    text = record["text"]
    pieces = split_by_chars(text, target_chars)

    splits = []
    for i, piece in enumerate(pieces):
        new_id = f"{record['id']}#split{i}_{short_hash(piece)}"
        if new_id in already_seen:
            continue
        already_seen.add(new_id)

        new_record = record.copy()
        new_record["id"] = new_id
        new_record["text"] = piece
        splits.append(new_record)

    return splits


def embed_record_with_retry(
    record: Dict[str, Any],
    embed_client,
    already_seen: Set[str],
    max_chars: int,
    verbose: bool = True
) -> Tuple[Optional[List[float]], Optional[List[Dict[str, Any]]]]:
    """
    Try to embed a record with retry logic for transient failures.

    Features:
    - Exponential backoff for rate limits and server errors
    - Network error handling
    - Automatic splitting after retries exhausted

    Args:
        record: Record to embed (must have 'id' and 'text' keys)
        embed_client: Client with embed() method
        already_seen: Set of IDs already processed
        max_chars: Maximum characters per item before splitting
        verbose: Whether to print warnings

    Returns:
        (embedding, None) on success
        (None, new_records) when caller should process the splits instead

    Raises:
        Exception if all retries fail and splitting is not possible
    """
    text = record["text"]
    delay = RETRY_INITIAL_DELAY
    last_exc = None

    for attempt in range(1, MAX_EMBED_RETRIES + 1):
        try:
            embedding = embed_client.embed(text)
            return embedding, None

        except httpx.HTTPStatusError as exc:
            last_exc = exc
            status = exc.response.status_code if exc.response is not None else None

            # Get error details for logging
            detail = ""
            if exc.response is not None:
                try:
                    detail = exc.response.text.strip()
                except Exception:
                    detail = ""

            if verbose and attempt == 1:
                print(f"[WARN] Embed HTTP {status} for {record['id']} (attempt {attempt}/{MAX_EMBED_RETRIES}, chars={len(text)})")
                if detail and len(detail) < 200:
                    print(f"       Detail: {detail}")

            # Retry on transient errors
            if status in {429, 500, 502, 503, 504} or (status is not None and status >= 500):
                if verbose and attempt > 1:
                    print(f"[RETRY] Attempt {attempt}/{MAX_EMBED_RETRIES} for {record['id']}")
                time.sleep(min(delay, RETRY_MAX_DELAY))
                delay = min(delay * 2, RETRY_MAX_DELAY)
                continue

            # Non-retryable error (e.g., 400, 413)
            raise

        except httpx.RequestError as exc:
            # Network issues, timeouts, connection errors
            last_exc = exc
            if verbose:
                print(f"[WARN] Network error for {record['id']} on attempt {attempt}/{MAX_EMBED_RETRIES}: {exc}")
            time.sleep(min(delay, RETRY_MAX_DELAY))
            delay = min(delay * 2, RETRY_MAX_DELAY)

    # All retries exhausted - try splitting if text is large enough
    text_len = len(text)
    if text_len > MIN_RETRY_CHARS:
        target = max(MIN_RETRY_CHARS, min(max_chars, text_len // 2))
        splits = make_retry_splits(record, already_seen, target)

        if splits:
            if verbose:
                print(f"[INFO] Splitting record {record['id']} into {len(splits)} pieces after {MAX_EMBED_RETRIES} failed attempts")
            return None, splits

    # Cannot split or splitting didn't help
    error_msg = f"Embedding failed for {record['id']} after {MAX_EMBED_RETRIES} attempts"
    if last_exc:
        raise last_exc
    else:
        raise RuntimeError(error_msg)

def get_existing_ids(col) -> set[str]:
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

def embed_and_add(records: List[Dict[str, Any]], col, verbose: bool = True):
    """
    Embed and add records to ChromaDB with retry logic and progress tracking.
    Uses character-based limits for embedding model compatibility.
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
    total_records = 0

    def flush_batch():
        nonlocal total_pending
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

    while queue:
        record = queue.popleft()
        if (record['id'] == 'b3Chunk.h:68-71-90e40ca4'):
            print("!")

        text = record["text"]
        text_len = len(text)

        # Check if record is too large for embedding model
        if text_len > MAX_EMBED_CHARS:
            # Split oversized record
            pieces = split_by_chars(text, MAX_EMBED_CHARS // 2)
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

        # Embed with retry logic
        try:
            embedding, new_records = embed_record_with_retry(
                record, embed_client, already_seen, MAX_EMBED_CHARS, verbose=verbose
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
        total_records +=1
        # Add to batch
        batch_records.append(record)
        batch_embeddings.append(embedding)
        if len(batch_records) >=1000:
            if verbose:
                print(f"Embedded {total_records} records.")
            flush_batch()

    # Flush remaining batch
    flush_batch()
