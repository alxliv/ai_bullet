import os
import time
from dotenv import load_dotenv
from config import DOCUMENTS_PATH, SOURCES_PATH, EXAMPLES_PATH, CHROMA_DB_DIR, EMBEDDING_MODEL, IGNORE_FILES
from path_utils import encode_path
from tokenizer_utils import count_tokens, split_by_tokens as split_by_tokens_util

import chromadb
import hashlib
import httpx
from collections import defaultdict, deque
import lizard
from pathlib import Path

CPP_EXTS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".hxx"}

# Optimized for Qwen3-4B (256K context window)
# See TOKEN_LIMITS_GUIDE.md for rationale
MAX_ITEM_TOKENS   = 3000           # Handles larger functions/classes with context
MAX_REQUEST_TOKENS= 12000          # Larger batches for structured code

MAX_EMBED_RETRIES = 3
RETRY_INITIAL_DELAY = 0.2
RETRY_MAX_DELAY = 5.0
MIN_RETRY_TOKENS = 512

DOCUMENTS_FULL_PATH = os.path.expanduser(DOCUMENTS_PATH)
SOURCES_FULL_PATH = os.path.expanduser(SOURCES_PATH)
EXAMPLES_FULL_PATH = os.path.expanduser(EXAMPLES_PATH)
CHROMA_DB_FULL_PATH = os.path.expanduser(CHROMA_DB_DIR)

load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", EMBEDDING_MODEL)

from retriever import OllamaEmbeddingClient  # reuse embedding helper

embed_client = OllamaEmbeddingClient(OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL)

def token_len(txt: str) -> int:
    """Count tokens using Qwen3 tokenizer."""
    try:
        return count_tokens(txt)
    except RuntimeError:
        # Fallback to character-based estimation if tokenizer not available
        return len(txt) // 4

def short_hash(text: str, length=8) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]

def split_by_tokens(txt: str, max_tokens: int) -> list[str]:
    """Split text by tokens using Qwen3 tokenizer."""
    try:
        return split_by_tokens_util(txt, max_tokens)
    except RuntimeError:
        # Fallback to character-based splitting if tokenizer not available
        step = max_tokens * 4
        return [txt[i:i+step] for i in range(0, len(txt), step)]

def grab_leading_comment(lines, start_idx, max_gap=2):
    """
    Walk upward from start_idx-1 to collect a contiguous block of comments
    separated from code by <= max_gap blank lines.
    """
    i = start_idx - 1
    collected = []
    blanks = 0
    while i >= 0:
        line = lines[i].rstrip()
        if line.strip().startswith("//") or "/*" in line:
            collected.append(lines[i])
            blanks = 0
        elif line.strip() == "":
            blanks += 1
            if blanks > max_gap:
                break
            collected.append(lines[i])
        else:
            break
        i -= 1
    collected.reverse()
    return collected

def slice_lines(lines, start, end):
    return "\n".join(lines[start-1:end])

def build_leftover_chunks(lines, used_spans, file_path):
    """Anything not covered by functions becomes its own chunk (contiguous blocks)."""
    n = len(lines)
    covered = [False]*(n+1)
    for s,e in used_spans:
        for i in range(s, e+1):
            if 0 < i <= n:
                covered[i] = True
    chunks = []
    i = 1
    while i <= n:
        if covered[i]:
            i += 1
            continue
        j = i
        while j <= n and not covered[j]:
            j += 1
        text = slice_lines(lines, i, j-1)
        if text.strip():
            cid = f"{Path(file_path).name}:{i}-{j-1}-{short_hash(text)}"
            chunks.append({
                "id": cid,
                "text": text,
                "metadata": {
                    "file_path": encode_path(file_path),
                    "start_line": i,
                    "end_line": j-1,
                    "node_type": "leftover_block"
                }
            })
        i = j
    return chunks

def extract_chunks_with_lizard(path, include_comments=True):
    """Return list[{id,text,metadata}] for a single file."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    src = "".join(lines)

    try:
        res = lizard.analyze_file(path)
    except Exception as e:
        print(f"[Lizard] Failed to analyze {path}: {e}")
        return []

    chunks = []
    spans = []
    for fn in res.function_list:
        sl, el = fn.start_line, fn.end_line
        body_lines = lines[sl-1:el]
        comment_lines = grab_leading_comment(lines, sl) if include_comments else []
        text = "".join(comment_lines + body_lines)
        if 'Copyright' in text:
            continue

        cid = f"{Path(path).name}:{sl}-{el}-{short_hash(text)}"
        chunks.append({
            "id": cid,
            "text": text,
            "metadata": {
                "file_path": encode_path(path),
                "start_line": sl,
                "end_line": el,
                "name": fn.name,
                "long_name": fn.long_name,
                "cyclomatic_complexity": fn.cyclomatic_complexity,
                "nloc": fn.nloc,
                "parameter_count": len(fn.parameters),
                "parameters": ",".join(fn.parameters) if fn.parameters else "",
                "node_type": "function"
            }
        })
        spans.append((sl, el))

    # Add non-function regions (classes, typedefs, etc.)
    leftovers = build_leftover_chunks(lines, spans, path)
    chunks.extend(leftovers)

    return chunks

def should_ignore_file(filename: str, ignore_patterns) -> bool:
    """
    Check if a file should be ignored based on ignore rules.

    Args:
        filename: Name of the file to check
        ignore_patterns: Set or iterable of ignore patterns (exact names or wildcards)

    Returns:
        True if file should be ignored, False otherwise

    Examples:
        >>> should_ignore_file("test.cpp", {"test.cpp"})
        True
        >>> should_ignore_file("landscapeData.h", {"landscapeData.h"})
        True
        >>> should_ignore_file("test_main.cpp", {"test_*.cpp"})
        True
    """
    import fnmatch

    if not ignore_patterns:
        return False

    # Check exact match first (faster)
    if filename in ignore_patterns:
        return True

    # Check wildcard patterns
    for pattern in ignore_patterns:
        if '*' in pattern or '?' in pattern:
            if fnmatch.fnmatch(filename, pattern):
                return True

    return False


def walk_repo_and_chunk(root_dir, ignore_files=IGNORE_FILES):
    """
    Walk directory tree and extract code chunks from C/C++ files.

    Args:
        root_dir: Root directory to search
        ignore_files: Set of filenames/patterns to skip. Supports:
                     - Exact names: {"landscapeData.h", "test.cpp"}
                     - Wildcards: {"test_*.cpp", "*_generated.h"}

    Returns:
        List of chunk dictionaries

    Examples:
        >>> chunks = walk_repo_and_chunk("./src", {"test_*.cpp", "generated.h"})
    """
    all_chunks = []
    num_folders = 0
    total_files_processed = 0
    total_files_skipped = 0

    for dirpath, _, files in os.walk(root_dir):
        print(f"#{num_folders}. walk_repo_and_chunk() path={dirpath}")
        num_folders += 1
        count = 0

        for name in files:
            # Check if file should be ignored
            if should_ignore_file(name, ignore_files):
                print(f"\tSkipping ignored file: {name}")
                total_files_skipped += 1
                continue

            # Check if it's a C/C++ file
            if os.path.splitext(name)[1].lower() in CPP_EXTS:
                p = os.path.join(dirpath, name)
                try:
                    chunks = extract_chunks_with_lizard(p)
                    all_chunks.extend(chunks)
                    count += 1
                    total_files_processed += 1
                    print(f"\tFile: {name}, {count} files chunked in this folder ({len(chunks)} chunks)")
                except Exception as e:
                    print(f"\t[ERROR] Failed to process {name}: {e}")
                    total_files_skipped += 1

    print(f"\n=== Summary ===")
    print(f"Folders processed: {num_folders}")
    print(f"Files processed: {total_files_processed}")
    print(f"Files skipped/ignored: {total_files_skipped}")
    print(f"Total chunks extracted: {len(all_chunks)}")

    return all_chunks

def get_existing_ids(collection):
    """Return a set of all IDs already stored."""
    total = collection.count()
    if total == 0:
        return set()
    res = collection.get(include=[], limit=total, offset=0)
    return set(res["ids"])

# --- ID helpers --------------------------------------------------------------

def stable_suffix(text: str, length=8) -> str:
    """Short hash so same text → same suffix, different text → new suffix."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]

def uniquify_records(records, already_seen: set):
    """
    Ensure every record has a unique id.
    - If id is free: keep it.
    - If id collides but text is identical: skip (we already have it).
    - If id collides and text differs: append '#dup{n}-{hash}'.
    """
    seen_text_by_id = {}
    out = []
    dup_counters = defaultdict(int)

    for r in records:
        rid, txt = r["id"], r["text"]

        if rid in already_seen:
            # we already stored it in Chroma → skip
            continue

        if rid in seen_text_by_id:
            if seen_text_by_id[rid] == txt:
                # exact duplicate inside this batch → skip
                continue
            # different text but same id → rename
            dup_counters[rid] += 1
            new_id = f"{rid}#dup{dup_counters[rid]}-{stable_suffix(txt)}"
            r = {**r, "id": new_id}
            rid = new_id

        seen_text_by_id[rid] = txt
        already_seen.add(rid)
        out.append(r)

    return out

def prepare_records(chunks):
    """Yield records, splitting any text that exceeds MAX_ITEM_TOKENS."""
    nrec = 0
    for c in chunks:
        nrec += 1
        print(f"prepare_record #{nrec} out of {len(chunks)}")
        base_id = c["id"]
        txt = c["text"]
        if token_len(txt) <= MAX_ITEM_TOKENS:
            yield c  # keep original id
        else:
            pieces = split_by_tokens(txt, MAX_ITEM_TOKENS)
            for idx, piece in enumerate(pieces):
                new_id = f"{base_id}#p{idx}-{stable_suffix(piece)}"
                meta = dict(c["metadata"])
                meta.update({"piece_index": idx, "piece_count": len(pieces)})
                yield {"id": new_id, "text": piece, "metadata": meta}

def make_retry_splits(record, already_seen: set, target_tokens: int):
    """Split a record into smaller pieces to satisfy embedding limits."""
    pieces = split_by_tokens(record["text"], target_tokens)
    if len(pieces) <= 1:
        return []
    level = record.get("_split_level", 0) + 1
    base_meta = dict(record["metadata"])
    base_meta.setdefault("parent_id", record["metadata"].get("parent_id", record["id"]))
    base_meta["piece_count"] = len(pieces)
    splits = []
    for idx, piece in enumerate(pieces):
        meta = dict(base_meta)
        meta.update({
            "piece_index": idx,
            "retry_split": True,
            "split_level": level,
        })
        new_id = f"{record['id']}#s{level}-{idx}-{stable_suffix(piece + f'|lvl{level}|{idx}')}"  # stable but unique per level
        # ensure uniqueness in this run
        while new_id in already_seen:
            new_id = f"{record['id']}#s{level}-{idx}-{stable_suffix(piece + f'|lvl{level}|{idx}|{len(splits)}')}"
        already_seen.add(new_id)
        splits.append({
            "id": new_id,
            "text": piece,
            "metadata": meta,
            "_split_level": level,
        })
    return splits

def embed_record_with_retry(record, already_seen: set):
    """
    Try to embed a record, retrying transient failures and requesting
    a split when the server keeps rejecting the payload.
    Returns (embedding, None) on success or (None, new_records) when the caller
    should process the returned splits instead.
    """
    text = record["text"]
    delay = RETRY_INITIAL_DELAY
    last_exc = None
    for attempt in range(1, MAX_EMBED_RETRIES + 1):
        try:
            embeds = embed_client.embed(text)
            return embeds, None
        except httpx.HTTPStatusError as exc:  # type: ignore[attr-defined]
            last_exc = exc
            status = exc.response.status_code if exc.response is not None else None
            detail = ""
            if exc.response is not None:
                try:
                    detail = exc.response.text.strip()
                except Exception:
                    detail = ""
 #           token_count = token_len(text)
 #           print(f"[WARN] Embed HTTP {status} for {record['id']} (attempt {attempt}/{MAX_EMBED_RETRIES}, tokens={token_count}).")
 #           if detail:
 #              print(f"       Detail: {detail[:200]}")
            if status in {429, 500, 502, 503, 504} or (status is not None and status >= 500):
                time.sleep(min(delay, RETRY_MAX_DELAY))
                delay = min(delay * 2, RETRY_MAX_DELAY)
                continue
            raise
        except httpx.RequestError as exc:  # network issues, timeouts, etc.
            last_exc = exc
            print(f"[WARN] Transport error during embed for {record['id']} on attempt {attempt}/{MAX_EMBED_RETRIES}: {exc}")
            time.sleep(min(delay, RETRY_MAX_DELAY))
            delay = min(delay * 2, RETRY_MAX_DELAY)
    # Retries exhausted -> attempt to split if possible
    tokens = token_len(text)
    if tokens > MIN_RETRY_TOKENS:
        target = max(MIN_RETRY_TOKENS, min(MAX_ITEM_TOKENS, max(tokens // 2, MIN_RETRY_TOKENS)))
        splits = make_retry_splits(record, already_seen, target)
        if splits:
            print(f"[INFO] Splitting record {record['id']} into {len(splits)} pieces after repeated failures.")
            return None, splits
    raise last_exc if last_exc is not None else RuntimeError(f"Embedding failed for {record['id']} with no additional detail.")

def embed_and_add(chunks, collection):
    existing = get_existing_ids(collection)
    already_seen = set(existing)
    records = list(prepare_records(chunks))
    records = uniquify_records(records, already_seen=already_seen)
    if not records:
        print("Nothing new to embed.")
        return

    queue = deque(records)
    total_pending = len(queue)
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
        collection.add(ids=ids, documents=texts, metadatas=metas, embeddings=batch_embeddings)
        total_pending -= len(ids)
        print(f"Added {len(ids)} records. {total_pending} to go")
        batch_records.clear()
        batch_embeddings.clear()
        batch_tokens = 0

    while queue:
        record = queue.popleft()
        text = record["text"]
        tl = token_len(text)
        if tl > MAX_ITEM_TOKENS:
            splits = make_retry_splits(record, already_seen, max(MIN_RETRY_TOKENS, MAX_ITEM_TOKENS // 2))
            if splits:
                queue.extendleft(reversed(splits))
                total_pending += len(splits) - 1
                continue
            else:
                print(f"[WARN] Record {record['id']} exceeds MAX_ITEM_TOKENS ({tl}) but cannot be split further.")
        if batch_records and batch_tokens + tl > MAX_REQUEST_TOKENS:
            flush_batch()
        try:
            embedding, new_records = embed_record_with_retry(record, already_seen)
        except Exception:
            flush_batch()
            raise
        if new_records:
            queue.extendleft(reversed(new_records))
            total_pending += len(new_records) - 1
            continue
        batch_records.append(record)
        batch_embeddings.append(embedding)
        batch_tokens += tl

    flush_batch()

if __name__ == "__main__":
    client = chromadb.PersistentClient(path=CHROMA_DB_FULL_PATH)
    collection = client.get_or_create_collection(name="cpp_code")

    code_chunks = walk_repo_and_chunk(SOURCES_FULL_PATH)
    print(f"All {len(code_chunks)} chunks from {SOURCES_FULL_PATH} collected.")
    example_chunks = walk_repo_and_chunk(EXAMPLES_FULL_PATH)
    print(f"All {len(example_chunks)} chunks from {EXAMPLES_FULL_PATH} collected.")
    all_chunks = code_chunks + example_chunks

    # SANITY: check that we do not have duplicate IDs
    #ids = [c["id"] for c in all_chunks]
    #dups = [i for i, cnt in Counter(ids).items() if cnt > 1]
    #if dups:
    #    print("WARNING: duplicate IDs in all_chunks:", dups[:10])

    # ---- call it ----
    embed_and_add(all_chunks, collection)

    print("All done.")

    embed_client.close()

