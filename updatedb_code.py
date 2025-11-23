import os
import sys
from dotenv import load_dotenv
from config import CHROMA_DB_DIR, EMBEDDING_MODEL, IGNORE_FILES, USE_OPENAI
from path_utils import encode_path
from tokenizer_utils import count_tokens, split_by_tokens
from embed_client import EmbedClientUni
from updatedb_helper import embed_record_with_retry, uniquify_records, short_hash, token_len
import chromadb
from collections import deque
import lizard
from pathlib import Path
from updatedb_docs import get_existing_ids, embed_and_add


from config import (
    CHROMA_DB_DIR,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    GLOBAL_RAGDATA_MAP,
    RAGType,
)

load_dotenv()

# Optimized for Qwen3-4B (256K context window)
# See TOKEN_LIMITS_GUIDE.md for rationale
if USE_OPENAI:
    MAX_ITEM_TOKENS   = 1200           # Handles larger functions/classes with context
    MAX_REQUEST_TOKENS= 6000           # Larger batches for structured code
else:
    MAX_ITEM_TOKENS   = 3000           # Handles larger functions/classes with context
    MAX_REQUEST_TOKENS= 12000          # Larger batches for structured code

CHROMA_DB_FULL_PATH = os.path.expanduser(CHROMA_DB_DIR)

CPP_EXTS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".hxx"}

embed_client = EmbedClientUni(use_openai = USE_OPENAI)

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


# --- ID helpers --------------------------------------------------------------

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
                new_id = f"{base_id}#p{idx}-{short_hash(piece)}"
                meta = dict(c["metadata"])
                meta.update({"piece_index": idx, "piece_count": len(pieces)})
                yield {"id": new_id, "text": piece, "metadata": meta}


def update_code_collection(db_client, name, full_path):
    print(f"Updating code collection {name}")
    collection = db_client.get_or_create_collection(name)

    code_chunks = walk_repo_and_chunk(full_path)
    print(f"All {len(code_chunks)} chunks from {full_path} collected.")
 
    existing = get_existing_ids(collection)
    already_seen = set(existing)
    records = list(prepare_records(code_chunks))
    uniq_records = uniquify_records(records, already_seen=already_seen)

    embed_and_add(uniq_records, collection)
    print("Done.")

def main():
    valid_names = ", ".join(
        sorted(key for key, (_, entry_type) in GLOBAL_RAGDATA_MAP.items() if entry_type == RAGType.SRC)
    )

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python updatedb_code.py <collection name>")
        print(f"  Valid names are: {valid_names}")
        cname = "BASECODE"
    else:
        cname = sys.argv[1]

    rag_entry = GLOBAL_RAGDATA_MAP.get(cname)
    if rag_entry is None:
        print(f"[ERROR] Unknown collection '{cname}'. Valid options are: {valid_names}")
        return

    doc_path, _ = rag_entry
    client = chromadb.PersistentClient(path=CHROMA_DB_FULL_PATH)
    update_code_collection(client, cname, doc_path)

if __name__ == "__main__":
    main()
