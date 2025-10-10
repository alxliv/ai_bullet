import os
from dotenv import load_dotenv
from config import DOCUMENTS_PATH, SOURCES_PATH, EXAMPLES_PATH, CHROMA_DB_DIR, EMBEDDING_MODEL
from path_utils import encode_path

from openai import OpenAI
import tiktoken
import chromadb
import hashlib
from collections import defaultdict
import lizard
from pathlib import Path

CPP_EXTS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".hxx"}

MAX_ITEM_TOKENS   = 7800           # < 8192 to leave headroom
MAX_REQUEST_TOKENS= 7800

DOCUMENTS_FULL_PATH = os.path.expanduser(DOCUMENTS_PATH)
SOURCES_FULL_PATH = os.path.expanduser(SOURCES_PATH)
EXAMPLES_FULL_PATH = os.path.expanduser(EXAMPLES_PATH)
CHROMA_DB_FULL_PATH = os.path.expanduser(CHROMA_DB_DIR)

enc = tiktoken.get_encoding("cl100k_base")

load_dotenv()
client_oa = OpenAI()

def token_len(txt: str) -> int:
    return len(enc.encode(txt))

def short_hash(text: str, length=8) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]

def split_by_tokens(txt: str, max_tokens: int) -> list[str]:
    toks = enc.encode(txt)
    return [enc.decode(toks[i:i+max_tokens]) for i in range(0, len(toks), max_tokens)]

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

def walk_repo_and_chunk(root_dir):
    all_chunks = []
    for dirpath, _, files in os.walk(root_dir):
        count = 0
        for name in files:
            if os.path.splitext(name)[1].lower() in CPP_EXTS:
                p = os.path.join(dirpath, name)
                all_chunks.extend(extract_chunks_with_lizard(p))
                count+=1
                print(f"{count}/{len(files)} files chunked")
    return all_chunks

all_chunks = walk_repo_and_chunk(SOURCES_FULL_PATH)
print(f"All {len(all_chunks)} chunks from {SOURCES_FULL_PATH} collected.")
all_chunks_examples = walk_repo_and_chunk(EXAMPLES_FULL_PATH)
print(f"All {len(all_chunks_examples)} chunks from {EXAMPLES_FULL_PATH} collected.")
all_chunks += all_chunks_examples

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
    for c in chunks:
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

def batch_by_token_budget(records):
    """Group records so total tokens per request stays under MAX_REQUEST_TOKENS."""
    batch, used = [], 0
    for r in records:
        tl = token_len(r["text"])
        # (tl should already be <= MAX_ITEM_TOKENS)
        if used + tl > MAX_REQUEST_TOKENS and batch:
            yield batch
            batch, used = [r], tl
        else:
            batch.append(r)
            used += tl
    if batch:
        yield batch

def embed_batch(texts):
    """Call OpenAI embeddings once for a batch of texts."""
    resp = client_oa.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def embed_and_add(chunks, collection):
    existing = get_existing_ids(collection)
    # split long ones
    records = list(prepare_records(chunks))
    # ensure uniqueness vs existing AND inside our current run
    records = uniquify_records(records, already_seen=set(existing))
    if not records:
        print("Nothing new to embed.")
        return

    records_left = len(records)
    print(f"Total new records to process: {records_left}")
    for batch in batch_by_token_budget(records):
        texts     = [r["text"] for r in batch]
        ids       = [r["id"]   for r in batch]
        metadatas = [r["metadata"] for r in batch]
        embs = embed_batch(texts)
        collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embs)
        records_left -= len(ids)
        print(f"Added {len(ids)} records. {records_left} to go")

client = chromadb.PersistentClient(path=CHROMA_DB_FULL_PATH)
collection = client.get_or_create_collection(name="cpp_code")

# SANITY: check that we do not have duplicate IDs
#ids = [c["id"] for c in all_chunks]
#dups = [i for i, cnt in Counter(ids).items() if cnt > 1]
#if dups:
#    print("WARNING: duplicate IDs in all_chunks:", dups[:10])

# ---- call it ----
embed_and_add(all_chunks, collection)

print("All done.")

