import os, tiktoken
from tree_sitter import Language, Parser
import tree_sitter_cpp as tscpp
from dotenv import load_dotenv
from config import DOCUMENTS_PATH, SOURCES_PATH, CHROMA_DB_DIR, EMBEDDING_MODEL

from openai import OpenAI, BadRequestError
import tiktoken
import chromadb
import hashlib
from collections import defaultdict, Counter

MAX_ITEM_TOKENS   = 7800           # < 8192 to leave headroom
MAX_REQUEST_TOKENS= 7800

DOCUMENTS_FULL_PATH = os.path.expanduser(DOCUMENTS_PATH)
SOURCES_FULL_PATH = os.path.expanduser(SOURCES_PATH)
CHROMA_DB_FULL_PATH = os.path.expanduser(CHROMA_DB_DIR)

enc = tiktoken.get_encoding("cl100k_base")

load_dotenv()
client_oa = OpenAI()

def token_len(txt: str) -> int:
    return len(enc.encode(txt))

def split_by_tokens(txt: str, max_tokens: int) -> list[str]:
    toks = enc.encode(txt)
    return [enc.decode(toks[i:i+max_tokens]) for i in range(0, len(toks), max_tokens)]

# 1) Load C++ grammar
CPP = Language(tscpp.language())
parser = Parser(CPP)

def debug_node_types(node, depth=0):
    print("  " * depth + f"{node.type}")
    for child in node.children:
        debug_node_types(child, depth + 1)


def extract_chunks_from_file(path, max_tokens=800, overlap=100):
    source = open(path, "rb").read()
    tree   = parser.parse(source)
    root   = tree.root_node
#    debug_node_types(root)

    chunks = []
#    interesting_types = ( "comment", "function_definition", "class_specifier", "struct_specifier",
#        "namespace_definition", "enum_specifier", "union_specifier",
#        "template_declaration", "constructor_definition", "destructor_definition")
    interesting_types = ["constructor_definition"]

    # Recursively walk to find top-level function/class nodes
    def walk(node):
        if node.type in interesting_types:
            start, end = node.start_point[0], node.end_point[0]
            text = source[node.start_byte:node.end_byte].decode()
            if 'Copyright' in text:
                return
            tok_count = len(enc.encode(text))
            # if too big, fall back to sliding window
            if tok_count <= max_tokens:
                chunks.append((start+1,end+1,text,node))
            else:
                # sliding window on lines
                lines = text.splitlines()
                for i in range(0, len(lines), max_tokens - overlap):
                    window = "\n".join(lines[i:i+max_tokens])
                    chunks.append((start+i+1, start+i+1+len(lines[i:i+max_tokens]), window, node))
        else:
            for c in node.children:
                walk(c)

    walk(root)
    return chunks


def simple_hash_text(text: str) -> str:
    h = hashlib.new('md5')
    h.update(text.encode('utf-8'))
    return h.hexdigest()[:8] # return first 8 chars of MD5

# Example usage:
all_chunks = []
for root_dir, _, files in os.walk(SOURCES_FULL_PATH):
    for fn in files:
        if fn.endswith((".cpp",".h")):
            path = os.path.join(root_dir, fn)
            for start_line,end_line,text,node in extract_chunks_from_file(path):
                all_chunks.append({
                    "id": f"{fn}:{start_line}-{end_line}:{simple_hash_text(text)}",
                    "text": text,
                    "metadata": {
                       "file_path": path,
                       "start_line": start_line,
                       "end_line": end_line,
                       "node_type": node.type,
                    }
                })


print(f"All {len(all_chunks)} chunks collected.")

import chromadb
from openai import OpenAI

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

