"""
retriever.py
--------------

Query -> retrieve from multiple Chroma collections (e.g. "cpp_code" and "bullet_docs")
-> fuse results -> build a prompt/context block for a local chat model.

Features
=========
- Single embedding of the query (local embedding model via Ollama).
- Retrieve from any number of Chroma collections.
- Reciprocal Rank Fusion (RRF) to merge ranked lists.
- Optional Maximal Marginal Relevance (MMR) diversification step.
- Token-budget aware context builder (uses tiktoken if available, else falls back to char budget).
- Clean dataclasses for results and configuration.

Usage
=====

    from retriever import Retriever, RetrieverConfig
    import chromadb

    client = chromadb.PersistentClient(path="./chroma_store")
    code_col = client.get_collection("cpp_code")
    doc_col  = client.get_collection("bullet_docs")

    r = Retriever(
        collections={"code": code_col, "docs": doc_col},
        config=RetrieverConfig()
    )

    hits = r.retrieve("How is the constraint solver implemented?", k_per_collection=5)
    ctx, sources = r.build_context(hits)
    messages = r.build_messages(query="How is the constraint solver implemented?", context=ctx)

    # send `messages` to the chat model

"""
from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, cast

import httpx
from chromadb_shim import chromadb
from chromadb.api.models.Collection import Collection as ChromaCollection
from dotenv import load_dotenv

from config import CHROMA_DB_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL, LLM_DEFAULT_MODEL, USE_OPENAI, CHROMA_DB_FULL_PATH

if USE_OPENAI:
    from openai import OpenAI
    from openai.types.chat import ChatCompletionMessageParam
else:
    OpenAI = None  # type: ignore[assignment]
    ChatCompletionMessageParam = Dict[str, Any]  # type: ignore[misc]

from tokenizer_utils import count_tokens as token_len_func, truncate as truncate_text
from embed_client import EmbedClientUni

# ----------------------------
# Data structures
# ----------------------------

class RetrieverConfig:
    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        distance_metric: str = "cosine",
        rrf_b: int = 60,
        mmr_lambda: float = 0.5,
        use_mmr: bool = True,
        max_context_tokens: int = 16000,  # Optimized for Qwen3-4B (256K context)
        max_snippets: int = 12,
        code_lang_key: str = "node_type",
        code_lang_values: Sequence[str] = ("function", "leftover_block"),
        default_code_lang: str = "cpp",
        default_doc_lang: str = "markdown",
        show_source_header: bool = True,
        # default to empty dict (override per collection in retrieve with .get(name, 5))
        k_per_collection: Dict[str, int] = {},
        system_template: str = (
            "You are a precise assistant for C/C++ code and project documentation. "
            "Use only the given CONTEXT to answer. If unsure, say you don't know. "
            "Format your responses in markdown. To render mathematical expressions and formulas, use LaTeX math notation."
            "Use $ ... $ for inline math or $$ ... $$ for block math."
            "Cite sources by file path, page number and line numbers when available.\n\n"
            "IMPORTANT - Citation Format Rules:\n"
            "- ALWAYS cite sources using markdown link syntax: [filename](path) : line-range\n"
            "- CORRECT format: Source: [DeformableMultibody.cpp](/examples/DeformableDemo/DeformableMultibody.cpp) : 39-42\n"
        ),
        system_template_full: str = (
            "You are a precise assistant for C/C++ code and project documentation. "
            "Use all of your knowledge to answer."
            "Format your responses in markdown. To render mathematical expressions and formulas, use LaTeX math notation."
            "Use $ ... $ for inline math or $$ ... $$ for block math."
            "Cite sources by file path, page number and line numbers when available.\n\n"
            "IMPORTANT - Citation Format Rules:\n"
            "- ALWAYS cite sources using markdown link syntax: [filename](path) : line-range\n"
            "- CORRECT format: Source: [DeformableMultibody.cpp](/examples/DeformableDemo/DeformableMultibody.cpp) : 39-42\n"

        ),
        user_template: str = (
            "QUESTION:\n{query}\n\nCONTEXT:\n{context}\n\n"
            "Please answer using markdown, show code in fenced blocks.\n\n"
            "IMPORTANT - Citation Format Rules:\n"
            "- ALWAYS cite sources using markdown link syntax: [filename](path) : line-range\n"
            "- CORRECT format: Source: [DeformableMultibody.cpp](/examples/DeformableDemo/DeformableMultibody.cpp) : 39-42\n"
            "- INCORRECT format: Source: `/examples/file.cpp` : 39-42 (DO NOT use backticks)\n"
            "- INCORRECT format: Source: /examples/file.cpp : 39-42 (DO NOT use plain text paths)\n"
            "- File paths must be clickable markdown links enclosed in [text](path) syntax.\n"
            "- Never give references (URL) to any source outside - that is those who start with http: or https:\n"
        ),
        use_llm_rerank: bool = False,
        llm_rerank_model: str = "gpt-3.5-turbo",
        max_candidates_for_rerank: int = 30,
        llm_rerank_max_chunk_tokens: int = 512,
    ):
        self.embedding_model = embedding_model
        self.distance_metric = distance_metric
        self.rrf_b = rrf_b
        self.mmr_lambda = mmr_lambda
        self.use_mmr = use_mmr
        self.max_context_tokens = max_context_tokens
        self.max_snippets = max_snippets
        self.code_lang_key = code_lang_key
        self.code_lang_values = code_lang_values
        self.default_code_lang = default_code_lang
        self.default_doc_lang = default_doc_lang
        self.show_source_header = show_source_header
        # copy to avoid accidental sharing if multiple instances are ever created
        self.k_per_collection = dict(k_per_collection)
        self.system_template = system_template
        self.system_template_full = system_template_full
        self.user_template = user_template
        self.use_llm_rerank = use_llm_rerank
        self.llm_rerank_model = llm_rerank_model
        self.max_candidates_for_rerank = max_candidates_for_rerank
        self.llm_rerank_max_chunk_tokens = llm_rerank_max_chunk_tokens

# Keep Hit as a dataclass
@dataclass
class Hit:
    id: str
    score: float  # higher = better
    document: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    collection_name: str = ""
    embedding: Optional[List[float]] = None

    def source_label(self) -> str:
        fp = self.metadata.get("file_path") or self.metadata.get("source") or self.id

        # Convert portable path format to web URL relative to static mount points
        # {DOCS}/file.pdf -> /docs/file.pdf
        # {SRC}/path/code.cpp -> /src/path/code.cpp
        # {EXAMPLES}/demo/main.cpp -> /examples/demo/main.cpp
        # Also support legacy $VAR$ format

        if isinstance(fp, str):
            if fp.startswith('{DOCS}/'):
                fp = '/docs/' + fp[7:]  # Remove {DOCS}/
            elif fp.startswith('{SRC}/'):
                fp = '/src/' + fp[6:]   # Remove {SRC}/
            elif fp.startswith('{EXAMPLES}/'):
                fp = '/examples/' + fp[11:]  # Remove {EXAMPLES}/
            elif fp.startswith('$DOCS$/'):
                fp = '/docs/' + fp[7:]  # Remove $DOCS$/
            elif fp.startswith('$SRC$/'):
                fp = '/src/' + fp[6:]   # Remove $SRC$/
            elif fp.startswith('$EXAMPLES$/'):
                fp = '/examples/' + fp[11:]  # Remove $EXAMPLES$/

        page = self.metadata.get("page_number")
        start = self.metadata.get("start_line")
        end = self.metadata.get("end_line")
        # collect non‐empty location parts
        parts: List[str] = []
        if page:
            parts.append(f"p {page}")
        if start and end and (start!=end):
            parts.append(f"{start}-{end}")
        else:
            if start:
                parts.append(str(start))
            elif end:
                parts.append(str(end))
        # join everything or just return the file path if no parts
        if parts:
            return f"{fp} : {', '.join(parts)}"
        return str(fp)

# ----------------------------
# Utility functions
# ----------------------------

def _token_len(text: str, model: str = "cl100k_base") -> int:
    """Count tokens using Qwen3 tokenizer (model parameter ignored for compatibility)."""
    return token_len_func(text)


def _cosine_to_similarity(distance: float) -> float:
    """
    Convert distance to similarity score (0-1 range).

    Handles both cosine distance (0-2 range) and L2 distance (unbounded).
    For L2 distances, uses exponential decay to map to 0-1 range.
    """
    if distance < 2.0:
        # Assume cosine distance (typical range 0-1 for normalized vectors)
        return 1.0 - distance
    else:
        # L2 distance - use exponential decay
        # Adjust scale factor based on your embedding dimensions
        # For 768-dim vectors, scale=50 works well
        # For 1536-dim vectors, try scale=70
        scale = 50.0
        return max(0.0, 1.0 / (1.0 + (distance / scale)))


def reciprocal_rank_fusion(list_of_lists: Sequence[Sequence[Hit]], b: int = 60) -> List[Hit]:
    """Fuse multiple ranked lists of Hit by summing each item's individual scores.
    Returns a single list of Hit with fused scores.
    """
    fused: Dict[str, Tuple[float, Hit]] = {}
    for lst in list_of_lists:
        for h in lst:
            fused[h.id] = (h.score, h)

    # overwrite scores with fused score
    out = []
    for _id, (score, h) in fused.items():
        out.append(Hit(id=h.id, score=score, document=h.document, metadata=h.metadata, collection_name=h.collection_name))
    out.sort(key=lambda x: x.score, reverse=True)
    return out

def _guess_block_language(hit: Hit, cfg: RetrieverConfig) -> str:
    # Determine syntax highlighting language for fenced code blocks
    meta = hit.metadata
    node_type = meta.get(cfg.code_lang_key)
    if node_type in cfg.code_lang_values:
        # infer from extension (path is in variable format like {SRC}/file.cpp)
        path = meta.get("file_path") or ""
        ext = os.path.splitext(path)[1].lower()
        if ext in (".c", ".h"): return "c"
        return cfg.default_code_lang
    return cfg.default_doc_lang

def mmr_select(hits: Sequence[Hit], k: int, lambda_relevance: float = 0.5) -> List[Hit]:
    """True Maximal Marginal Relevance over document embeddings."""
    if not hits or k <= 0:
        return []
    candidates = list(hits)
    selected: List[Hit] = [candidates.pop(0)]  # start with highest‐scoring

    while len(selected) < k and candidates:
        mmr_scores: List[tuple[float, Hit]] = []
        for h in candidates:
            rel = h.score
            if h.embedding is None or any(s.embedding is None for s in selected):
                div = 0.0
            else:
                # diversity = max similarity to any already selected
                div = max(_cosine_similarity(h.embedding, s.embedding) for s in selected)  # type: ignore
            mmr_score = lambda_relevance * rel - (1 - lambda_relevance) * div
            mmr_scores.append((mmr_score, h))

        _, best = max(mmr_scores, key=lambda x: x[0])
        selected.append(best)
        candidates.remove(best)

    return selected

# ----------------------------
# Core class
# ----------------------------

class Retriever:
    def __init__(
        self,
        collections: Mapping[str, ChromaCollection],
        config: Optional[RetrieverConfig] = None,
        embedding_client: Optional[EmbedClientUni] = None,
    ) -> None:
        """
        collections: dict name -> chroma collection object
        """
        self.collections = dict(collections)
        self.cfg = config or RetrieverConfig()
        embed_model = self.cfg.embedding_model or EMBEDDING_MODEL
        self.embedder = embedding_client or EmbedClientUni(use_openai=USE_OPENAI)
        backend = "OpenAI" if USE_OPENAI else "Ollama"
        print(f"Retriever initialized with {backend} embedding model: {embed_model}")
        self._oa_client: Optional[Any] = None

    # --------- Embedding ---------
    def embed_query(self, query: str) -> List[float]:
        return self.embedder.embed(query)

    def _get_openai_client(self):
        """Lazy init so rerank users only pay for OpenAI dependency when needed."""
        if OpenAI is None:
            raise RuntimeError("openai package is required when llm rerank is enabled")
        if self._oa_client is None:
            self._oa_client = OpenAI()
        return self._oa_client

    # --------- Retrieval ---------
    def _query_one(self, name: str, col, q_emb: List[float], k: int) -> List[Hit]:
        res = col.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res.get("distances", [[None]*len(ids)])[0]
        embs  = res.get("embeddings", [[None]*len(ids)])[0]

        out: List[Hit] = []
        for rid, doc, meta, dist, emb in zip(ids, docs, metas, dists, embs):
            score = _cosine_to_similarity(dist) if dist is not None else 0.0
            out.append(Hit(
               id=rid,
               score=score,
               document=doc,
               metadata=meta,
               collection_name=name,
               embedding=emb
            ))

        return out

    def retrieve(
        self,
        query: str,
        k_per_collection: Optional[Mapping[str,int]] = None,
        where_filters: Optional[Mapping[str, Dict[str, Any]]] = None,  # not used yet but could be
    ) -> List[Hit]:
        """Embed + query all collections + fuse.
        where_filters: dict name -> where filter
        """
        kcol = k_per_collection or self.cfg.k_per_collection
        q_emb = self.embed_query(query)

        per_col_hits: List[List[Hit]] = []
        for name, col in self.collections.items():
            k = kcol.get(name, 5)
            hits = self._query_one(name, col, q_emb, k)
            per_col_hits.append(hits)

        fused = reciprocal_rank_fusion(per_col_hits, b=self.cfg.rrf_b)
        if self.cfg.use_llm_rerank:
            print(f"Asking rerank from llm {self.cfg.llm_rerank_model}..")
            fused = self.llm_rerank(query, fused, top_n=self.cfg.max_snippets)
            print("rerank received.")
        elif self.cfg.use_mmr:
            start_time = time.perf_counter()
            fused = mmr_select(fused, k=self.cfg.max_snippets, lambda_relevance=self.cfg.mmr_lambda)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            print(f"mmr_select() done in {elapsed_ms:.2f} ms")
        return fused

    # --------- Context / Prompt building ---------
    def llm_rerank(self, query: str, hits: Sequence[Hit], top_n: Optional[int] = None) -> List[Hit]:
        """Use a cheap LLM to score each chunk 0–5 and return the top_n hits sorted by that score."""
        cfg = self.cfg
        n = top_n or cfg.max_snippets
        candidates = list(hits[:cfg.max_candidates_for_rerank])
        if not candidates:
            return []

        max_tok = getattr(cfg, 'llm_rerank_max_chunk_tokens', 512)

        def _trunc(t: str) -> str:
            try:
                truncated, _ = truncate_text(t, max_tok)
                return truncated
            except RuntimeError:
                # Fallback if tokenizer not available
                return t[:max_tok*4]

        parts = []
        for h in candidates:
            parts.append(f"ID: {h.id} TEXT:{_trunc(h.document)}")
        chunk_block = "-----".join(parts)

        sys_prompt = (
            "You will receive a user query and a list of text chunks. "
            "For EACH chunk, assign an integer relevance score 0-5 (5 = essential, 0 = irrelevant). "
            "Return ONLY a JSON array like [{\"id\": \"...\", \"score\": 3}, ...]. No other text."
        )
        user_prompt = f"QUERY: {query} CHUNKS: {chunk_block}"

        client = self._get_openai_client()
        resp = client.chat.completions.create(
            model=cfg.llm_rerank_model,
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0,
        )
        try:
            # narrow Optional[str] → str
            content = resp.choices[0].message.content
            if content is None:
                raise ValueError("LLM returned no content")
            data = json.loads(content)

            score_map = {str(d['id']): float(d.get('score', 0)) for d in data if 'id' in d}
        except Exception:
            # Fallback to original order if parse fails
            return candidates[:n]

        ranked = sorted(candidates, key=lambda h: score_map.get(h.id, 0.0), reverse=True)
        return ranked[:n]

    def build_context(self, hits: Sequence[Hit]) -> Tuple[str, List[Dict[str, Any]]]:
        """Return (context_text, sources_list).
        sources_list: list of dicts with id, source_label, collection_name
        """
        context_parts: List[str] = []
        sources: List[Dict[str, Any]] = []

        budget = self.cfg.max_context_tokens
        used_tokens = 0
        used_snippets = 0

        for h in hits:
            if used_snippets >= self.cfg.max_snippets:
                break
            lang = _guess_block_language(h, self.cfg)
            header = f"— {h.source_label()}\n" if self.cfg.show_source_header else ""
            snippet = f"```{lang}\n{h.document}\n```\n{header}"
            tl = _token_len(snippet)
            if used_tokens + tl > budget and used_snippets > 0:
                break
            context_parts.append(snippet)
            used_tokens += tl
            used_snippets += 1
            sources.append({
                "id": h.id,
                "source": h.source_label(),
                "collection": h.collection_name,
                "score": h.score,
            })

        return "\n\n".join(context_parts), sources

    def build_messages(self, query: str, context: str, use_full_knowledge: bool = False) -> List[ChatCompletionMessageParam]:
        system_msg = self.cfg.system_template_full if use_full_knowledge else self.cfg.system_template
        user_msg = self.cfg.user_template.format(query=query, context=context)
        messages: List[ChatCompletionMessageParam] = [
            cast(ChatCompletionMessageParam, {"role": "system", "content": system_msg}),
            cast(ChatCompletionMessageParam, {"role": "user", "content": user_msg}),
        ]
        return messages

def ask_llm(query: str, retriever: Retriever, model: str = LLM_DEFAULT_MODEL, use_full_knowledge: bool = False):
    # 1) retrieve + build context
    hits = retriever.retrieve(query)
    ctx, sources = retriever.build_context(hits)
    messages = retriever.build_messages(query, ctx, use_full_knowledge)

    if USE_OPENAI:
        # 2) call OpenAI
        know_str='full knowledge' if use_full_knowledge else 'only context'
        print(f"Calling OpenAI [{know_str}]...")
        temperature = 1 if ("-mini" in model or "-nano" in model) else 0  # set 1 for mini/nano models, else 0
        oa = retriever._get_openai_client()
        resp = oa.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            temperature=temperature
        )
        print("Got answer from OpenAI")
        answer = resp.choices[0].message.content
    else:
        # 2) call Ollama
        know_str='full knowledge' if use_full_knowledge else 'only context'
        print(f"Calling Ollama [{know_str}]...")
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,  # Explicitly disable streaming
        }
        with httpx.Client(base_url=OLLAMA_BASE_URL, timeout=None) as client:
            resp = client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        message = data.get("message") or {}
        answer = message.get("content") or data.get("response", "")
        print("Got answer from Ollama")
    return answer, sources


# ----------------------------
# CLI example (optional)
# ----------------------------
def create_retriever():
    client = chromadb.PersistentClient(path=CHROMA_DB_FULL_PATH)
    all_collections = client.list_collections()
    print(f"All available collections in the DB: {[collection.name for collection in all_collections]}")
    cols = {c.name: c for c in all_collections}
    r = Retriever(cols, RetrieverConfig())
    return r

def _demo_cli():  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(description="Multi-collection Chroma retriever")
    parser.add_argument("query", type=str, nargs="?", default="", help="user query text")
    args = parser.parse_args()
    print("_demo_cli()")
    retr = create_retriever()

    if args.query:
        q=args.query
    else:
#        q = "Does Bullet3 support soft body dynamics, and if so, what are some examples?"
#        q = "What are the primary components of a Bullet3 physics world setup?"
#        q = "How do you create a basic rigid body in Bullet3 using the C++ API?"
#        q="What is the purpose of the Bullet3 physics library?"
#        q = "How do I detect collisions between objects in Bullet3?"
#        q = "What types of constraints are available in Bullet3 and how do I create a hinge joint?"
#       q = "What value of timeStep is recommended for the integration?"
#        q = "What numerical integration method does Bullet3 use for dynamics simulation?"
#        q = "In stepSimulation() why we need to clamp the number of substeps?"
#        q = "How can I perform raycasting using Bullet3?"
#        q = "Explain dynamicsWorld->rayTest()"
#        q = "What examples are provided in Bullet3 library?"
#        q = "Describe how numerical integration is performed in the physics simulation?"
#        q = "What is new in Bullet version 2.81?"
#        q = "How to compute the object AABBs?"
#        q = "Where class btMotionState is defined?"
#        q = "Does Coriolis force is taken into account for bodies that fly around Earth?"
#        q = "Explain struct LuaPhysicsSetup"
#        q = "In stepSimulation() why we need to clamp the number of substeps?"
#       q = "Describe DeformableDemo example"
#        q = "What is Jacobi solver?"
#        q = "Explain b3TestTriangleAgainstAabb2"
#        q = "What are Bullet Basic Data Types?"

#        q = "How to solve several concurrent constraints?"
#        q = "Explain position based dynamics"
#        q = "Write vertex shader for per pixel lighting of a single omni plus ambient"
        q = "What are some Helpful Lies about Our Universe?"

    print("\n--- Question: ---")
    print(q)
    answer, sources = ask_llm(q, retr)
    print("\n==== Reply: =====")
    print(answer)
    print("\n=== DONE ===")

    print("\nSources:")
    for s in sources:
        print(f'source={s["source"]}, score={s["score"]:.3f}')


if __name__ == "__main__":  # pragma: no cover
    _demo_cli()
