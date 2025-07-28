"""
retriever.py
--------------

Query -> retrieve from multiple Chroma collections (e.g. "cpp_code" and "bullet_docs")
-> fuse results -> build a prompt/context block for an OpenAI chat model.

Features
=========
- Single embedding of the query (OpenAI embeddings API).
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

    # send `messages` to OpenAI Chat Completions

"""
from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import tiktoken  # type: ignore
from openai import OpenAI
import chromadb
from chromadb.api.models.Collection import Collection as ChromaCollection
from dotenv import load_dotenv
from config import DOCUMENTS_PATH, SOURCES_PATH, EXAMPLES_PATH, CHROMA_DB_DIR, EMBEDDING_MODEL
from types import MappingProxyType

CHROMA_DB_FULL_PATH = os.path.expanduser(CHROMA_DB_DIR)

load_dotenv()

# ----------------------------
# Data structures
# ----------------------------

@dataclass
class RetrieverConfig:
    embedding_model: str = EMBEDDING_MODEL
    distance_metric: str = "cosine"  # assume Chroma collection uses cosine
    rrf_b: int = 60                   # RRF hyperparameter
    mmr_lambda: float = 0.5           # 0 -> diversity only, 1 -> relevance only
    use_mmr: bool = True             # turn on if you want diversity
    max_context_tokens: int = 6000    # budget for snippets
    max_snippets: int = 12            # hard cap on snippets even if under token budget
    code_lang_key: str = "node_type"  # metadata key that tells us if it's code
    code_lang_values: Sequence[str] = ("function", "leftover_block")  # values that indicate code
    default_code_lang: str = "cpp"
    default_doc_lang: str = "markdown"
    show_source_header: bool = True
    # How many results (k) to fetch per collection
    k_per_collection: Mapping[str,int] = MappingProxyType({"cpp_code":12, "bullet_docs":4})
    # Build messages: system / user templates
    system_template: str = (
        "You are a precise assistant for C/C++ code and project documentation. "
        "Use only the given CONTEXT to answer. If unsure, say you don't know. "
        "Format your responses in markdown, and use LaTeX math notation (enclosed in $...$) for mathematical expressions and formulas."
        "Cite sources by file path, page number and line numbers when available."
    )

    user_template: str = (
        "QUESTION:\n{query}\n\nCONTEXT:\n{context}\n\n"
        "Please answer using markdown, show code in fenced blocks, and include citations also in markdown format like (Source: [file name](path) : page,line-range)."
    )

    use_llm_rerank: bool = False
    llm_rerank_model: str = "gpt-3.5-turbo" # "gpt-4o-mini"
    max_candidates_for_rerank: int = 30
    llm_rerank_max_chunk_tokens: int = 512

def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x*y for x,y in zip(a,b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(y*y for y in b))
    return dot/(norm_a*norm_b) if norm_a and norm_b else 0.0

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
    enc = tiktoken.get_encoding(model)
    return len(enc.encode(text))


def _cosine_to_similarity(distance: float) -> float:
    return 1.0 - distance


def reciprocal_rank_fusion(list_of_lists: Sequence[Sequence[Hit]], b: int = 60) -> List[Hit]:
    """Fuse multiple ranked lists of Hit using RRF.
    Returns a single list of Hit with fused scores.
    """
    fused: Dict[str, Tuple[float, Hit]] = {}
    for lst in list_of_lists:
        for rank, h in enumerate(lst, start=1):
            contrib = 1.0 / (b + rank)
            if h.id in fused:
                fused[h.id] = (fused[h.id][0] + contrib, fused[h.id][1])
            else:
                # copy hit but replace score later
                fused[h.id] = (contrib, h)
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
        # infer from extension
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
        openai_client: Optional[OpenAI] = None,
    ) -> None:
        """
        collections: dict name -> chroma collection object
        """
        self.collections = dict(collections)
        self.cfg = config or RetrieverConfig()
        self.oa = openai_client or OpenAI()

    # --------- Embedding ---------
    def embed_query(self, query: str) -> List[float]:
        resp = self.oa.embeddings.create(model=self.cfg.embedding_model, input=query)
        return resp.data[0].embedding

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
            print("asking use mmr_select()")
            fused = mmr_select(fused, k=self.cfg.max_snippets, lambda_relevance=self.cfg.mmr_lambda)
            print("mmr_select() done.")
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
        if tiktoken is not None:
            enc_local = tiktoken.get_encoding("cl100k_base")
            def _trunc(t: str) -> str:
                toks = enc_local.encode(t)
                return enc_local.decode(toks[:max_tok]) if len(toks) > max_tok else t
        else:
            def _trunc(t: str) -> str:
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

        resp = self.oa.chat.completions.create(
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

    def build_messages(self, query: str, context: str) -> List[Dict[str, str]]:
        system_msg = self.cfg.system_template
        user_msg = self.cfg.user_template.format(query=query, context=context)
        return [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ]


def ask_llm(query: str, retriever, model="gpt-4o-mini", streaming=False):
    # 1) retrieve + build context
    hits = retriever.retrieve(query)
    ctx, sources = retriever.build_context(hits)
    messages = retriever.build_messages(query, ctx)

    # 2) call OpenAI
    print("Calling OpenAI...")
    temperature = 1 if "o4-mini" in model else 0 # temperature is not supported (only 1) in o4-mini
    resp = retriever.oa.chat.completions.create(
        model=model,
        messages=messages,
        stream=streaming,
        temperature=temperature
    )
    if streaming:
        print("Start stream from OpenAI")
        return resp, sources
    else:
        print("Got answer from OpenAI")

    answer = resp.choices[0].message.content
    return answer, sources


# ----------------------------
# CLI example (optional)
# ----------------------------
def create_retriever():
    client = chromadb.PersistentClient(path=CHROMA_DB_FULL_PATH)
    all_collections = client.list_collections()
    print(f"All available collections in the DB: {[collection.name for collection in all_collections]}")
    collections_names = ["cpp_code", "bullet_docs"]
    cols = {name: client.get_collection(name) for name in collections_names}

    r = Retriever(cols, RetrieverConfig())
    return r

def _demo_cli():  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(description="Multi-collection Chroma retriever")
    parser.add_argument("query", type=str, nargs="?", default="", help="user query text")
    args = parser.parse_args()
    print("_demo_cli()")
    retr = create_retriever()

    #hits = retr.retrieve(args.query)
    #context, sources = retr.build_context(hits)
    #messages = retr.build_messages(args.query, context)

    # print("\n--- SOURCES ---")
    # for s in sources:
    #     print(f"[{s['collection']}] {s['source']} (score={s['score']:.3f})")

    # print("\n--- CONTEXT ---\n")
    # print(context[:2000] + ("..." if len(context) > 2000 else ""))

    # print("\n--- MESSAGES ---\n")
    # for m in messages:
    #     print(m["role"].upper()+":\n"+m["content"])
    #     print()
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
        q = "Describe DeformableDemo example"

        pass
    print("\n--- Question: ---")
    print(q)
    answer, sources = ask_llm(q, retr)
    print("\n==== Reply: =====")
    print(answer)
    print("\n=== DONE ===")

    print("\nSources:")
    for s in sources:
        print(s["source"])


if __name__ == "__main__":  # pragma: no cover
    _demo_cli()
