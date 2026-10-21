"""
Retrieval over the ingested documentation.

Returns source-aware results and, critically, says plainly when nothing relevant
was found so the assistant can answer "that isn't in the documentation" instead
of inventing an answer.
"""

from typing import Any, Dict, List, Optional

from agent import config
from agent.logging_setup import get_logger
from agent.rag.embeddings import (
    GeminiEmbedder,
    LexicalEmbedder,
    _STOPWORDS,
    get_embedder,
    tokenize,
)
from agent.rag.store import DocumentStore

log = get_logger("RAG")

_store_cache: Optional[DocumentStore] = None
_cached_dir: Optional[str] = None


def get_store(index_dir: Optional[str] = None, refresh: bool = False) -> DocumentStore:
    global _store_cache, _cached_dir
    directory = index_dir or config.RAG_INDEX_DIR
    if refresh or _store_cache is None or _cached_dir != directory:
        _store_cache = DocumentStore.load(directory)
        _cached_dir = directory
    return _store_cache


def search_documentation(
    query: str,
    top_k: Optional[int] = None,
    index_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Searches the ingested manuals/SOPs/troubleshooting documents.

    Use this for questions about procedures, specifications, error codes,
    maintenance steps or anything documented -- NOT for live machine values.

    Returns {"found": bool, "results": [{text, source, score, ...}], "message": str}.
    When `found` is False the answer must say the information is not in the
    available documentation.
    """
    store = get_store(index_dir)
    limit = top_k or config.RAG_TOP_K

    if len(store) == 0:
        return {
            "found": False,
            "results": [],
            "message": (
                "No documentation has been ingested yet. Add documents with: "
                "python ingest_docs.py <file-or-folder>"
            ),
        }

    lexical_scores = LexicalEmbedder().score(query, store.texts)
    scores: List[float] = []
    search_mode = "lexical"

    if store.is_semantic and any(c.embedding for c in store.chunks):
        embedder = get_embedder()
        query_vector = embedder.embed_query(query) if isinstance(embedder, GeminiEmbedder) else None
        if query_vector is not None:
            semantic_scores = [
                GeminiEmbedder.cosine(query_vector, c.embedding) if c.embedding else 0.0
                for c in store.chunks
            ]
            # Hybrid fusion: linear combination of semantic similarity and BM25 relevance
            scores = [
                0.6 * sem + 0.4 * lex
                for sem, lex in zip(semantic_scores, lexical_scores)
            ]
            search_mode = "hybrid"

    if not scores:
        # Offline / fallback path: deterministic lexical scoring.
        scores = lexical_scores

    ranked = sorted(zip(store.chunks, scores), key=lambda pair: pair[1], reverse=True)
    hits = [(c, s) for c, s in ranked[:limit] if s >= config.RAG_MIN_SCORE]

    # Grounding check: when a query contains specific informative terms,
    # reject spurious hits that do not actually mention the core subject.
    # For queries with >= 3 informative tokens, candidate chunks must cover
    # at least 50% of the query's terms.
    informative_query_terms = [
        t for t in tokenize(query, expand_synonyms=True) if t not in _STOPWORDS
    ]
    if len(informative_query_terms) >= 3:
        grounded_hits = []
        for c, s in hits:
            chunk_tokens = set(tokenize(c.text))
            matched = [t for t in informative_query_terms if t in chunk_tokens]
            if len(matched) >= len(informative_query_terms) * 0.5:
                grounded_hits.append((c, s))
        hits = grounded_hits

    if not hits:
        return {
            "found": False,
            "results": [],
            "message": (
                "That information was not found in the available documentation. "
                f"({len(store)} chunks searched across {len(store.files())} document(s).)"
            ),
        }

    return {
        "found": True,
        "results": [
            {
                "text": chunk.text,
                "source": chunk.citation(),
                "score": round(float(score), 3),
                **chunk.metadata(),
            }
            for chunk, score in hits
        ],
        "message": f"{len(hits)} relevant passage(s) found.",
        "search_mode": search_mode,
    }


def index_summary(index_dir: Optional[str] = None) -> Dict[str, Any]:
    store = get_store(index_dir, refresh=True)
    return {
        "chunks": len(store),
        "documents": store.files(),
        "embedder": store.embedder_name or "none",
        "semantic": store.is_semantic,
    }
