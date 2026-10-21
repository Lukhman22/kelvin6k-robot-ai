"""
Embedding backends for the RAG index.

Two are provided deliberately:

  * GeminiEmbedder -- real semantic embeddings via the Gemini API. Used when
    GEMINI_API_KEY is set. Best retrieval quality.
  * LexicalEmbedder -- a dependency-free BM25-style scorer used when no API key
    is configured (and by the test suite). Retrieval is keyword-based rather
    than semantic, but it is deterministic, offline, and never silently
    pretends to be semantic: `is_semantic` says which one is in use.

Both expose the same interface so the index format and retriever are identical.
"""

import math
import re
from collections import Counter
from typing import List, Optional, Sequence

from agent import config
from agent.logging_setup import get_logger

log = get_logger("RAG")

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Dropped from the QUERY before scoring (never from the documents). A question
# word carries no retrieval signal, but it does distort the relevance ceiling in
# `LexicalEmbedder.score`: words like "how" or "i" are usually absent from a
# small technical corpus, which gives them maximum idf and lets phrasing --
# rather than subject -- decide the score.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am",
    "how", "what", "which", "when", "where", "why", "who", "often",
    "do", "does", "did", "can", "could", "should", "would", "will",
    "i", "me", "my", "we", "us", "our", "you", "your", "it", "its",
    "to", "of", "in", "on", "for", "at", "by", "from", "with", "about",
    "and", "or", "but", "if", "this", "that", "these", "those",
    "tell", "please", "give", "show", "need", "want",
})


_SYNONYMS = {
    "saaf": "clean",
    "saf": "clean",
    "badalna": "replace",
    "badalni": "replace",
    "badle": "replace",
    "roz": "daily",
    "hafta": "weekly",
    "hafte": "weekly",
    "ghante": "hours",
    "ghanta": "hour",
    "din": "day",
    "patthar": "stone",
    "geela": "wet",
    "geeli": "wet",
    "hawa": "air",
    "achanak": "sudden",
    "kam": "drop",
    "zyada": "excessive",
    "rok": "stop",
    "roko": "pause",
    "band": "stop",
    "shuru": "start",
    "rukna": "pause",
}


def tokenize(text: str, expand_synonyms: bool = False) -> List[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if not expand_synonyms:
        return tokens
    expanded: List[str] = []
    for t in tokens:
        expanded.append(t)
        syn = _SYNONYMS.get(t)
        if syn and syn != t:
            expanded.append(syn)
    return expanded


class LexicalEmbedder:
    """
    BM25 scorer dressed as an embedder.

    It stores no vectors; scoring happens in `score()` against the raw chunk
    text, which the retriever passes in. This keeps the offline path free of
    numpy/torch and of fake "semantic" claims.
    """

    name = "lexical-bm25"
    is_semantic = False

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def embed_documents(self, texts: Sequence[str]) -> Optional[List[List[float]]]:
        return None  # no vectors -- scoring is done lexically at query time

    def embed_query(self, text: str) -> Optional[List[float]]:
        return None

    def score(self, query: str, chunk_texts: Sequence[str]) -> List[float]:
        """BM25 scores in 0..1, comparable against RAG_MIN_SCORE across queries."""
        query_terms = tokenize(query, expand_synonyms=True)
        if not query_terms or not chunk_texts:
            return [0.0] * len(chunk_texts)

        # Stopwords are removed from the query only. If a question is nothing
        # but stopwords there is no signal to drop, so keep the raw tokens.
        informative = [t for t in query_terms if t not in _STOPWORDS]
        query_terms = informative or query_terms

        docs = [tokenize(t) for t in chunk_texts]
        doc_count = len(docs)
        avg_len = sum(len(d) for d in docs) / max(1, doc_count)
        doc_freq = Counter()
        for doc in docs:
            for term in set(doc):
                doc_freq[term] += 1

        def idf(term: str) -> float:
            return math.log(1 + (doc_count - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))

        scores = []
        for doc in docs:
            counts = Counter(doc)
            length = len(doc) or 1
            score = 0.0
            for term in query_terms:
                if term not in counts:
                    continue
                freq = counts[term]
                denom = freq + self.k1 * (1 - self.b + self.b * length / avg_len)
                score += idf(term) * (freq * (self.k1 + 1)) / denom
            scores.append(score)

        # Normalise against the best score this query COULD achieve, not the
        # best it did. Dividing by max(scores) made the top hit exactly 1.0 by
        # construction, so RAG_MIN_SCORE could never reject it -- a chunk
        # sharing a single word outranked the "not in the documentation"
        # answer, which is how a wrong passage reached the operator as fact.
        #
        # The ceiling is BM25's limit as term frequency grows: idf * (k1 + 1)
        # per distinct query term. A chunk matching every term approaches 1.0;
        # one matching a single term of five scores near zero. Query terms
        # absent from the corpus stay in the ceiling deliberately -- they are
        # what makes an unanswerable question score low instead of scoring 1.0.
        ceiling = sum(idf(term) * (self.k1 + 1) for term in set(query_terms))
        if ceiling <= 0:
            return [0.0] * len(chunk_texts)
        return [s / ceiling for s in scores]


class GeminiEmbedder:
    """Semantic embeddings via the Gemini embedding model."""

    is_semantic = True

    def __init__(self, model: Optional[str] = None):
        self.name = model or config.GEMINI_EMBED_MODEL
        self._client = None

    def _client_or_none(self):
        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(api_key=config.GEMINI_API_KEY)
            except Exception as exc:
                log.warning("Could not create Gemini client for embeddings: %s", exc)
                return None
        return self._client

    def _embed(self, texts: Sequence[str]) -> Optional[List[List[float]]]:
        client = self._client_or_none()
        if client is None:
            return None
        try:
            batch_size = 50
            all_embeddings = []
            text_list = list(texts)
            for i in range(0, len(text_list), batch_size):
                batch = text_list[i : i + batch_size]
                response = client.models.embed_content(model=self.name, contents=batch)
                all_embeddings.extend([list(e.values) for e in response.embeddings])
            return all_embeddings
        except Exception as exc:
            log.warning("Embedding request failed (%s); falling back to lexical search.", exc)
            return None

    def embed_documents(self, texts: Sequence[str]) -> Optional[List[List[float]]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> Optional[List[float]]:
        vectors = self._embed([text])
        return vectors[0] if vectors else None

    @staticmethod
    def cosine(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


def get_embedder(prefer_semantic: bool = True):
    """Gemini when a key is configured, otherwise the offline lexical scorer."""
    if prefer_semantic and config.GEMINI_API_KEY:
        return GeminiEmbedder()
    return LexicalEmbedder()
