"""
Local vector/chunk store for the RAG index.

Deliberately plain JSON on disk rather than a vector database: the corpus here
is machine manuals and SOPs (hundreds to low thousands of chunks), where a
dependency-free store is faster to operate and easier to inspect. Swap in
Chroma/FAISS later behind the same three methods if the corpus grows.

Each chunk keeps the metadata needed for source-aware answers:
filename, document type, page number, section, chunk index, ingested_at.
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from agent import config
from agent.logging_setup import get_logger

log = get_logger("RAG")

_INDEX_FILE = "index.json"


@dataclass
class Chunk:
    text: str
    filename: str
    doc_type: str = "document"
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_index: int = 0
    ingested_at: float = 0.0
    embedding: Optional[List[float]] = None

    def citation(self) -> str:
        parts = [self.filename]
        if self.page is not None:
            parts.append(f"page {self.page}")
        if self.section:
            parts.append(self.section)
        return ", ".join(parts)

    def metadata(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "doc_type": self.doc_type,
            "page": self.page,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "ingested_at": self.ingested_at,
        }


@dataclass
class DocumentStore:
    chunks: List[Chunk] = field(default_factory=list)
    embedder_name: str = ""
    is_semantic: bool = False

    # --- persistence ---

    @staticmethod
    def _path(index_dir: Optional[str] = None) -> str:
        return os.path.join(index_dir or config.RAG_INDEX_DIR, _INDEX_FILE)

    def save(self, index_dir: Optional[str] = None) -> str:
        directory = index_dir or config.RAG_INDEX_DIR
        os.makedirs(directory, exist_ok=True)
        path = self._path(directory)
        payload = {
            "embedder": self.embedder_name,
            "is_semantic": self.is_semantic,
            "chunks": [asdict(c) for c in self.chunks],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        log.info("Saved %d chunks to %s", len(self.chunks), path)
        return path

    @classmethod
    def load(cls, index_dir: Optional[str] = None) -> "DocumentStore":
        path = cls._path(index_dir)
        if not os.path.exists(path):
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            log.warning("Could not read RAG index %s: %s", path, exc)
            return cls()
        return cls(
            chunks=[Chunk(**c) for c in payload.get("chunks", [])],
            embedder_name=payload.get("embedder", ""),
            is_semantic=bool(payload.get("is_semantic")),
        )

    # --- content ---

    def add(self, chunks: List[Chunk]) -> None:
        self.chunks.extend(chunks)

    def remove_file(self, filename: str) -> int:
        """Drops all chunks for one file, so re-ingesting doesn't duplicate it."""
        before = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.filename != filename]
        return before - len(self.chunks)

    def remove_issues(self, issue_ids: List[str]) -> int:
        """Drops any chunks matching an issue ID to prevent duplicates across files."""
        if not issue_ids:
            return 0
        before = len(self.chunks)
        patterns = [
            re.compile(rf"##\s*Issue:\s*{re.escape(iid)}\b", re.IGNORECASE)
            for iid in issue_ids
        ]
        self.chunks = [
            c for c in self.chunks
            if not any(p.search(c.text) for p in patterns)
        ]
        return before - len(self.chunks)

    @property
    def texts(self) -> List[str]:
        return [c.text for c in self.chunks]

    def files(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for c in self.chunks:
            counts[c.filename] = counts.get(c.filename, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self.chunks)
