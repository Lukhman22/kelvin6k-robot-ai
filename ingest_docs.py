"""
Add documents to the Kelvin6K RAG index (manuals, SOPs, troubleshooting, etc.).

    python ingest_docs.py docs/                  # ingest a folder (recursive)
    python ingest_docs.py manual.pdf             # ingest one file
    python ingest_docs.py --list                 # what's currently indexed
    python ingest_docs.py --search "nozzle clog" # test retrieval

Supported: .txt .md .rst .log .pdf   (PDF needs: pip install pypdf)
Re-ingesting a file replaces its previous chunks rather than duplicating them.
"""

import argparse
import sys

from agent import config
from agent.logging_setup import configure_logging
from agent.rag.ingest import ingest_path
from agent.rag.retriever import index_summary, search_documentation


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG index")
    parser.add_argument("path", nargs="?", default=None, help="file or folder to ingest")
    parser.add_argument("--list", action="store_true", help="show what is indexed")
    parser.add_argument("--search", type=str, default=None, help="test a retrieval query")
    parser.add_argument("--index-dir", type=str, default=None, help="override RAG_INDEX_DIR")
    args = parser.parse_args()

    configure_logging()

    if args.list:
        summary = index_summary(args.index_dir)
        print("=== RAG index ===")
        print(f"Location : {args.index_dir or config.RAG_INDEX_DIR}")
        print(f"Chunks   : {summary['chunks']}")
        print(f"Embedder : {summary['embedder']} ({'semantic' if summary['semantic'] else 'lexical'})")
        if summary["documents"]:
            print("Documents:")
            for name, count in sorted(summary["documents"].items()):
                print(f"  - {name}: {count} chunks")
        else:
            print("Documents: none -- ingest some with: python ingest_docs.py docs/")
        return 0

    if args.search:
        result = search_documentation(args.search, index_dir=args.index_dir)
        print(f"=== Search: {args.search!r} ===")
        print(result["message"])
        for hit in result["results"]:
            print(f"\n[{hit['score']}] {hit['source']}  ({hit['doc_type']})")
            print(hit["text"][:400])
        return 0 if result["found"] else 1

    if not args.path:
        parser.print_help()
        return 1

    print(f"Ingesting {args.path} ...")
    result = ingest_path(args.path, args.index_dir)

    if not result["ok"]:
        print(f"FAILED: {result.get('error')}", file=sys.stderr)
        return 1

    print(f"Files ingested : {result['files']}  ({', '.join(result['ingested'])})")
    print(f"Chunks added   : {result['chunks']}")
    print(f"Index total    : {result['total_chunks_in_index']} chunks")
    print(f"Search mode    : {'semantic embeddings' if result['semantic'] else 'lexical (no GEMINI_API_KEY)'}")
    print(f"Saved to       : {result['index_path']}")
    for error in result["errors"]:
        print(f"  WARNING: {error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
