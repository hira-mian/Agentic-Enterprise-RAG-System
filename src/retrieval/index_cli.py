"""Build or query local baseline indexes; explicit benchmark scope is required."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

from src.config import EMBEDDING_REVISION
from src.contracts import Document, SearchRequest, UserContext
from src.data.preprocessing import ChunkingConfig, chunk_documents
from src.retrieval.bm25 import BM25Index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build")
    build.add_argument("--input", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--method", choices=["bm25", "dense"], default="bm25")
    build.add_argument("--chunk-size", type=int, default=1000)
    build.add_argument("--overlap", type=int, default=100)
    build.add_argument("--limit", type=int, default=256)
    search = sub.add_parser("search")
    search.add_argument("--index", type=Path, required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--allow-doc", action="append", default=[])
    search.add_argument("--benchmark-all-docs", action="store_true")
    search.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.action == "build":
        if args.limit < 1:
            parser.error("--limit must be positive")
        documents, digest = [], sha256()
        with args.input.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                digest.update(line.encode())
                documents.append(Document.model_validate_json(line))
                if len(documents) == args.limit:
                    break
        config = ChunkingConfig(size=args.chunk_size, overlap=args.overlap)
        chunks = chunk_documents(documents, config)
        provenance = {
            "selected_input_sha256": digest.hexdigest(),
            "documents": len(documents),
            "chunking": config.model_dump(),
        }
        if args.method == "bm25":
            index = BM25Index(chunks, provenance=provenance)
        else:
            from src.retrieval.dense import BGEEncoder, DenseIndex

            index = DenseIndex(
                chunks, BGEEncoder(EMBEDDING_REVISION), provenance=provenance
            )
        index.save(args.output)
        print(
            json.dumps(
                {
                    "documents": len(documents),
                    "chunks": len(chunks),
                    "method": args.method,
                }
            )
        )
    else:
        manifest = json.loads((args.index / "manifest.json").read_text())
        if manifest["method"] == "bm25":
            index = BM25Index.load(args.index)
        elif manifest["method"] == "dense":
            from src.retrieval.dense import BGEEncoder, DenseIndex

            encoder = BGEEncoder(
                manifest["config"]["revision"], model_name=manifest["config"]["name"]
            )
            index = DenseIndex.load(args.index, encoder)
        else:
            parser.error("Unsupported index method")
        ids = (
            {c.doc_id for c in index.chunks}
            if args.benchmark_all_docs
            else set(args.allow_doc)
        )
        request = SearchRequest(
            query=args.query,
            top_k=args.top_k,
            user=UserContext(user_id="local-benchmark", allowed_doc_ids=ids),
        )
        print(
            json.dumps(
                [
                    {
                        "doc_id": e.chunk.doc_id,
                        "chunk_id": e.chunk.chunk_id,
                        "score": e.score,
                        "excerpt": e.chunk.text[:200],
                    }
                    for e in index.search(request)
                ],
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
