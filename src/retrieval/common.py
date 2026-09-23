"""Shared scoped retrieval and JSON persistence helpers."""
from hashlib import sha256
import json
from pathlib import Path

from src.contracts import Chunk, Evidence, SearchRequest, validate_evidence


def validate_chunks(chunks):
    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs")


def eligible(chunk: Chunk, request: SearchRequest) -> bool:
    if not request.user.permits(chunk.doc_id):
        return False
    if request.source_types and chunk.source_type not in request.source_types:
        return False
    if request.after or request.before:
        if chunk.timestamp is None:
            return False
        if request.after and chunk.timestamp < request.after:
            return False
        if request.before and chunk.timestamp > request.before:
            return False
    return True


def evidence_for(chunks, scores, request, method):
    ranked = sorted(zip(chunks, scores), key=lambda pair: (-float(pair[1]), pair[0].chunk_id))
    results = tuple(Evidence(citation_id=c.chunk_id, chunk=c, score=float(score),
                             retrieval_method=method)
                    for c,score in ranked[:request.top_k])
    validate_evidence(request.user, results)
    return results


def save_metadata(path: Path, chunks, method: str, config: dict, provenance: dict):
    path.mkdir(parents=True, exist_ok=True)
    payload = "".join(c.model_dump_json()+"\n" for c in chunks)
    (path / "chunks.jsonl").write_text(payload)
    manifest = {"version":1, "method":method, "config":config, "provenance":provenance,
                "chunk_count":len(chunks), "chunks_sha256":sha256(payload.encode()).hexdigest()}
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    return manifest


def load_metadata(path: Path, method: str):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["version"] != 1 or manifest["method"] != method:
        raise ValueError("Unsupported index format/method")
    payload = (path / "chunks.jsonl").read_text()
    if sha256(payload.encode()).hexdigest() != manifest["chunks_sha256"]:
        raise ValueError("Chunk artifact checksum mismatch")
    chunks = tuple(Chunk.model_validate_json(line) for line in payload.splitlines())
    validate_chunks(chunks)
    if len(chunks) != manifest["chunk_count"]:
        raise ValueError("Chunk count mismatch")
    return chunks, manifest
