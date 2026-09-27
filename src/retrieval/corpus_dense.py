"""Full-corpus BGE-small + FAISS document retrieval (max score over passages).

Documents longer than the model limit are split into overlapping token windows.
`embed` writes resumable shards (run it on a GPU); `rank` searches all passages
with exact inner product and scores each document by its best passage.
"""

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import CORPUS_DIR, EMBEDDING_MODEL, EMBEDDING_REVISION, ROOT
from src.retrieval.corpus_bm25 import document_text, file_sha256

INDEX_VERSION = 1
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
WINDOW_TOKENS = 500  # BGE accepts 512 tokens including [CLS]/[SEP]; leave margin.
OVERLAP_TOKENS = 50
PASSAGE_SCHEMA = pa.schema(
    [("doc_index", pa.int32()), ("start_char", pa.int32()), ("end_char", pa.int32())]
)


def windows(offsets, window=WINDOW_TOKENS, overlap=OVERLAP_TOKENS):
    """Character spans of overlapping token windows covering a document."""
    if not offsets:
        return []
    if overlap >= window:
        raise ValueError("overlap must be smaller than window")
    spans, start, n = [], 0, len(offsets)
    while True:
        end = min(start + window, n)
        spans.append((offsets[start][0], offsets[end - 1][1]))
        if end == n:
            return spans
        start = end - overlap


class BGEModel:
    """Pinned sentence-transformers BGE model; fp16 on GPU unless disabled."""

    def __init__(self, device="cpu", fp16=None, batch_size=128):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            revision=EMBEDDING_REVISION,
            device=device,
            cache_folder=str(ROOT / ".cache" / "models"),
            trust_remote_code=False,
        )
        self.fp16 = device.startswith("cuda") if fp16 is None else fp16
        if self.fp16:
            self.model.half()
        self.batch_size = batch_size
        self.identity = {
            "name": EMBEDDING_MODEL,
            "revision": EMBEDDING_REVISION,
            "query_prefix": QUERY_PREFIX,
            "normalize": True,
            "max_seq_length": self.model.max_seq_length,
            "fp16": self.fp16,
        }

    def offsets(self, texts):
        encoded = self.model.tokenizer(
            texts,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        return encoded["offset_mapping"]

    def encode(self, texts, query=False):
        inputs = [QUERY_PREFIX + t for t in texts] if query else texts
        return self.model.encode(
            inputs,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)


def embed_corpus(
    corpus_path: Path,
    output: Path,
    model,
    shard_docs=20000,
    window=WINDOW_TOKENS,
    overlap=OVERLAP_TOKENS,
):
    """Write one embedding shard per `shard_docs` documents; existing shards are kept."""
    output.mkdir(parents=True, exist_ok=True)
    corpus = pq.ParquetFile(corpus_path)
    manifest = {
        "version": INDEX_VERSION,
        "method": "bge-small + faiss (max over passages)",
        "unit": f"{window}-token windows, {overlap}-token overlap, of title + content",
        "model": model.identity,
        "corpus_sha256": file_sha256(corpus_path),
        "documents": corpus.metadata.num_rows,
        "shard_docs": shard_docs,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if {k: previous.get(k) for k in manifest} != manifest:
            raise ValueError("Existing shards were made with different settings")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    shard, start, doc_ids = 0, 0, []
    batches = corpus.iter_batches(
        batch_size=shard_docs, columns=["doc_id", "title", "content"]
    )
    for batch in batches:
        rows = batch.to_pydict()
        doc_ids.extend(rows["doc_id"])
        name = f"shard_{shard:05d}"
        if not (output / f"{name}.npy").exists():
            began = time.perf_counter()
            texts = list(map(document_text, rows["title"], rows["content"]))
            passages, meta = [], {"doc_index": [], "start_char": [], "end_char": []}
            for i, (text, offsets) in enumerate(zip(texts, model.offsets(texts))):
                for s, e in windows(offsets, window, overlap):
                    passages.append(text[s:e])
                    meta["doc_index"].append(start + i)
                    meta["start_char"].append(s)
                    meta["end_char"].append(e)
            vectors = model.encode(passages).astype(np.float16)
            pq.write_table(
                pa.table(meta, schema=PASSAGE_SCHEMA), output / f"{name}.parquet"
            )
            # Write vectors last: their presence marks the shard complete.
            np.save(output / f"{name}.tmp.npy", vectors)
            (output / f"{name}.tmp.npy").replace(output / f"{name}.npy")
            print(
                f"{name}: {len(texts)} docs, {len(passages)} passages, "
                f"{time.perf_counter() - began:.0f}s",
                flush=True,
            )
        start += len(rows["doc_id"])
        shard += 1
    (output / "doc_ids.json").write_text(json.dumps(doc_ids))
    manifest["complete"] = True
    manifest["shards"] = shard
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


class CorpusDense:
    def __init__(self, vectors, doc_index, doc_ids, model, manifest):
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.doc_index = doc_index
        self.doc_ids = doc_ids
        self.model = model
        self.manifest = manifest

    def rank(self, query: str, top_k: int, mask: np.ndarray | None = None):
        """Return (doc_id, score) for the top documents by best passage score."""
        return self.rank_vector(self.model.encode([query], query=True)[0], top_k, mask)

    def rank_vector(self, vector, top_k, mask=None):
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        fetch = min(max(10 * top_k, 100), self.index.ntotal)
        while True:
            scores, positions = self.index.search(vector, fetch)
            best = {}
            for score, position in zip(scores[0], positions[0]):
                if position < 0:
                    continue
                doc = int(self.doc_index[position])
                if (mask is None or mask[doc]) and doc not in best:
                    best[doc] = float(score)  # Results are sorted, so first is max.
            if len(best) >= top_k or fetch == self.index.ntotal:
                break
            fetch = min(fetch * 4, self.index.ntotal)
        ranked = sorted(
            best.items(), key=lambda item: (-item[1], self.doc_ids[item[0]])
        )
        return [(self.doc_ids[d], s) for d, s in ranked[:top_k]]

    @classmethod
    def load(cls, directory: Path, model=None, corpus_path: Path | None = None):
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("version") != INDEX_VERSION or not manifest.get("complete"):
            raise ValueError("Embedding directory is incomplete or unsupported")
        if (
            corpus_path is not None
            and file_sha256(corpus_path) != manifest["corpus_sha256"]
        ):
            raise ValueError("Corpus differs from the one used for the embeddings")
        if model is not None and model.identity != manifest["model"]:
            raise ValueError("Query model settings differ from the embeddings")
        names = [f"shard_{i:05d}" for i in range(manifest["shards"])]
        vectors = np.concatenate(
            [np.load(directory / f"{n}.npy") for n in names]
        ).astype(np.float32)
        doc_index = np.concatenate(
            [
                pq.read_table(directory / f"{n}.parquet").column("doc_index").to_numpy()
                for n in names
            ]
        )
        doc_ids = json.loads((directory / "doc_ids.json").read_text())
        if len(vectors) != len(doc_index) or len(doc_ids) != manifest["documents"]:
            raise ValueError("Embedding shards are inconsistent")
        faiss.normalize_L2(vectors)  # Undo fp16 rounding of vector norms.
        return cls(vectors, doc_index, doc_ids, model, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR / "documents.parquet")
    parser.add_argument("--output", type=Path, default=CORPUS_DIR / "indexes" / "dense")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fp32", action="store_true", help="Disable fp16 on GPU")
    parser.add_argument("--shard-docs", type=int, default=20000)
    args = parser.parse_args()
    model = BGEModel(
        args.device, fp16=False if args.fp32 else None, batch_size=args.batch_size
    )
    started = time.perf_counter()
    manifest = embed_corpus(args.corpus, args.output, model, args.shard_docs)
    print(
        json.dumps(
            {**manifest, "seconds": round(time.perf_counter() - started)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
