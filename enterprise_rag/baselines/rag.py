"""Retrieval-augmented baselines built on LlamaIndex defaults.

* `llamaindex_default` — the out-of-the-box LlamaIndex pipeline, equivalent to

      index = VectorStoreIndex.from_documents(documents)
      index.as_query_engine().query(question)

  i.e. SentenceSplitter(chunk_size=1024, chunk_overlap=200), dense retrieval
  with similarity_top_k=2, and the default `compact` response synthesizer with
  LlamaIndex's default QA prompt. Only the embedding model and LLM are swapped
  for open-source ones (the LlamaIndex default is OpenAI).

* `bm25` — the minimal retrieval baseline: same chunks, same top_k, same
  synthesizer and LLM, but sparse BM25 retrieval instead of embeddings.

Both expose `retrieve(query, k)` so the harness can also measure retrieval at
depths beyond the default top-2 (diagnostics only; answers always use top_k).
Taking the first `top_k` of a deeper ranked list gives exactly the nodes a
`similarity_top_k=top_k` retriever would return.
"""

from __future__ import annotations

from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import get_response_synthesizer

from ..data import REPO_ROOT, load_documents
from ..models import slug
from .base import System

DEFAULT_CHUNK_SIZE = 1024  # llama_index.core.constants.DEFAULT_CHUNK_SIZE
DEFAULT_CHUNK_OVERLAP = 200  # SentenceSplitter default
DEFAULT_TOP_K = 2  # llama_index.core.constants.DEFAULT_SIMILARITY_TOP_K

STORAGE_DIR = REPO_ROOT / "storage"


def build_nodes():
    splitter = SentenceSplitter(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
    return splitter.get_nodes_from_documents(load_documents())


class _RAGSystem(System):
    def __init__(self, llm, top_k: int = DEFAULT_TOP_K):
        self.llm = llm
        self.top_k = top_k
        self.synthesizer = get_response_synthesizer(llm=llm) if llm is not None else None

    def answer(self, query: str, nodes: list, question_type: str) -> str | None:
        if self.synthesizer is None:
            return None
        return str(self.synthesizer.synthesize(query, nodes=nodes[: self.top_k]))

    def config(self) -> dict:
        return {
            **super().config(),
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
            "response_mode": "compact",
            "llm": getattr(self.llm, "model", None) if self.llm is not None else None,
        }


class LlamaIndexDefaultRAG(_RAGSystem):
    name = "llamaindex_default"

    def __init__(self, embed_model, embed_spec: str, llm, top_k: int = DEFAULT_TOP_K, rebuild: bool = False):
        super().__init__(llm, top_k)
        self.embed_spec = embed_spec
        Settings.embed_model = embed_model
        persist_dir = STORAGE_DIR / f"vector-{slug(embed_spec)}-c{DEFAULT_CHUNK_SIZE}-o{DEFAULT_CHUNK_OVERLAP}"
        if persist_dir.exists() and not rebuild:
            print(f"Loading persisted index from {persist_dir}")
            self.index = load_index_from_storage(StorageContext.from_defaults(persist_dir=str(persist_dir)))
        else:
            print(f"Building vector index with {embed_spec} (one-off; persisted to {persist_dir})")
            self.index = VectorStoreIndex(build_nodes(), show_progress=True)
            self.index.storage_context.persist(persist_dir=str(persist_dir))
        self._retrievers = {}

    def retrieve(self, query: str, k: int) -> list:
        if k not in self._retrievers:
            self._retrievers[k] = self.index.as_retriever(similarity_top_k=k)
        return self._retrievers[k].retrieve(query)

    def config(self) -> dict:
        return {**super().config(), "retriever": "dense", "embed_model": self.embed_spec}


class BM25RAG(_RAGSystem):
    name = "bm25"

    def __init__(self, llm, top_k: int = DEFAULT_TOP_K):
        super().__init__(llm, top_k)
        import Stemmer
        from llama_index.retrievers.bm25 import BM25Retriever

        self._nodes = build_nodes()
        self._stemmer = Stemmer.Stemmer("english")
        self._cls = BM25Retriever
        self._retrievers = {}

    def retrieve(self, query: str, k: int) -> list:
        if k not in self._retrievers:
            self._retrievers[k] = self._cls.from_defaults(
                nodes=self._nodes, similarity_top_k=k, stemmer=self._stemmer, language="english"
            )
        return self._retrievers[k].retrieve(query)

    def config(self) -> dict:
        return {**super().config(), "retriever": "bm25"}
