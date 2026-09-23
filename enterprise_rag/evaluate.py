"""Evaluation harness.

Examples
--------
Retrieval-only (no LLM needed):
    python -m enterprise_rag.evaluate --system llamaindex_default --llm none
    python -m enterprise_rag.evaluate --system bm25 --llm none

End-to-end RAG with an open-source LLM served by Ollama:
    python -m enterprise_rag.evaluate --system llamaindex_default --llm ollama:llama3.1:8b

Answer-prior baselines:
    python -m enterprise_rag.evaluate --system majority_global
    python -m enterprise_rag.evaluate --system majority_per_type

Each run writes results/<run-name>/predictions.jsonl (one row per query) and
results/<run-name>/summary.json (config + aggregate and per-question-type metrics).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .data import REPO_ROOT, load_split
from .metrics import answer_correct, dedupe, exact_match, retrieval_scores, token_f1
from .models import build_embed_model, build_llm, slug

SYSTEMS = ["llamaindex_default", "bm25", "majority_global", "majority_per_type"]
RESULTS_DIR = REPO_ROOT / "results"


def build_system(args):
    if args.system.startswith("majority_"):
        from .baselines.majority import MajorityBaseline

        return MajorityBaseline(load_split("dev"), mode=args.system.removeprefix("majority_"))
    llm = build_llm(args.llm)
    if args.system == "bm25":
        from .baselines.rag import BM25RAG

        return BM25RAG(llm, top_k=args.top_k)
    from .baselines.rag import LlamaIndexDefaultRAG

    return LlamaIndexDefaultRAG(
        build_embed_model(args.embed), args.embed, llm, top_k=args.top_k, rebuild=args.rebuild_index
    )


def run_name(args) -> str:
    if args.name:
        return args.name
    parts = [args.system, args.split]
    if args.system == "llamaindex_default":
        parts.append(slug(args.embed))
    if not args.system.startswith("majority_"):
        parts.append("retrieval-only" if args.llm == "none" else slug(args.llm))
    if args.limit:
        parts.append(f"n{args.limit}")
    return "__".join(parts)


def evaluate_one(system, q, eval_ks: list[int]) -> dict:
    row = {
        "qid": q.qid,
        "question_type": q.question_type,
        "query": q.query,
        "gold_answer": q.answer,
        "gold_urls": q.evidence_urls,
    }
    t0 = time.perf_counter()
    nodes = system.retrieve(q.query, max(eval_ks)) if system.top_k else []
    t1 = time.perf_counter()
    prediction = system.answer(q.query, nodes, q.question_type)
    t2 = time.perf_counter()

    chunk_urls = [n.node.metadata.get("url", "") for n in nodes]
    row["retrieved_urls"] = dedupe(chunk_urls)
    row["retrieved_chunks"] = [
        {"url": n.node.metadata.get("url", ""), "score": n.score, "node_id": n.node.node_id}
        for n in nodes[: system.top_k]
    ]
    if system.top_k and q.evidence_urls:
        for k in eval_ks:
            row.update(retrieval_scores(chunk_urls, q.evidence_urls, k))
    row["prediction"] = prediction
    if prediction is not None:
        row["accuracy"] = float(answer_correct(prediction, q.answer, q.question_type))
        row["exact_match"] = float(exact_match(prediction, q.answer))
        row["token_f1"] = token_f1(prediction, q.answer)
    row["latency_retrieval_s"] = t1 - t0
    row["latency_generation_s"] = t2 - t1
    return row


METRIC_PREFIXES = ("hit@", "recall@", "all@", "mrr@", "accuracy", "exact_match", "token_f1", "latency_")


def aggregate(rows: list[dict]) -> dict:
    keys = sorted({k for r in rows for k in r if k.startswith(METRIC_PREFIXES)})
    out = {"n": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if vals:
            out[k] = round(mean(vals), 4)
            if not k.startswith("latency_") and len(vals) != len(rows):
                out[f"n_{k}"] = len(vals)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", choices=SYSTEMS, required=True)
    p.add_argument("--split", default="test", choices=["dev", "test", "all"])
    p.add_argument("--llm", default="none", help="none | ollama:<model> | openai:<model> | mock")
    p.add_argument("--embed", default="fastembed:BAAI/bge-base-en-v1.5", help="fastembed:<model> | openai:<model> | mock")
    p.add_argument("--top-k", type=int, default=2, help="chunks passed to the LLM (LlamaIndex default: 2)")
    p.add_argument("--eval-ks", type=int, nargs="+", default=[2, 5, 10], help="depths for retrieval metrics")
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N queries of the split")
    p.add_argument("--name", default="", help="override the results directory name")
    p.add_argument("--rebuild-index", action="store_true")
    p.add_argument("--resume", action="store_true", help="skip queries already in predictions.jsonl")
    args = p.parse_args(argv)
    args.eval_ks = sorted(set(args.eval_ks) | {args.top_k})

    queries = load_split(args.split)
    if args.limit:
        queries = queries[: args.limit]
    out_dir = RESULTS_DIR / run_name(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / "predictions.jsonl"

    rows = []
    if args.resume and pred_file.exists():
        rows = [json.loads(line) for line in pred_file.open(encoding="utf-8")]
    done = {r["qid"] for r in rows}

    system = build_system(args)
    print(f"Evaluating {system.name} on {len(queries)} {args.split} queries -> {out_dir}")
    with open(pred_file, "a" if args.resume else "w", encoding="utf-8") as f:
        for i, q in enumerate(queries, 1):
            if q.qid in done:
                continue
            row = evaluate_one(system, q, args.eval_ks)
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0 or i == len(queries):
                print(f"  {i}/{len(queries)}")

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["question_type"]].append(r)
    summary = {
        "run": out_dir.name,
        "split": args.split,
        "config": {**system.config(), "eval_ks": args.eval_ks, "llm_spec": args.llm,
                   "embed_spec": args.embed if args.system == "llamaindex_default" else None},
        "overall": aggregate(rows),
        "by_question_type": {t: aggregate(rs) for t, rs in sorted(by_type.items())},
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
