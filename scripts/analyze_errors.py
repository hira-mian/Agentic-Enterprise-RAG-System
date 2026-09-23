"""Initial error analysis for retrieval runs (numbers quoted in docs/BASELINE_RESULTS.md).

    python scripts/analyze_errors.py results/<run-a> [results/<run-b> ...]

For each run it breaks retrieval down by number of gold articles, measures how
often retrieved articles come from a publisher the query never mentions, and
finds where the first gold article is ranked. With two runs it also reports how
complementary they are. `--chunk-stats` reports how many default-size chunks
exceed the embedding model's 512-token input limit.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from enterprise_rag.data import load_corpus  # noqa: E402

K = 2


def load(run_dir: Path) -> list[dict]:
    with open(run_dir / "predictions.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def topk_docs(row: dict) -> set[str]:
    """Distinct articles among the top-K chunks actually passed to the LLM."""
    return {c["url"] for c in row["retrieved_chunks"][:K]}


def analyze(run_dir: Path, url_source: dict) -> dict:
    rows = [r for r in load(run_dir) if r["gold_urls"]]
    print(f"\n=== {run_dir.name} ({len(rows)} queries with evidence) ===")

    by_n = defaultdict(list)
    for r in rows:
        by_n[len(r["gold_urls"])].append(r)
    print("recall@2 / all@2 / recall@10 by number of gold articles:")
    for n in sorted(by_n):
        rs = by_n[n]
        print(f"  {n} articles (n={len(rs)}): recall@2={mean(r['recall@2'] for r in rs):.3f} "
              f"all@2={mean(r['all@2'] for r in rs):.3f} recall@10={mean(r['recall@10'] for r in rs):.3f}")

    # Top-2 chunks can cover at most 2 distinct articles.
    capped = sum(len(r["gold_urls"]) > K for r in rows)
    ceiling = mean(min(K, len(r["gold_urls"])) / len(r["gold_urls"]) for r in rows)
    print(f"queries needing more than {K} articles: {capped}/{len(rows)} ({capped/len(rows):.1%}); "
          f"max achievable recall@{K} = {ceiling:.3f}")

    # Top-2 chunks from the same article (wasted slot).
    same = sum(len({c['url'] for c in r['retrieved_chunks']}) < len(r['retrieved_chunks']) for r in rows)
    print(f"top-{K} chunks both from one article: {same}/{len(rows)} ({same/len(rows):.1%})")

    # Retrieved articles from a publisher the query does not name.
    top_docs = [(r, u) for r in rows for u in topk_docs(r)]
    off = sum(url_source[u] not in r["query"] for r, u in top_docs)
    print(f"top-{K} retrieved articles whose publisher is not named in the query: {off}/{len(top_docs)} ({off/len(top_docs):.1%})")

    # Where does the first gold article sit when it is missed at k=2?
    missed = [r for r in rows if r["hit@2"] == 0]
    in10 = sum(r["hit@10"] == 1 for r in missed)
    print(f"hit@2 misses: {len(missed)}; of those, a gold article is in the top-10: {in10} ({in10/max(1,len(missed)):.1%})")

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["question_type"]].append(r["recall@2"])
    print("recall@2 by type: " + ", ".join(f"{t}={mean(v):.3f}" for t, v in sorted(by_type.items())))
    return {r["qid"]: r for r in rows}


def chunk_stats() -> None:
    from llama_index.core.schema import MetadataMode
    from tokenizers import Tokenizer

    from enterprise_rag.baselines.rag import build_nodes
    from enterprise_rag.models import LOCAL_MODELS_DIR

    tok = Tokenizer.from_file(str(LOCAL_MODELS_DIR / "fast-bge-base-en-v1.5" / "tokenizer.json"))
    lens = [len(tok.encode(n.get_content(metadata_mode=MetadataMode.EMBED)).ids) for n in build_nodes()]
    over = sum(n > 512 for n in lens)
    unseen = mean(max(0, n - 512) / n for n in lens)
    print(f"\nchunks: {len(lens)}, mean bge tokens (incl. metadata header): {mean(lens):.0f}, "
          f">512 tokens: {over} ({over/len(lens):.1%}), mean share of chunk text truncated: {unseen:.1%}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--chunk-stats", action="store_true")
    args = p.parse_args()

    url_source = {d["url"]: d["source"] for d in load_corpus()}
    results = [analyze(r, url_source) for r in args.runs]
    if len(results) == 2:
        a, b = results
        both = [q for q in a if q in b]
        only_a = sum(a[q]["hit@2"] == 1 and b[q]["hit@2"] == 0 for q in both)
        only_b = sum(b[q]["hit@2"] == 1 and a[q]["hit@2"] == 0 for q in both)
        union = mean(len((topk_docs(a[q]) | topk_docs(b[q])) & set(a[q]["gold_urls"]))
                     / len(a[q]["gold_urls"]) for q in both)
        print(f"\nhit@2 only in {args.runs[0].name}: {only_a}; only in {args.runs[1].name}: {only_b}; "
              f"recall of union of both top-2 lists: {union:.3f}")
    if args.chunk_stats:
        chunk_stats()


if __name__ == "__main__":
    main()
