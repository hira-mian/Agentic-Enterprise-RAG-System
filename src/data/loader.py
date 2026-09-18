"""Load and inspect EnterpriseRAG-Bench documents and questions."""

from pathlib import Path

from datasets import load_dataset


DATASET_ID = "onyx-dot-app/EnterpriseRAG-Bench"
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "huggingface"


def load_enterprise_rag_bench(*, streaming: bool = True):
    """Return both test subsets; stream by default to avoid a full download."""
    return {
        name: load_dataset(
            DATASET_ID,
            name,
            split="test",
            streaming=streaming,
            cache_dir=str(CACHE_DIR),
        )
        for name in ("documents", "questions")
    }


if __name__ == "__main__":
    for name, rows in load_enterprise_rag_bench().items():
        print(f"\n{name}: {rows}")
        print(f"Schema: {rows.features}")
        sample = next(iter(rows))
        print("Sample (values truncated to 200 characters):")
        for key, value in sample.items():
            print(f"  {key}: {str(value)[:200]}")
