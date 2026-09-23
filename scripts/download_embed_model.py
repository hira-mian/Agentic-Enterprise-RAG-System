"""Fetch a FastEmbed ONNX model from FastEmbed's public GCS mirror into models/.

Only needed when the Hugging Face Hub is unreachable (e.g. behind a restrictive
proxy); otherwise FastEmbed downloads the model automatically on first use.
"""

import argparse
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from enterprise_rag.models import LOCAL_MODELS_DIR  # noqa: E402

MIRRORS = {
    "BAAI/bge-base-en-v1.5": "https://storage.googleapis.com/qdrant-fastembed/fast-bge-base-en-v1.5.tar.gz",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5", choices=sorted(MIRRORS))
    args = parser.parse_args()

    target = LOCAL_MODELS_DIR / f"fast-{args.model.split('/')[-1]}"
    if target.exists():
        print(f"[ok] {target} already present")
        return
    LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[download] {MIRRORS[args.model]}")
    with urllib.request.urlopen(MIRRORS[args.model]) as resp:
        data = resp.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(LOCAL_MODELS_DIR, filter="data")
    print(f"[ok] extracted to {target}")


if __name__ == "__main__":
    main()
