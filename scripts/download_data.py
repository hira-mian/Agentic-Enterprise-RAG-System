"""Download the MultiHop-RAG corpus and queries into data/raw/ and verify checksums.

The files are pinned to a specific commit of the upstream repository
(github.com/yixuantt/MultiHop-RAG, ODC-BY licence) so every run of the
harness sees byte-identical data.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from enterprise_rag.data import RAW_DIR  # noqa: E402

UPSTREAM_COMMIT = "89d5b19"
BASE_URL = f"https://media.githubusercontent.com/media/yixuantt/MultiHop-RAG/{UPSTREAM_COMMIT}/dataset"
FILES = {
    "corpus.json": "20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f",
    "MultiHopRAG.json": "03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        dest = RAW_DIR / name
        if dest.exists() and sha256(dest) == expected:
            print(f"[ok] {dest} already present")
            continue
        url = f"{BASE_URL}/{name}"
        print(f"[download] {url}")
        urllib.request.urlretrieve(url, dest)
        actual = sha256(dest)
        if actual != expected:
            raise SystemExit(f"Checksum mismatch for {name}: {actual} != {expected}")
        print(f"[ok] {dest} ({dest.stat().st_size:,} bytes, sha256 verified)")


if __name__ == "__main__":
    main()
