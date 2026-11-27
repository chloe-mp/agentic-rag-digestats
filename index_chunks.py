"""
Indexe chunks.jsonl (build_chunks.py) dans une NOUVELLE collection Qdrant.

Embeddings BGE-M3 : à lancer sur GPU (hf_job.sh index <collection>), pas sur le Mac.

    python index_chunks.py --collection reglementation_digestats_v2
"""

import argparse
from pathlib import Path

from indexing_pipeline import index_chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--chunks", default=str(Path(__file__).parent / "chunks.jsonl"))
    args = parser.parse_args()
    n = index_chunks(Path(args.chunks), args.collection)
    print(f"[OK] {n} chunks indexés dans '{args.collection}'")
