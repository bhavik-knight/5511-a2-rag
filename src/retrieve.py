# src/retrieve.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class RetrieveConfig:
    kb_dir: Path = Path("knowledge_base")
    embeddings_path: Path = Path("knowledge_base/embeddings.npy")
    metadata_path: Path = Path("knowledge_base/metadata.jsonl")

    model_name: str = "BAAI/bge-small-en"
    k: int = 3


def load_embeddings(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {path}. Run: uv run python -m src.embed"
        )
    emb = np.load(path)
    if emb.ndim != 2:
        raise ValueError(f"Expected embeddings to be 2D array, got shape {emb.shape}")
    # We assume L2-normalized embeddings from embed.py
    return emb.astype(np.float32, copy=False)


def load_metadata_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {path}. Run: uv run python -m src.embed"
        )
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError("Loaded 0 metadata records. Is metadata.jsonl empty?")
    return records


def embed_query(query: str, model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    q = model.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype=np.float32)[0]
    # Safety normalization
    q = q / (np.linalg.norm(q) + 1e-12)
    return q


def topk_cosine(
    query_vec: np.ndarray,
    doc_emb: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Since both query and docs are L2-normalized, cosine similarity = dot product.
    Returns (top_scores, top_indices) sorted descending by score.
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    k = min(k, doc_emb.shape[0])

    scores = doc_emb @ query_vec  # shape (N,)
    # argpartition for speed, then sort the top-k
    top_idx_unsorted = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx_unsorted[np.argsort(scores[top_idx_unsorted])[::-1]]
    top_scores = scores[top_idx]
    return top_scores, top_idx


def retrieve(
    query: str,
    embeddings: np.ndarray,
    metadata: List[Dict[str, Any]],
    model_name: str = "BAAI/bge-small-en",
    k: int = 3,
) -> List[Dict[str, Any]]:
    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"Embeddings rows ({embeddings.shape[0]}) != metadata records ({len(metadata)})."
        )

    qvec = embed_query(query, model_name=model_name)
    scores, idx = topk_cosine(qvec, embeddings, k=k)

    results: List[Dict[str, Any]] = []
    for score, i in zip(scores.tolist(), idx.tolist()):
        rec = dict(metadata[i])  # copy
        rec["score"] = float(score)
        results.append(rec)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve top-k documents for a query.")
    parser.add_argument("--query", required=True, help="User query text")
    parser.add_argument("--k", type=int, default=3, help="Top-k docs to retrieve")
    parser.add_argument("--model", default="BAAI/bge-small-en", help="Embedding model")
    args = parser.parse_args()

    cfg = RetrieveConfig(model_name=args.model, k=args.k)

    emb = load_embeddings(cfg.embeddings_path)
    meta = load_metadata_jsonl(cfg.metadata_path)

    results = retrieve(
        query=args.query,
        embeddings=emb,
        metadata=meta,
        model_name=cfg.model_name,
        k=args.k,
    )

    print(f"\nQuery: {args.query}\n")
    for rank, r in enumerate(results, start=1):
        title = r.get("title", "")
        terms = r.get("terms", "")
        score = r.get("score", 0.0)
        summary = r.get("summary", "")
        print(f"--- #{rank} | score={score:.4f} | terms={terms}")
        print(f"Title: {title}")
        print(f"Summary (first 300 chars): {summary[:300]}{'...' if len(summary) > 300 else ''}")
        print()


if __name__ == "__main__":
    main()