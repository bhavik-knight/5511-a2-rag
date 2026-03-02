# src/embed.py
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class EmbedConfig:
    # Your chosen dataset name
    data_csv: Path = Path("data/arxiv_data.csv")

    # Where we store generated artifacts (do NOT commit these)
    kb_dir: Path = Path("knowledge_base")
    embeddings_path: Path = Path("knowledge_base/embeddings.npy")
    metadata_path: Path = Path("knowledge_base/metadata.jsonl")

    # Assignment-suggested embedding model
    model_name: str = "BAAI/bge-small-en"

    # Performance knobs
    batch_size: int = 128
    normalize: bool = True  # recommended for cosine similarity
    max_rows: Optional[int] = None  # set for quick smoke tests


REQUIRED_COLUMNS = {"titles", "summaries", "terms"}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_csv(csv_path: Path, max_rows: Optional[int] = None) -> List[Dict]:
    """
    Reads CSV with columns: titles, summaries, terms
    Returns list of dicts with keys: id, title, summary, terms, doc_text (+ lengths).
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found at: {csv_path}\n"
            f"Put your dataset at data/arxiv_data.csv or pass --data-csv."
        )

    docs: List[Dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - cols
        if missing:
            raise ValueError(
                f"CSV missing columns: {sorted(missing)}\n"
                f"Found columns: {reader.fieldnames}"
            )

        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break

            title = (row.get("titles") or "").strip()
            summary = (row.get("summaries") or "").strip()
            terms = (row.get("terms") or "").strip()

            # Text we embed (title improves retrieval)
            doc_text = f"{title}. {summary}".strip()

            docs.append(
                {
                    "id": i,
                    "title": title,
                    "summary": summary,
                    "terms": terms,
                    "doc_text": doc_text,
                    "title_len": len(title),
                    "summary_len": len(summary),
                    "doc_text_len": len(doc_text),
                }
            )

    if not docs:
        raise ValueError("Loaded 0 rows from CSV. Check that the file is not empty.")

    return docs


def embed_documents(
    docs: List[Dict],
    model_name: str,
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    """
    Returns embeddings as float32 numpy array shape (N, d).
    If normalize=True, embeddings are L2-normalized (best for cosine similarity retrieval).
    """
    model = SentenceTransformer(model_name)
    texts = [d["doc_text"] for d in docs]

    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,  # sentence-transformers supports this
    )
    emb = np.asarray(emb, dtype=np.float32)

    # Safety normalization (in case env ignores normalize_embeddings)
    if normalize:
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb = emb / norms

    return emb


def save_metadata_jsonl(docs: List[Dict], path: Path) -> None:
    """
    Writes one JSON object per line (easy to stream later).
    """
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            record = {
                "id": d["id"],
                "title": d["title"],
                "summary": d["summary"],
                "terms": d["terms"],
                "title_len": d["title_len"],
                "summary_len": d["summary_len"],
                "doc_text_len": d["doc_text_len"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_kb(cfg: EmbedConfig) -> Tuple[np.ndarray, List[Dict]]:
    _ensure_dir(cfg.kb_dir)

    docs = _read_csv(cfg.data_csv, max_rows=cfg.max_rows)
    embeddings = embed_documents(
        docs=docs,
        model_name=cfg.model_name,
        batch_size=cfg.batch_size,
        normalize=cfg.normalize,
    )

    np.save(cfg.embeddings_path, embeddings)
    save_metadata_jsonl(docs, cfg.metadata_path)

    return embeddings, docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build embeddings knowledge base for RAG.")
    parser.add_argument("--data-csv", default="data/arxiv_data.csv", help="Path to CSV")
    parser.add_argument("--model", default="BAAI/bge-small-en", help="Embedding model")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization")
    parser.add_argument("--max-rows", type=int, default=None, help="Embed only first N rows (smoke test)")
    args = parser.parse_args()

    cfg = EmbedConfig(
        data_csv=Path(args.data_csv),
        model_name=args.model,
        batch_size=args.batch_size,
        normalize=not args.no_normalize,
        max_rows=args.max_rows,
    )

    embeddings, docs = build_kb(cfg)

    print(f"Done. Embedded {len(docs)} documents.")
    print(f"Embeddings saved to: {cfg.embeddings_path} | shape={embeddings.shape} | dtype={embeddings.dtype}")
    print(f"Metadata saved to:   {cfg.metadata_path}")


if __name__ == "__main__":
    main()