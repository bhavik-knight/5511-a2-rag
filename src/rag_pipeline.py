# src/rag_pipeline.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from src.retrieve import load_embeddings, load_metadata_jsonl, retrieve
from src.generate import GenConfig, LocalGenerator


@dataclass(frozen=True)
class RagConfig:
    embeddings_path: Path = Path("knowledge_base/embeddings.npy")
    metadata_path: Path = Path("knowledge_base/metadata.jsonl")

    embed_model: str = "BAAI/bge-small-en"
    k: int = 3

    gen_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    gen_device: str = "auto"


def run_rag(query: str, cfg: RagConfig) -> Dict[str, Any]:
    embeddings = load_embeddings(cfg.embeddings_path)
    metadata = load_metadata_jsonl(cfg.metadata_path)

    retrieved_docs = retrieve(
        query=query,
        embeddings=embeddings,
        metadata=metadata,
        model_name=cfg.embed_model,
        k=cfg.k,
    )

    gen_cfg = GenConfig(model_name=cfg.gen_model, device=cfg.gen_device)
    generator = LocalGenerator(gen_cfg)
    answer = generator.generate_answer(query, retrieved_docs)

    return {
        "query": query,
        "k": cfg.k,
        "retrieved": retrieved_docs,
        "answer": answer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end RAG pipeline (retrieve + generate).")
    parser.add_argument("--query", required=True, help="User question")
    parser.add_argument("--k", type=int, default=3, help="Top-k docs to retrieve")
    parser.add_argument("--embed-model", default="BAAI/bge-small-en", help="Embedding model (must match KB)")
    parser.add_argument("--gen-model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Generator model")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    args = parser.parse_args()

    cfg = RagConfig(
        embed_model=args.embed_model,
        k=args.k,
        gen_model=args.gen_model,
        gen_device=args.device,
    )

    out = run_rag(args.query, cfg)

    print(f"\nQuery: {out['query']}\n")
    print("Retrieved docs:")
    for i, d in enumerate(out["retrieved"], start=1):
        print(f"  #{i} score={d.get('score', 0.0):.4f} terms={d.get('terms','')}")
        print(f"     title: {d.get('title','')}")
    print("\nAnswer:\n")
    print(out["answer"])


if __name__ == "__main__":
    main()