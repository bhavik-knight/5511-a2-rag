# src/generate.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class GenConfig:
    # Local generator model (small + instruct)
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # "auto" chooses cuda if available, else cpu
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    # Generation settings (conservative for cleaner output)
    max_new_tokens: int = 120
    temperature: float = 0.1
    top_p: float = 0.9

    # Context formatting controls
    max_docs_in_context: int = 5
    max_chars_per_doc: int = 1200


def pick_device(device: str) -> torch.device:
    device = device.lower()
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this machine.")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_prompt(query: str, retrieved_docs: List[Dict[str, Any]], cfg: GenConfig) -> str:
    """
    RAG prompt:
    - Use documents to answer.
    - If insufficient, output EXACT fallback sentence.
    """
    docs = retrieved_docs[: cfg.max_docs_in_context]

    context_blocks: List[str] = []
    for i, d in enumerate(docs, start=1):
        title = (d.get("title") or "").strip()
        summary = (d.get("summary") or "").strip()
        terms = (d.get("terms") or "").strip()

        if len(summary) > cfg.max_chars_per_doc:
            summary = summary[: cfg.max_chars_per_doc] + "..."

        context_blocks.append(
            f"[Doc {i}]\n"
            f"Title: {title}\n"
            f"Terms: {terms}\n"
            f"Content: {summary}\n"
        )

    context = "\n".join(context_blocks).strip()

    prompt = f"""You are a strict and precise assistant for a Retrieval-Augmented Generation (RAG) system.

Task:
- Answer the question using ONLY the provided Documents.
- You MUST NOT rely on your prior knowledge under any circumstances.
- If the Documents provide information on a similar topic but DO NOT directly answer the specific question, or if there is not enough information, you MUST reply with exactly:
I don't know based on the provided documents.
- Do not mention the documents explicitly (do not say "Document 1" or "Document 2"). Just answer.


Documents:
{context}

Question: {query}

Final answer:"""
    return prompt


def clean_generated_text(text: str) -> str:
    """
    Extract just the final answer and remove common over-generation artifacts.
    """
    # Prefer our marker
    if "Final answer:" in text:
        text = text.split("Final answer:", 1)[-1].strip()
    elif "Answer:" in text:
        text = text.split("Answer:", 1)[-1].strip()

    # Stop at common chat continuations
    for stop in ["Human:", "User:", "Assistant:", "\n\nHuman:", "\n\nUser:", "\n\nAssistant:"]:
        if stop in text:
            text = text.split(stop, 1)[0].strip()

    # If model appends fallback after already answering, drop the trailing fallback
    fallback = "I don't know based on the provided documents."
    if fallback in text:
        before, after = text.split(fallback, 1)
        before = before.strip()
        after = after.strip()
        # If there is meaningful content before the fallback, keep it and remove fallback
        if len(before) >= 10:
            text = before

    # Remove repeated blank lines
    text = "\n".join([line.rstrip() for line in text.splitlines()]).strip()
    return text


class LocalGenerator:
    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        self.device = pick_device(cfg.device)

        # Use float16 on GPU; float32 on CPU
        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=torch_dtype)

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def generate_answer(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        prompt = build_prompt(query, retrieved_docs, self.cfg)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.temperature > 0,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return clean_generated_text(decoded)

def main() -> None:
    parser = argparse.ArgumentParser(description="Local generator (requires retrieved docs).")
    parser.add_argument("--query", required=True, help="User question")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    args = parser.parse_args()

    cfg = GenConfig(device=args.device)
    gen = LocalGenerator(cfg)

    print(
        "This module generates answers GIVEN retrieved documents.\n"
        "Run the end-to-end pipeline instead:\n"
        "  uv run python -m src.rag_pipeline --query \"...\"\n"
    )


if __name__ == "__main__":
    main()