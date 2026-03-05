import csv
import time
from pathlib import Path
from src.rag_pipeline import run_rag, RagConfig

def main():
    root_dir = Path(__file__).parent.parent
    input_file = root_dir / "output" / "qa_pairs.csv"
    output_file = root_dir / "output" / "qa_results.csv"

    if not input_file.exists():
        print(f"Error: {input_file} not found. Did you run generate_qa_pairs.py?")
        return

    print("Loading existing RAG components (this may take a moment)...")
    cfg = RagConfig(k=3)
    
    # Just to run a dummy query to load models into memory if needed
    # (HuggingFace transformers usually lazy-load, but run_rag doesn't keep them in memory across calls,
    # wait... run_rag initializes models on every call!
    # Let's check `run_rag` definition.
    # Ah, `run_rag` calls load_embeddings, load_metadata_jsonl, then `retrieve` (which initializes `SentenceTransformer` inside it!). 
    # And then `LocalGenerator` which loads the model inside. 
    # Wait, `run_rag` in `src/rag_pipeline.py` is:
    # def run_rag(query, cfg):
    #     embeddings = load_embeddings(cfg.embeddings_path)
    #     metadata = load_metadata_jsonl(cfg.metadata_path)
    #     retrieved_docs = retrieve(query, embeddings, metadata, cfg.embed_model, cfg.k)
    #     gen_cfg = GenConfig(...)
    #     generator = LocalGenerator(gen_cfg)
    #     return ...
    #
    # If we call run_rag in a loop, it re-loads the LLM and tokenizer EACH TIME! That will take forever!
    # Let's write custom logic using the underlying classes so we don't reload weights 20 times.
    from src.retrieve import load_embeddings, load_metadata_jsonl, retrieve
    from src.generate import GenConfig, LocalGenerator
    
    embeddings = load_embeddings(cfg.embeddings_path)
    metadata = load_metadata_jsonl(cfg.metadata_path)
    
    gen_cfg = GenConfig(model_name=cfg.gen_model, device=cfg.gen_device)
    # Load once!
    print("Loading generator LLM...")
    generator = LocalGenerator(gen_cfg)
    print("LLM loaded.")

    with open(input_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        questions = list(reader)

    results = []

    # Metrics
    matched_topic_count = 0
    total_cosine = 0.0
    num_retrieved_total = 0
    
    intra_cosine_total = 0.0
    intra_retrieved_count = 0
    
    extra_cosine_total = 0.0
    extra_retrieved_count = 0

    print("\nStarting evaluation of 20 questions...\n")
    for row in questions:
        q_id = row["question_id"]
        q_text = row["question"]
        expected_topic = row["expected_topic"]
        in_dataset = row["in_dataset"]
        q_type = row["question_type"]

        print(f"Processing {q_id}/q20: {q_text[:60]}...")
        
        # 1. Retrieve
        retrieved_docs = retrieve(
            query=q_text,
            embeddings=embeddings,
            metadata=metadata,
            model_name=cfg.embed_model,
            k=cfg.k,
        )
        
        # 2. Generate
        answer = generator.generate_answer(q_text, retrieved_docs)

        # 3. Store row
        res_row = {
            "question_id": q_id,
            "question": q_text,
            "expected_topic": expected_topic,
            "in_dataset": in_dataset,
            "question_type": q_type,
            "generated_response": answer
        }

        has_target_topic = False
        for i in range(3):
            doc = retrieved_docs[i] if i < len(retrieved_docs) else {}
            title = doc.get("title", "")
            summary = doc.get("summary", "")[:200]
            topic = doc.get("terms", "")
            score = doc.get("score", 0.0)

            res_row[f"retrieved_doc_{i+1}_title"] = title
            res_row[f"retrieved_doc_{i+1}_summary"] = summary
            res_row[f"retrieved_doc_{i+1}_topic"] = topic
            res_row[f"retrieved_doc_{i+1}_cosine_score"] = score
            
            if score > 0:
                total_cosine += score
                num_retrieved_total += 1
                if q_type == "intra":
                    intra_cosine_total += score
                    intra_retrieved_count += 1
                else:
                    extra_cosine_total += score
                    extra_retrieved_count += 1

            if expected_topic in topic:
                has_target_topic = True

        if has_target_topic:
            matched_topic_count += 1

        results.append(res_row)

    # Write CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(output_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"\n✅ Successfully wrote {len(results)} results to {output_file}\n")
    
    # Print Summary
    print("--- EVALUATION SUMMARY ---")
    print(f"Questions with >=1 matching expected topic: {matched_topic_count}/{len(questions)}")
    
    avg_cosine = total_cosine / num_retrieved_total if num_retrieved_total > 0 else 0
    print(f"Average cosine similarity (all 60 retrievals): {avg_cosine:.4f}")
    
    avg_intra = intra_cosine_total / intra_retrieved_count if intra_retrieved_count > 0 else 0
    print(f"Average cosine similarity (Intra-dataset): {avg_intra:.4f}")

    avg_extra = extra_cosine_total / extra_retrieved_count if extra_retrieved_count > 0 else 0
    print(f"Average cosine similarity (Extra-dataset): {avg_extra:.4f}")

if __name__ == "__main__":
    main()
