import json
import os
import argparse
import cohere
from pathlib import Path

# System prompt for the LLM judge
JUDGE_PROMPT = """You are an expert evaluator of RAG (Retrieval-Augmented Generation) systems.
Your task is to judge a generated answer based on a user's question and the retrieved documents.

You will evaluate the answer on two criteria:
1. RELEVANCE: Does the generated answer directly address the user's question? (Score: 1 for Yes, 0 for No)
2. FAITHFULNESS (Hallucination Check): Is the generated answer fully supported by the provided retrieved documents? It should NOT contain external information not present in the documents. (Score: 1 for Faithful/Supported, 0 for Hallucinated/Unsupported)

If the generated answer correctly states "I don't know based on the provided documents." because the documents lack the info, it should score 1 for Relevance (it correctly answered the prompt's constraint) and 1 for Faithfulness (it didn't hallucinate).

Provide your evaluation exactly in this JSON format:
{
  "relevance_score": <0 or 1>,
  "faithfulness_score": <0 or 1>,
  "reasoning": "<A brief 1-2 sentence explanation of your scores>"
}
"""

def evaluate_responses(input_file: Path, output_file: Path, model_name: str = "command-r-plus-08-2024"):
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        print("ERROR: COHERE_API_KEY environment variable not found.")
        print("Please add it to your .env file or export it.")
        return

    co = cohere.ClientV2(api_key=api_key)
    
    # Load input data
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
            
    print(f"Loaded {len(records)} records for evaluation using Cohere ({model_name})...")
    
    results = []
    
    for i, record in enumerate(records, 1):
        print(f"[{i}/{len(records)}] Evaluating Question: {record['question']}")
        
        # Format the retrieved docs for the prompt
        docs_text = ""
        for j, doc in enumerate(record['retrieved_docs'], 1):
            docs_text += f"[Document {j}]\nTitle: {doc['title']}\nSummary: {doc['summary']}\n\n"
            
        user_prompt = f"""Question: {record['question']}

Retrieved Documents:
{docs_text}

Generated Answer: {record['generated_answer']}
"""

        try:
            response = co.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            eval_content = response.message.content[0].text
            eval_result = json.loads(eval_content)
            
            # Combine original record with eval results
            record["evaluation"] = eval_result
            results.append(record)
            
        except Exception as e:
            print(f"  Error evaluating record {record['id']}: {e}")
            record["evaluation"] = {"error": str(e)}
            results.append(record)
            
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
    print(f"\nEvaluation complete! Results saved to {output_file}")
    
    # Print a quick summary
    valid_evals = [r for r in results if "relevance_score" in r.get("evaluation", {})]
    if valid_evals:
        avg_rel = sum(r["evaluation"]["relevance_score"] for r in valid_evals) / len(valid_evals)
        avg_faith = sum(r["evaluation"]["faithfulness_score"] for r in valid_evals) / len(valid_evals)
        print(f"\nSummary of {len(valid_evals)} successful evaluations:")
        print(f"  Average Relevance:    {avg_rel*100:.1f}%")
        print(f"  Average Faithfulness: {avg_faith*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG responses using LLM-as-a-judge (Cohere)")
    parser.add_argument("--input", default="data/test_eval_data.jsonl", help="Input JSONL file with Q-A pairs")
    parser.add_argument("--output", default="output/eval_results.jsonl", help="Output JSONL file to save evaluations")
    parser.add_argument("--model", default="command-r-plus-08-2024", help="Cohere model to use as judge")
    
    args = parser.parse_args()
    
    from dotenv import load_dotenv
    load_dotenv() # Load from .env if present
    
    evaluate_responses(Path(args.input), Path(args.output), args.model)
