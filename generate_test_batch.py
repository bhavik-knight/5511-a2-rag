import json
from src.rag_pipeline import run_rag, RagConfig

# A test batch of questions based on common ML/NLP topics that should be in the ArXiv dataset
TEST_QUESTIONS = [
    "What is attention in transformers?",
    "How does BERT differ from GPT?",
    "What are the advantages of using graph neural networks?",
    "Explain the concept of self-supervised learning.",
    "What is the role of dropout in neural networks?",
    "How is reinforcement learning used in robotics?",
    "What are word embeddings and how are they created?",
    "What is the vanishing gradient problem in RNNs?",
    "How do convolutional neural networks work on images?",
    "What is contrastive learning?",
    "Explain the architecture of a Variational Autoencoder (VAE)",
    "What is few-shot learning in NLP?",
    "How does a diffusion model generate images?",
    "What is the purpose of position encodings in transformers?",
    "What are the main applications of sequence-to-sequence models?"
]

def generate_test_data(output_file="data/test_eval_data.jsonl"):
    print(f"Generating test data for {len(TEST_QUESTIONS)} questions...")
    cfg = RagConfig()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, query in enumerate(TEST_QUESTIONS, 1):
            print(f"[{i}/{len(TEST_QUESTIONS)}] Processing: {query}")
            
            # Run the existing RAG pipeline
            result = run_rag(query, cfg)
            
            # Format the output just like Member C's expected output
            record = {
                "id": f"q_{i}",
                "question": query,
                "retrieved_docs": [
                    {
                        "title": doc.get('title', ''),
                        "summary": doc.get('summary', ''),
                        "score": doc.get('score', 0.0)
                    } 
                    for doc in result['retrieved']
                ],
                "generated_answer": result['answer']
            }
            
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print(f"\nDone! Test data saved to {output_file}")

if __name__ == "__main__":
    generate_test_data()
