import json
from src.rag_pipeline import run_rag, RagConfig

# A test batch of intra (in dataset) and extra (not in dataset) questions
TEST_QUESTIONS = [
    {"question": "What are the primary advantages of masked autoencoders for self-supervised visual representation learning?", "type": "intra"},
    {"question": "How do recent multi-scale feature pyramids improve the detection of small objects in aerial imagery?", "type": "intra"},
    {"question": "What techniques are most effective for mitigating semantic shift in unsupervised domain adaptation for image segmentation?", "type": "intra"},
    {"question": "Under what conditions does the Adam optimizer exhibit poor generalization compared to standard stochastic gradient descent?", "type": "intra"},
    {"question": "How can contrastive learning objectives be adapted to handle long-tailed data distributions effectively?", "type": "intra"},
    {"question": "What role does over-parameterization play in avoiding local minima in deep neural network optimization?", "type": "intra"},
    {"question": "How do neuro-symbolic systems integrate symbolic reasoning with neural representations for complex planning tasks?", "type": "intra"},
    {"question": "What are the fundamental limitations of using offline reinforcement learning for dynamic knowledge representation?", "type": "intra"},
    {"question": "What approaches exist for removing motion artifacts in magnetic resonance imaging without relying on paired training data?", "type": "intra"},
    {"question": "How can generative adversarial networks be constrained to preserve anatomical topology in medical image translation?", "type": "intra"},
    {"question": "How does rotary position embedding differ from absolute position embedding in the context of transformer-based language models?", "type": "intra"},
    {"question": "What strategies can be employed to reduce the computational complexity of self-attention mechanisms for very long document classification?", "type": "intra"},
    {"question": "How do topological qubits provide inherent protection against decoherence compared to superconducting qubits?", "type": "extra"},
    {"question": "What constraints limit the practical implementation of Shor's algorithm on current noisy intermediate-scale quantum devices?", "type": "extra"},
    {"question": "Which parameters contribute most significantly to the uncertainty in global coupled ocean-atmosphere models predicting multidecadal sea-level rise?", "type": "extra"},
    {"question": "How do variations in stratospheric aerosol optical depth influence tropospheric circulation patterns during El Niño events?", "type": "extra"},
    {"question": "How does the incorporation of bounded rationality into dynamic stochastic general equilibrium models affect inflation forecasting?", "type": "extra"},
    {"question": "What empirical methods are most robust for identifying causal effects of monetary policy shocks on long-term structural unemployment?", "type": "extra"},
    {"question": "What computational tools are most reliable for calling structural variants from long-read sequencing data in highly repetitive genomic regions?", "type": "extra"},
    {"question": "By what molecular mechanisms do RNA interference pathways regulate post-transcriptional gene silencing in higher eukaryotes?", "type": "extra"}
]

def generate_test_data(output_file="data/test_eval_data.jsonl"):
    print(f"Generating test data for {len(TEST_QUESTIONS)} questions...")
    cfg = RagConfig(k=5)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, item in enumerate(TEST_QUESTIONS, 1):
            query = item["question"]
            q_type = item["type"]
            print(f"[{i}/{len(TEST_QUESTIONS)}] Processing ({q_type}): {query}")
            
            # Run the existing RAG pipeline
            result = run_rag(query, cfg)
            
            # Format the output just like Member C's expected output
            record = {
                "id": f"q_{i}",
                "question": query,
                "type": q_type,
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
