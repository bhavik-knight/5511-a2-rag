import json
from src.rag_pipeline import run_rag, RagConfig

# A test batch of intra (in dataset), related-missing, and extra (not in dataset) questions
TEST_QUESTIONS = [
    {"qid": 1, "question": "What are the primary advantages of masked autoencoders for self-supervised visual representation learning?", "type": "intra", "expected_topic": "cs.CV"},
    {"qid": 2, "question": "How do recent multi-scale feature pyramids improve the detection of small objects in aerial imagery?", "type": "intra", "expected_topic": "cs.CV"},
    {"qid": 3, "question": "What techniques are most effective for mitigating semantic shift in unsupervised domain adaptation for image segmentation?", "type": "intra", "expected_topic": "cs.CV"},
    {"qid": 4, "question": "Under what conditions does the Adam optimizer exhibit poor generalization compared to standard stochastic gradient descent?", "type": "intra", "expected_topic": "cs.LG"},
    {"qid": 5, "question": "How can contrastive learning objectives be adapted to handle long-tailed data distributions effectively?", "type": "intra", "expected_topic": "cs.LG"},
    {"qid": 6, "question": "What role does over-parameterization play in avoiding local minima in deep neural network optimization?", "type": "intra", "expected_topic": "stat.ML"},
    {"qid": 7, "question": "How do neuro-symbolic systems integrate symbolic reasoning with neural representations for complex planning tasks?", "type": "intra", "expected_topic": "cs.AI"},
    {"qid": 8, "question": "What are the fundamental limitations of using offline reinforcement learning for dynamic knowledge representation?", "type": "intra", "expected_topic": "cs.AI"},
    {"qid": 9, "question": "What approaches exist for removing motion artifacts in magnetic resonance imaging without relying on paired training data?", "type": "intra", "expected_topic": "eess.IV"},
    {"qid": 10, "question": "How can generative adversarial networks be constrained to preserve anatomical topology in medical image translation?", "type": "intra", "expected_topic": "eess.IV"},
    {"qid": 11, "question": "How does rotary position embedding differ from absolute position embedding in the context of transformer-based language models?", "type": "intra", "expected_topic": "cs.CL"},
    {"qid": 12, "question": "What strategies can be employed to reduce the computational complexity of self-attention mechanisms for very long document classification?", "type": "intra", "expected_topic": "cs.CL"},
    {"qid": 13, "question": "What are the theoretical guarantees on sample complexity for quantum kernel methods applied to classical supervised learning tasks?", "type": "related-missing", "expected_topic": "quant-ph"},
    {"qid": 14, "question": "How do spike-timing-dependent plasticity rules in neuromorphic hardware compare to backpropagation in terms of energy efficiency and learning stability?", "type": "related-missing", "expected_topic": "cs.NE"},
    {"qid": 15, "question": "What are the formal differential privacy guarantees of the Gaussian mechanism when applied to gradient aggregation in cross-silo federated learning with heterogeneous data distributions?", "type": "related-missing", "expected_topic": "cs.CR"},
    {"qid": 16, "question": "What were the primary economic consequences of the Black Death on feudal land tenure systems in 14th century England?", "type": "extra", "expected_topic": "history"},
    {"qid": 17, "question": "How does the doctrine of proportionality differ from rational basis review in constitutional adjudication of fundamental rights?", "type": "extra", "expected_topic": "law"},
    {"qid": 18, "question": "By what physiological mechanisms do deep-sea fish regulate buoyancy at extreme hydrostatic pressures below 1000 metres?", "type": "extra", "expected_topic": "biology"},
    {"qid": 19, "question": "What mechanistic pathways govern the stereoselectivity of asymmetric aldol reactions catalysed by proline derivatives?", "type": "extra", "expected_topic": "chemistry"},
    {"qid": 20, "question": "How did the structural principles of Roman concrete (opus caementicium) enable the construction of the Pantheon's unreinforced dome?", "type": "extra", "expected_topic": "architecture"}
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
