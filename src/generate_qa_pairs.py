import csv
from pathlib import Path

def main():
    # Define paths
    root_dir = Path(__file__).parent.parent
    output_dir = root_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "qa_pairs.csv"

    # Define the QA dataset
    questions = [
        # Intra-dataset: cs.CV (Computer Vision)
        {
            "question_id": "q01",
            "question": "What are the primary advantages of masked autoencoders for self-supervised visual representation learning?",
            "expected_topic": "cs.CV",
            "in_dataset": True,
            "question_type": "intra"
        },
        {
            "question_id": "q02",
            "question": "How do recent multi-scale feature pyramids improve the detection of small objects in aerial imagery?",
            "expected_topic": "cs.CV",
            "in_dataset": True,
            "question_type": "intra"
        },
        {
            "question_id": "q03",
            "question": "What techniques are most effective for mitigating semantic shift in unsupervised domain adaptation for image segmentation?",
            "expected_topic": "cs.CV",
            "in_dataset": True,
            "question_type": "intra"
        },
        
        # Intra-dataset: cs.LG / stat.ML (Machine Learning)
        {
            "question_id": "q04",
            "question": "Under what conditions does the Adam optimizer exhibit poor generalization compared to standard stochastic gradient descent?",
            "expected_topic": "cs.LG",
            "in_dataset": True,
            "question_type": "intra"
        },
        {
            "question_id": "q05",
            "question": "How can contrastive learning objectives be adapted to handle long-tailed data distributions effectively?",
            "expected_topic": "cs.LG",
            "in_dataset": True,
            "question_type": "intra"
        },
        {
            "question_id": "q06",
            "question": "What role does over-parameterization play in avoiding local minima in deep neural network optimization?",
            "expected_topic": "stat.ML",
            "in_dataset": True,
            "question_type": "intra"
        },
        
        # Intra-dataset: cs.AI (Artificial Intelligence)
        {
            "question_id": "q07",
            "question": "How do neuro-symbolic systems integrate symbolic reasoning with neural representations for complex planning tasks?",
            "expected_topic": "cs.AI",
            "in_dataset": True,
            "question_type": "intra"
        },
        {
            "question_id": "q08",
            "question": "What are the fundamental limitations of using offline reinforcement learning for dynamic knowledge representation?",
            "expected_topic": "cs.AI",
            "in_dataset": True,
            "question_type": "intra"
        },
        
        # Intra-dataset: eess.IV (Image Processing)
        {
            "question_id": "q09",
            "question": "What approaches exist for removing motion artifacts in magnetic resonance imaging without relying on paired training data?",
            "expected_topic": "eess.IV",
            "in_dataset": True,
            "question_type": "intra"
        },
        {
            "question_id": "q10",
            "question": "How can generative adversarial networks be constrained to preserve anatomical topology in medical image translation?",
            "expected_topic": "eess.IV",
            "in_dataset": True,
            "question_type": "intra"
        },
        
        # Intra-dataset: cs.CL (Natural Language Processing)
        {
            "question_id": "q11",
            "question": "How does rotary position embedding differ from absolute position embedding in the context of transformer-based language models?",
            "expected_topic": "cs.CL",
            "in_dataset": True,
            "question_type": "intra"
        },
        {
            "question_id": "q12",
            "question": "What strategies can be employed to reduce the computational complexity of self-attention mechanisms for very long document classification?",
            "expected_topic": "cs.CL",
            "in_dataset": True,
            "question_type": "intra"
        },

        # Extra-dataset: Quantum Computing
        {
            "question_id": "q13",
            "question": "How do topological qubits provide inherent protection against decoherence compared to superconducting qubits?",
            "expected_topic": "quant-ph",
            "in_dataset": False,
            "question_type": "extra"
        },
        {
            "question_id": "q14",
            "question": "What constraints limit the practical implementation of Shor's algorithm on current noisy intermediate-scale quantum devices?",
            "expected_topic": "quant-ph",
            "in_dataset": False,
            "question_type": "extra"
        },

        # Extra-dataset: Climate Science
        {
            "question_id": "q15",
            "question": "Which parameters contribute most significantly to the uncertainty in global coupled ocean-atmosphere models predicting multidecadal sea-level rise?",
            "expected_topic": "physics.ao-ph",
            "in_dataset": False,
            "question_type": "extra"
        },
        {
            "question_id": "q16",
            "question": "How do variations in stratospheric aerosol optical depth influence tropospheric circulation patterns during El Niño events?",
            "expected_topic": "physics.ao-ph",
            "in_dataset": False,
            "question_type": "extra"
        },

        # Extra-dataset: Economics / Finance
        {
            "question_id": "q17",
            "question": "How does the incorporation of bounded rationality into dynamic stochastic general equilibrium models affect inflation forecasting?",
            "expected_topic": "econ.TH",
            "in_dataset": False,
            "question_type": "extra"
        },
        {
            "question_id": "q18",
            "question": "What empirical methods are most robust for identifying causal effects of monetary policy shocks on long-term structural unemployment?",
            "expected_topic": "econ.EM",
            "in_dataset": False,
            "question_type": "extra"
        },

        # Extra-dataset: Biology / Genomics
        {
            "question_id": "q19",
            "question": "What computational tools are most reliable for calling structural variants from long-read sequencing data in highly repetitive genomic regions?",
            "expected_topic": "q-bio.GN",
            "in_dataset": False,
            "question_type": "extra"
        },
        {
            "question_id": "q20",
            "question": "By what molecular mechanisms do RNA interference pathways regulate post-transcriptional gene silencing in higher eukaryotes?",
            "expected_topic": "q-bio.MN",
            "in_dataset": False,
            "question_type": "extra"
        }
    ]

    # Write to CSV
    fieldnames = ["question_id", "question", "expected_topic", "in_dataset", "question_type"]
    
    with open(out_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(questions)

    print(f"✅ Successfully wrote {len(questions)} Q-A pairs to {out_file}\n")
    
    # Print summary table
    print(f"{'ID':<5}| {'Type':<6}| {'In Data':<7}| {'Topic':<15}| {'Question'}")
    print("-" * 120)
    for q in questions:
        q_text = q['question'][:70] + "..." if len(q['question']) > 70 else q['question']
        print(f"{q['question_id']:<5}| {q['question_type']:<6}| {str(q['in_dataset']):<7}| {q['expected_topic']:<15}| {q_text}")

if __name__ == "__main__":
    main()
