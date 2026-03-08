# 5511-a2-rag

MCDA 5511 - Deep Learning / NLP Course
**Retrieval-Augmented Generation (RAG) System Evaluation & Model Criticism**

---

## 🚀 Project Overview

This project implements and evaluates a RAG (Retrieval-Augmented Generation) system using a specialized corpus of over **38,000 ArXiv paper abstracts** (Computer Science and Statistics). The primary objective is to analyze the performance of the RAG pipeline across various query types and implement **Model Criticism** via both manual and automated evaluation methods.

The system is tested against 20 key questions:
- **Intra Questions (12):** Topics present within the source abstracts.
- **Extra/Missing Questions (8):** Topics completely outside the dataset or partially related but missing from the local corpus.

## 📁 Project Structure

```text
5511-a2-rag/
├── data/                    # Dataset storage
│   ├── arxiv_data.csv       # Preprocessed ArXiv abstracts
│   ├── qa_pairs.csv         # Target questions and ground truth topics
│   └── test_eval_data.jsonl # Input data for LLM judge
├── output/                  # Analytics and Metrics
│   ├── ir_metrics_summary.txt # Retrieval evaluation metrics (Precision/Recall)
│   ├── manual_audit.csv     # Human-labelled relevance audit results
│   ├── eval_results.jsonl   # Raw outputs from the LLM-as-a-Judge
│   └── *.png                # Topic frequency and length distribution plots
├── src/                     # Source Repository logic
│   ├── rag_pipeline.py      # Core RAG implementation
│   ├── evaluate.py          # LLM judge evaluation script
│   └── run_qa_evaluation.py # Batch processing for evaluation
├── rag.ipynb                # Main implementation & experimentation notebook
├── TODO.md                  # Task tracking and member responsibilities
└── pyproject.toml           # Dependency configuration (managed by uv)
```

## 🛠️ Setup & Installation

This project utilizes `uv` for dependency management.

1. **Install Dependencies**:
   ```bash
   uv sync
   ```

2. **Configure Environment**:
   Create a `.env` file based on the example:
   ```bash
   cp .env.example .env
   ```
   Ensure you provide:
   - `JINA_API_KEY` or `HUGGINGFACE_API_KEY` for embeddings.
   - `COHERE_API_KEY` for the Automated Judge (Command R+).

## 📊 Evaluation & Model Criticism

### 1. Information Retrieval (IR) Metrics
We manually audited the Top-3 retrieved documents for every evaluation question.
- **Recall (0.93):** Excellent recall for intra-dataset queries, indicating that cosine similarity effectively identifies relevant abstracts.
- **Precision (0.45):** Precision is lower overall due to the "Extra" questions, where the retriever is forced to return context that is inherently irrelevant.

### 2. LLM-as-a-Judge Analysis
We utilized **Cohere Command R+** to score the generated answers.

| Metric | k=3 (Initial) | k=5 (Deep Retrieval) |
| :--- | :--- | :--- |
| **Relevance** | 70.0% | 55.0% |
| **Faithfulness** | 65.0% | 50.0% |
| **Fluency** | - | 80.0% |

### 🔍 Key Insights
- **The "k=5" Paradox:** Increasing the number of retrieved chunks (`k`) actually **degraded** the performance of small generator models (e.g., `Qwen2.5-0.5B`). The model suffered from "context overload," getting lost in the technical noise of multiple abstracts.
- **Defensive Refusal:** Stricter prompts improved faithfulness by encouraging the model to say "I don't know" for out-of-dataset queries, but occasionally caused it to refuse valid intra-dataset questions.

## 👥 Team
- **Member A:** Data selection, curation, and distribution statistics.
- **Member B:** Vector DB setup and RAG pipeline engineering.
- **Member C:** Manual audit and IR metrics analysis.
- **Member D:** Automated evaluation and generative criticism.
