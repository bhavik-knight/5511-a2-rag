# 5511-a2-rag

MCDA5511 - Deep Learning / NLP Course - RAG Assignment with the goal to understand "model criticism".

## Setup and Installation

This project uses `uv` for dependency management. If you don't have it installed, follow the instructions at [astral.sh/uv](https://astral.sh/uv).

To set up the environment and install dependencies, run:

```bash
uv sync
```

This will create a virtual environment and install all required packages from `pyproject.toml` and `uv.lock`.

## Environment Variables

Copy the `.env.example` file to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Ensure you provide keys for the models you intend to use (e.g., Jina, Hugging Face, OpenAI).

## RAG Components

### Models
- **Embeddings:** 
    - Default: `BAAI/bge-small-en` via `sentence-transformers`.
    - Alternatives: `jina-embeddings-v4`, `text-embedding-3-small` (OpenAI).
- **Generation (LLM):**
    - Default: `Qwen 1.5 0.5B` or `Qwen 2 1.5B`.
    - Alternatives: Larger models like `Llama 3`, `Mistral-7B`, or `GPT-4o` (requires API keys).

### Vector Store
- **Chroma DB:** Used as the primary vector database for storing and retrieving document embeddings.
- **LangChain/LangGraph:** Leveraged for building the RAG pipeline and potentially complex agentic workflows for evaluation.

## Usage

Run the Jupyter notebook `src/rag.ipynb` to explore the data, build the RAG system, and perform manual and automated evaluations.

## Member D - Automated Evaluation Analysis (RAG Critic)

As part of the model criticism phase, we ran an automated LLM-as-a-judge evaluation using `evaluate.py`. The evaluation dataset consists of 20 questions: 12 "intra" questions (topics present in the source dataset) and 8 "extra" questions (topics completely outside the dataset). 

The judge (using Cohere's Command R+) scores the generated outputs on **Relevance** and **Faithfulness** (scale of 0-1). It was prompted to reward the generator if it correctly states "I don't know" when the retrieved context lacks the answer.

### Initial Results (k=3)
* Overall (20 questions) - **Relevance:** 70.0% | **Faithfulness:** 65.0%
* Intra Questions (12) - **Relevance:** 66.7% | **Faithfulness:** 58.3%
* Extra Questions (8) - **Relevance:** 75.0% | **Faithfulness:** 75.0%

### Follow-up Experiment (k=5, stricter prompt, added metrics)
To improve performance, we increased the retrieval depth (`k=5`), enforced a stricter generator prompt to prevent hallucinations, and added **Completeness** and **Fluency** metrics to the LLM judge.
* Overall (20 questions) - **Relevance:** 55.0% | **Faithfulness:** 50.0% | **Completeness:** 55.0% | **Fluency:** 80.0%
* Intra Questions (12) - **Relevance:** 50.0% | **Faithfulness:** 41.7% | **Completeness:** 50.0% | **Fluency:** 75.0%
* Extra Questions (8) - **Relevance:** 62.5% | **Faithfulness:** 62.5% | **Completeness:** 62.5% | **Fluency:** 87.5%

### Analysis & Key Findings
1. **The Generator Excels at "I don't know":** The pipeline initially scored higher on out-of-dataset (*extra*) questions because the LLM reliably realized the retrieved documents were irrelevant and correctly refused to answer. 
2. **Context Overload (The k=5 Degradation):** Surprisingly, increasing `k` to 5 *degraded* the scores. By feeding up to 5 full document summaries into a small generator model (`Qwen2.5-0.5B-Instruct`), the model suffered from "context overload" and got lost in the noise of academic abstracts, causing it to fail on questions it previously answered correctly.
3. **Strict Prompts are Double-Edged:** The stricter prompt made the generator overly cautious. It became more defensive and occasionally refused to answer valid *intra* questions even when the correct context was provided among the 5 documents.

**Conclusion:** Small models struggle with too much context. To get higher *intra* scores, the best path forward is upgrading the **Embedding Model** (so that `k=3` finds better matches) rather than forcing more context into a small generator.
