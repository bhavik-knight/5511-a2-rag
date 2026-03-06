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

### Results
* Overall (20 questions):
  * **Average Relevance:** 70.0%
  * **Average Faithfulness:** 65.0%

* Intra Questions (12 questions):
  * **Average Relevance:** 66.7%
  * **Average Faithfulness:** 58.3%

* Extra Questions (8 questions):
  * **Average Relevance:** 75.0%
  * **Average Faithfulness:** 75.0%

### Analysis & Key Findings
1. **The Generator Excels at "I don't know":** The pipeline scored higher on out-of-dataset (*extra*) questions. High scores were achieved because the LLM reliably realized the retrieved documents were irrelevant and correctly refused to answer based on the prompt constraints. 
2. **Retrieval Bottlenecks:** For *intra* questions, the retriever sometimes failed to fetch the exact papers needed, lowering overall success.
3. **Subtle Hallucinations:** When retrieved documents shared keywords with the prompt but lacked the actual answer, the generator occasionally forced a response by blending the question's concepts with the irrelevant text, leading to 0s for Faithfulness.

**Next Steps:** We recommend tightening the generation prompt to make it stricter when evaluating "similar-but-irrelevant" concepts, as well as refining our retrieval strategy to improve the accuracy of finding in-domain *intra* documents.
