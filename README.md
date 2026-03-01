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
