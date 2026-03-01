# Assignment 2: RAG System TODO List

This document outlines the tasks for each group member to ensure equal contribution and coverage of all requirements for Assignment 2.

## Member A: Data Architect & Explorer
**Role:** Responsible for Requirement (1): Data selection, sampling, and generating summary statistics.

- [ ] **Data Selection & Curation:** Select and sample the dataset from the ArXiv abstracts (Kaggle). [Requirement (1)]
- [ ] **Processing Documentation:** Document any sampling or processing steps taken. [Requirement (1.1)]
- [ ] **Summary Statistics:** Generate distribution of document lengths and vocabulary size. [Requirement (1.2)]
- [ ] **Topic Identification:** Identify the topics covered in the dataset. [Requirement (1.3)]
- [ ] **Frequency Analysis:** Calculate the relevant frequencies of each topic. [Requirement (1.4)]
- [ ] **Final Summary:** Write a clear data exploration summary in Markdown within `rag.ipynb`. [Requirement (1)]

---

## Member B: RAG Pipeline Engineer
**Role:** Responsible for Requirement (2): Implementing the retrieval and generation components.

- [ ] **Embedding Configuration:** Set up `sentence-transformers` using `BAAI/bge-small-en` (or chosen model). [Requirement (2)]
- [ ] **Vector Database Setup:** Initialize and configure the vector store (e.g., Chroma). [Requirement (2)]
- [ ] **Retrieval Implementation:** Implement top-k retrieval (k=3+) using cosine similarity. [Requirement (2)]
- [ ] **LLM Integration:** Integrate a generative model (e.g., Qwen 1.5/2 or larger) for answer generation. [Requirement (2)]
- [ ] **Prompt Engineering:** Experiment with and document the prompt template used. [Requirement (2)]
- [ ] **Pipeline Testing:** Ensure the end-to-end RAG flow works correctly in `rag.ipynb`. [Requirement (2)]

---

## Member C: Quality Assurance & Manual Evaluation
**Role:** Responsible for Requirements (3) & (4): Dataset construction and manual retrieval audit.

- [ ] **Q-A Pair Generation:** Create 15+ Q-A pairs covering intra-dataset and extra-dataset topics. [Requirement (3)]
- [ ] **Manual Data Storage:** For each question, store retrieved docs, topic labels, and generated responses. [Requirement (3)]
- [ ] **Manual Retrieval Audit:** Label each retrieved document as correct or incorrect. [Requirement (4)]
- [ ] **IR Metrics Calculation:** Calculate Precision, Recall, F1-score, and Accuracy for retrieval. [Requirement (4)]
- [ ] **Cosine Similarity Explanation:** Write an explanation of cosine similarity, its use, and its limitations. [Requirement (4)]
- [ ] **Failure Analysis:** Identify and document instances where retrieval failed due to similarity limitations. [Requirement (4)]

---

## Member D: Generative Criticism & Automation (LLM-as-a-Judge)
**Role:** Responsible for Requirements (5) & (6): Response evaluation and automated testing pipeline.

- [ ] **Hallucination Check:** Manually review responses for hallucinations against retrieved docs. [Requirement (5)]
- [ ] **Quality Assessment:** Comment on the overall quality of generated responses. [Requirement (5)]
- [ ] **Automated Evaluation Pipeline:** Set up an "LLM as a judge" to redo the evaluations (3-5). [Requirement (6)]
- [ ] **Judge Validation:** Identify cases where the 'LLM judge' assessed performance incorrectly. [Requirement (6)]
- [ ] **Effectiveness Summary:** Summarize the overall effectiveness and utility of the automated pipeline. [Requirement (6)]
- [ ] **Final Submission Prep:** Ensure all data from requirements 3-6 is ready for submission. [Requirement (6)]
