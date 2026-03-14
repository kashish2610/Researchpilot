# ResearchPilot QA 

A **multi-agent Retrieval-Augmented Generation (RAG)** pipeline for intelligent research question answering. It combines local PDF knowledge bases with live web search, orchestrated through a chain of specialized LLM agents.

---

## Overview

ResearchPilot QA answers complex research questions by decomposing them into sub-queries, intelligently routing each sub-query to either a local vector store (research papers) or live web search, then reviewing and synthesizing all results into a single, coherent response.

The system was demonstrated on the **NextG-GPT** paper — a domain-specific RAG-enhanced LLM assistant for telecom/O-RAN research.

---

## Architecture

```
User Question
     │
     ▼
┌─────────────────────┐
│  Query Decomposition │  → Generates 3 sub-queries
│       Agent          │
└─────────────────────┘
          │
          ▼ (for each sub-query)
┌─────────────────────┐
│   Decision Agent     │  → Web search needed? YES / NO
└─────────────────────┘
     │              │
   YES              NO
     ▼              ▼
┌─────────┐   ┌──────────────┐
│   Web   │   │  FAISS Vector │
│ Search  │   │  Retriever    │
│(DDG API)│   │  (PDF Chunks) │
└─────────┘   └──────────────┘
     │              │
     ▼              ▼
┌─────────────────────┐
│    Reader Agent      │  → Extracts factual points
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│    Critic Agent      │  → Validates & corrects answer
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│  Synthesizer Agent   │  → Combines all into final answer
└─────────────────────┘
          │
          ▼
    Final Answer
```

---

## Agents

| Agent | Role |
|---|---|
| **Query Decomposition Agent** | Breaks the user question into 3 targeted sub-queries for better coverage |
| **Decision Agent** | Determines whether each sub-query needs live web data or can be answered from research papers |
| **Web Reader Agent** | Extracts relevant bullet-point facts from DuckDuckGo search results |
| **Retriever Agent** | Fetches top-6 relevant chunks from the FAISS vector store |
| **Reader Agent** | Analyzes retrieved paper chunks and extracts factual points with citations |
| **Critic Agent** | Reviews each answer for accuracy and returns a corrected version |
| **Synthesizer Agent** | Merges all reviewed answers into one clear, structured final response |

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| LLM | `meta-llama/Meta-Llama-3.1-8B-Instruct` via HuggingFace Inference API |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (`faiss-cpu`) |
| Orchestration | LangChain (`langchain`, `langchain-community`, `langchain-core`) |
| Web Search | DuckDuckGo Search (`duckduckgo-search`, `ddgs`) |
| PDF Loading | `PyPDFLoader` (pypdf) |
| Text Splitting | `RecursiveCharacterTextSplitter` (chunk_size=800, overlap=150) |
| Runtime | Google Colab (Python 3) |

---

## Setup & Usage

### Prerequisites

- A [HuggingFace](https://huggingface.co) account with API token
- Google Colab (or a local Python environment)
- A research PDF (e.g., `NEXTG_GPT.pdf`)

### Installation

```bash
pip install langchain langchain-community langchain-text-splitters
pip install sentence-transformers faiss-cpu pypdf chromadb
pip install transformers accelerate langchain_huggingface
pip install duckduckgo-search ddgs
```

### Configuration

Set your HuggingFace token (in Colab):
```python
from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

### Running

```python
question = "What work is Gen AI doing in wireless communication, and how does NextG-GPT use RAG?"
print(multi_agent_rag_with_web(question))
```

---

## Example Output

For the question about **Gen AI in wireless communication and NextG-GPT**, the pipeline produced a structured critical review including:
- Strengths of the NextG-GPT approach (real-world testbed, RAG integration, O-RAN use cases)
- Weaknesses (limited evaluation, insufficient comparisons)
- Recommendations for future work
- Overall rating breakdown (Originality, Significance, Technical Quality, Clarity)

---

## Project Structure

```
ResearchPilot_QA.ipynb   # Main notebook with full pipeline
NEXTG_GPT.pdf            # Source research paper (user-provided)
README.md                # This file
```

---

## Key Design Decisions

- **Hybrid retrieval**: Dynamically routes between local PDF knowledge and live web search per sub-query, rather than using one source for everything.
- **Critic-in-the-loop**: Every agent answer passes through a critic before synthesis, reducing hallucination and improving factual accuracy.
- **Query decomposition**: Splitting the original question into 3 sub-queries improves recall and ensures different technical angles are covered.

---

## Limitations

- Requires an active HuggingFace Inference API token (rate limits may apply on free tier).
- Web search quality depends on DuckDuckGo result availability.
- The decision agent (YES/NO routing) may occasionally misclassify edge-case queries.

---

## License

This project is for research and educational purposes.
