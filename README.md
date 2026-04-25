# 5sting at SemEval-2026 Task 8: MTRAG
> **Multi-Turn RAG Benchmark** — A comprehensive retrieval and generation pipeline for the MTRAG shared task. [See detail here](https://github.com/IBM/mt-rag-benchmark)

---
## Task Descriptions

| Subtask | Description |
|---------|-------------|
| **Subtask A - Retrieval** | Given a task that is answerable and has a set of relevant passages, evaluate whether the retrieval system retrieves the relevant passages. |
| **Subtask C - Generation with Retrieved Passages (RAG)** | Given a task with a target answer, first retrieve 5 passages, then generate an answer. Evaluate the predicted answer of the generator system against the reference answer. |

---
## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Session 2: Setup & Execution Guide](#2-session-2-setup--execution-guide)
3. [Subtask A - Retrieval](#3-subtask-a--development-phase)
4. [Subtask C - Generation with Retrieved Passages (RAG)](#4-subtask-c--generation)

---

## 1. Project Overview

This codebase implements a **Multi-Turn Retrieval-Augmented Generation (RAG)** pipeline for the MTRAG benchmark competition. The benchmark evaluates RAG systems on multi-turn conversational queries across 4 domains:

| Domain  | Description |
|--------|-------------|
| **ClapNQ** |  Questions from ClapNQ QA dataset |
| **Cloud** |  IBM Cloud documentation |
| **FiQA** |  Financial QA dataset |
| **Govt** |  Government documents |

The project is organized into two main execution phases:
- **`subtask_A/`** — Development & validation pipeline (tested on provided dev data with ground-truth qrels)
- **`subtask_C/`** — Development & validation pipeline (tested on provided dev data with ground-truth qrels)

---

## 2. Session 2: Setup & Execution Guide

This section provides step-by-step instructions for setting up the environment and running each subtask.

### 2.1 Prerequisites

- **Python**: 3.8+
- **CUDA** (optional): For GPU acceleration (11.0+)
- **OpenAI API Key**: Required for reranking and answer generation (GPT-4o-mini)
- **Git**: For cloning and version control
- **Virtual Environment**: Recommended (venv or conda)

### 2.2 Environment Configuration (.env Setup)

#### 2.2.1 Create .env File from Template

```bash
# Copy the example configuration
cp .env.example .env
```

#### 2.2.2 Configure .env Parameters

In `.env`, replace `your-openai-api-key-here` with your actual key:
   ```bash
   OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

### 2.3 Installation & Dependencies

#### 2.3.1 Create Virtual Environment

**Using venv (Python):**
```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Windows cmd)
venv\Scripts\activate.bat
```

**Using conda:**
```bash
conda create -n mtrag python=3.10
conda activate mtrag
```

#### 2.3.2 Install Dependencies

**For Subtask A (Retrieval):**
```bash
cd subtask_A
pip install -r requirements.txt
cd ..
```

**For Subtask C (Generation):**
```bash
# Subtask C uses dependencies from subtask_A, but you can install separately:
cd subtask_A
pip install -r requirements.txt
cd ..
```

### 2.3 Data Preparation

Ensure the following data files are in place (you can get data from [SemEval-2026](https://github.com/IBM/mt-rag-benchmark)):

```
mt-rag-benchmark/
├── corpora/                        # Document corpus
│   └── passage_level/              # Passage-level files
│       ├── clapnq.jsonl
│       ├── cloud.jsonl
│       ├── fiqa.jsonl
│       └── govt.jsonl
│
├── human/
│   ├── generation_tasks/           # Question files
│   │   ├── RAG.jsonl               # For Subtask C
│   │   └── reference_subset_with_human_evaluations.json
│   │
│   ├── retrieval_tasks/            # Query files for Subtask A
│   │   ├── {domain}_questions.jsonl
│   │   ├── {domain}_rewrite.jsonl
│   │   └── {domain}_lastturn.jsonl
│   │
│   └── evaluations/
│       ├── reference.json          # Reference answers
│       ├── reference+RAG.json      # For Subtask C eval
│       └── RAG.json                # Answer evaluation data
```

### 2.5 Subtask A: Retrieval — Step-by-Step Execution

#### 2.5.1 Overview

Subtask A implements a dense retrieval pipeline with LLM-based reranking:

```
Queries → Embedding → FAISS Search → LLM Reranking → Results
```

#### 2.5.2 Step 1: Build Indexes (Optional but Recommended)

Pre-build FAISS indexes for all domains to speed up retrieval:

```bash
cd subtask_A
python build_index.py
cd ..
```


#### 2.5.3 Step 2: Run Main Retrieval Pipeline

Execute the full retrieval + reranking + evaluation pipeline:

```bash
cd subtask_A
python main.py
```
---

### 2.6 Subtask C: Generation — Step-by-Step Execution

#### 2.6.1 Overview

Subtask C generates grounded answers using LLM with retrieved passages:

```
History + Question + Retrieved Passages → LLM → Generated Answer
```

#### 2.6.2 Step 1: Prepare Retrieval Input (from Subtask A)

Extract top-5 contexts from Subtask A results:

```bash
cd subtaskC/data_from_A
python create_top5.py \
  --input "../../subtask_A/results/merged_retrieval_results.jsonl" \
  --output "../A_top5/retrieval_top5.jsonl" \
  --k 5
cd ../..
```

**Parameters:**
- `--input`: Path to Subtask A retrieval results
- `--output`: Where to save filtered results
- `--k`: Number of top contexts to keep (default: 5)


#### 2.6.3 Step 2: Generate Answers (LLM-based)

Run the answer generation pipeline:

```bash
cd subtaskC

# Test mode (process 1 record only)
python generate_llm_answers.py \
  --histories ../human/generation_tasks/RAG.jsonl \
  --retrieval A_top5/retrieval_top5.jsonl \
  --test true

cd ..
```

**Full mode (all records):**
```bash
cd subtaskC
python generate_llm_answers.py \
  --histories ../human/generation_tasks/RAG.jsonl \
  --retrieval A_top5/retrieval_top5.jsonl \
  --output results/submission.jsonl

cd ..
```

**Parameters:**
- `--histories`: Path to conversation history JSONL file
- `--retrieval`: Path to retrieval results JSONL file
- `--test`: `true` for 1 record (validation); `false` for all (default: `true`)
- `--output`: Where to save submission JSONL (default: `subtaskC/results/submission.jsonl`)

---

## 3. Subtask A — Development Phase

### 3.1 Overview

The `subtask_A/` folder contains the complete retrieval pipeline used during development. It processes **3 types of queries** per domain:

| Query Type | Description |
|-----------|-------------|
| `questions` | Extracted individual questions from each conversation turn |
| `rewrite` | LLM-rewritten queries that incorporate conversation context |
| `lastturn` | Only the last user turn in the conversation |

### 3.2 Key Models & Technologies

| Component | Technology | Details |
|-----------|-----------|---------|
| **Embedding Model** | `BAAI/bge-m3` | Multilingual embedding model, 1024-dim, max 8192 tokens |
| **Vector Index** | FAISS (`IndexFlatIP`) | Cosine similarity via Inner Product on normalized vectors |
| **Reranking LLM** | `gpt-4o-mini`  | LLM-as-a-judge reranker with JSON structured output |
| **Framework** | SentenceTransformers + FAISS | Dense retrieval with exact search |
| **Device** | CUDA (auto-detect, fallback to CPU) | Configurable via `.env` |

### 3.3 Module Descriptions

| File | Responsibility |
|------|---------------|
| `main.py` | Orchestrates the full pipeline: load → index → retrieve → rerank → save → evaluate |
| `config.py` | Loads settings from `.env`: model name, paths, datasets, query types, device |
| `data_loader.py` | Loads corpus (JSONL), queries, qrels (TSV), and MTRAGEval format files |
| `embedding_generator.py` | Wraps `SentenceTransformer` to encode documents and queries into dense vectors |
| `index_builder.py` | Builds, saves, loads, and searches FAISS indexes; manages doc_id ↔ index mapping |
| `retrieval_engine.py` | Combines embedding + index to perform retrieval; saves results in TSV/JSONL/MTRAGEval formats |
| `reranker.py` | LLM-based reranking: sends query + retrieved chunks to GPT-4o-mini, gets reordered ranking |
| `create_input_files.py` | Converts raw BEIR-format queries into MTRAGEval pipeline input format |
| `build_index.py` | Standalone script to pre-build FAISS indexes for all domains |
| `device_utils.py` | Auto-detects CUDA availability; returns `'cuda'` or `'cpu'` |
| `examples.py` | Example usage scripts demonstrating individual components |


### 3.5 Execution Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    START: main.py                               │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│         STEP 1: LOAD CONFIGURATION                              │
│  • Read .env: embedding model, device, datasets, query types   │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│      STEP 2: INITIALIZE COMPONENTS                              │
├─────────────────────────────────────────────────────────────────┤
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ EmbeddingGenerator: BAAI/bge-m3 (1024-dim, max=8192)    │   │
│ ├──────────────────────────────────────────────────────────┤   │
│ │ IndexBuilder: FAISS IndexFlatIP (Cosine Similarity)     │   │
│ ├──────────────────────────────────────────────────────────┤   │
│ │ RetrievalEngine: Combine embedding + index              │   │
│ ├──────────────────────────────────────────────────────────┤   │
│ │ LLMReranker: gpt-4o-mini (retrieve_k=10-20, final_k=10)   │   │
│ └──────────────────────────────────────────────────────────┘   │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│   FOR EACH DOMAIN: clapnq, cloud, fiqa, govt                   │
└─────────────────┬───────────────────────────────────────────────┘
                  │
         ┌────────▼────────┐
         │ FOR EACH QUERY  │ q: questions, r: rewrite, l: lastturn
         │ TYPE: q, r, l   │
         └────────┬────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│         STEP 3: LOAD CORPUS                                     │
│  • Load from corpora/passage_level/{domain}.jsonl              │
└─────────────────┬───────────────────────────────────────────────┘
                  │
         ┌────────▼────────────────┐
         │ Index exists?           │
         └────┬──────────────┬─────┘
        YES   │              │   NO
              │              │
    ┌─────────▼─┐    ┌───────▼──────────────┐
    │ Load from  │    │ Encode corpus with  │
    │ disk       │    │ BAAI/bge-m3 →       │
    │            │    │ Build FAISS → Save  │
    └─────────┬──┘    └───────┬──────────────┘
              │               │
              └───────┬───────┘
                      │
┌─────────────────────▼────────────────────────────────────────────┐
│         STEP 4: LOAD QUERIES                                    │
│  • Load from input_data/{domain}_{type}_input.jsonl            │
└─────────────────┬─────────────────────────────────────────────────┘
                  │
         ┌────────▼────────────────┐
         │ Query Type?             │
         └────┬──────────────┬─────┘
  questions   │              │  lastturn
    /rewrite  │              │
         ┌────▼────┐    ┌────▼────────────────────┐
         │Retrieve │    │Retrieve from lastturn   │
         │Top-20   │    │+                        │
         │(single) │    │Retrieve from rewrite    │
         │         │    │Merge & Deduplicate      │
         └────┬────┘    └────┬───────────────────┘
              │              │
              └──────┬───────┘
                     │
┌────────────────────▼──────────────────────────────────────────────┐
│         STEP 5: LLM RERANKING (GPT-4o-mini)                      │
├────────────────────────────────────────────────────────────────────┤
│ • Send: query + top-K document chunks                             │
│ • System prompt: "You are a document re-ranker..."               │
│ • Response: JSON {"order": [ranked chunk IDs]}                  │
│ • Output: Reorder documents by LLM ranking                        │
│ • Return: Top-10 after reranking                                  │
└────────────────┬──────────────────────────────────────────────────┘
                 │
┌────────────────▼──────────────────────────────────────────────────┐
│         STEP 6: SAVE RESULTS                                      │
│  • TSV format: query-id → corpus-id → score                      │
│  • JSONL format (MTRAGEval submission)                            │
└────────────────┬──────────────────────────────────────────────────┘
                 │
┌────────────────▼──────────────────────────────────────────────────┐
│         STEP 7: EVALUATE                                          │
│  • Load qrels from human/retrieval_tasks/{domain}/qrels/dev.tsv  │
│  • Compute: NDCG@{1,3,5,10}, Recall@{1,3,5,10}                  │
│  • Export: results_summary.csv                                    │
└────────────────┬──────────────────────────────────────────────────┘
                 │
         ┌───────▼──────────┐
         │ Next query type? │ ──(loop)──→ FOR EACH QUERY TYPE
         └───────┬──────────┘
                 │ No
         ┌───────▼──────────┐
         │ Next domain?     │ ──(loop)──→ FOR EACH DOMAIN
         └───────┬──────────┘
                 │ No
┌────────────────▼──────────────────────────────────────────────────┐
│              END: PIPELINE COMPLETE                               │
│       All results saved & evaluated                                │
└───────────────────────────────────────────────────────────────────┘
```

### 3.6 Reranking Detail

The reranker (`reranker.py`) uses the following approach:

1. **Input**: Query text + top-K retrieved document chunks (with content)
2. **LLM Call**: Sends a structured prompt to `gpt-4o-mini` via LiteLLM:
   - **System Prompt**: Instructs the model to act as a document re-ranker
   - **User Prompt**: Contains the question and numbered chunks
   - **Response Format**: JSON object `{"order": [3, 1, 5, 2, 4, ...]}`
3. **Output**: Documents reordered by LLM's relevance judgment, truncated to `final_k=10`
4. **Retry Strategy**: Exponential backoff (min=2s, max=10s) on API failures
5. **Fallback**: Returns original order if LLM call fails

### 3.7 Configuration Parameters

| Parameter | Default Value | Description |
|-----------|--------------|-------------|
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | HuggingFace embedding model |
| `EMBEDDING_DIMENSION` | `1024` | Embedding vector dimension |
| `MAX_SEQ_LENGTH` | `8192` | Maximum input sequence length |
| `BATCH_SIZE` | `32` | Encoding batch size |
| `SIMILARITY_METRIC` | `cosine` | FAISS similarity metric |
| `RETRIEVAL_K` | `20` | Initial retrieval count |
| `FINAL_K` | `10` | Final count after reranking |
| `DOMAINS` | `clapnq,cloud,fiqa,govt` | Domains to process |
| `QUERY_TYPES` | `questions,rewrite,lastturn` | Query types to evaluate |

---


### 4.7 Merged Retrieval Strategy Detail

```
┌────────────────────────────────────────┬────────────────────────────────────┐
│         QUERY INPUT (Phase 1)          │      QUERY INPUT (Phase 2)        │
├────────────────────────────────────────┼────────────────────────────────────┤
│ LASTTURN Query                         │ REWRITTEN Query                    │
│                                        │                                    │
│ e.g. "Tell me more about Y."          │ e.g. "...X... Y..."              │
│ (only last user message)               │ (full conversation context)        │
└────────┬───────────────────────────────┴──────────┬───────────────────────┘
         │                                          │
         │                                          │
    ┌────▼──────────────────┐              ┌────────▼─────────────────┐
    │ STEP 1A:              │              │ STEP 1B:                │
    │ Encode with           │              │ Encode with             │
    │ BAAI/bge-m3           │              │ BAAI/bge-m3             │
    │ (1024-dim vectors)    │              │ (1024-dim vectors)      │
    └────┬──────────────────┘              └────────┬─────────────────┘
         │                                          │
    ┌────▼──────────────────┐              ┌────────▼─────────────────┐
    │ STEP 2A:              │              │ STEP 2B:                │
    │ FAISS Search          │              │ FAISS Search            │
    │ IndexFlatIP           │              │ IndexFlatIP             │
    │ (Cosine similarity)   │              │ (Cosine similarity)     │
    └────┬──────────────────┘              └────────┬─────────────────┘
         │                                          │
    ┌────▼──────────────────┐              ┌────────▼─────────────────┐
    │ RESULT 1A:            │              │ RESULT 1B:              │
    │ Top-10 documents      │              │ Top-10 documents        │
    │ from lastturn query   │              │ from rewritten query    │
    │                       │              │                         │
    │ Score: [0.92, 0.91,   │              │ Score: [0.88, 0.85,     │
    │         0.89, ...]    │              │         0.84, ...]      │
    └────┬──────────────────┘              └────────┬─────────────────┘
         │                                          │
         └──────────────────┬───────────────────────┘
                            │
        ┌───────────────────▼───────────────────┐
        │   STEP 3: MERGE & DEDUPLICATE        │
        ├───────────────────────────────────────┤
        │ • Combine both result sets            │
        │ • For duplicate doc_ids:              │
        │   Keep maximum score                  │
        │ • Sort by score (descending)          │
        │ • Result: ~10-20 unique documents    │
        └───────────────────┬───────────────────┘
                            │
        ┌───────────────────▼───────────────────┐
        │   STEP 4: LLM RERANKING               │
        │         (GPT-4o-mini)                 │
        ├───────────────────────────────────────┤
        │ Input:                                │
        │ • Query: REWRITTEN query text         │
        │   (captures full context)             │
        │ • Candidates: All merged docs         │
        │   (~10-20 documents)                  │
        │                                       │
        │ Process:                              │
        │ • Send to gpt-4o-mini                 │
        │ • System prompt: re-ranker            │
        │ • Get JSON response with ranking      │
        │                                       │
        │ Output:                               │
        │ • Reorder by LLM judgment             │
        │ • Select TOP-10                       │
        └───────────────────┬───────────────────┘
                            │
        ┌───────────────────▼───────────────────┐
        │   FINAL RESULT: TOP-10 DOCUMENTS     │
        │   Ready for submission                │
        └───────────────────────────────────────┘
```

**Key insight**: The **rewritten query** is used for the LLM reranking step (not the lastturn query), because the rewritten query captures the full conversational context, making it more effective for relevance judgment.

## 4. Subtask C — Generation with Retrieved Passages (RAG)

### 4.1 Overview

The `subtaskC/` folder contains the **answer generation pipeline** for the RAG generation task. Given conversation history, a user question, and retrieved passages from Subtask A, the system generates grounded answers using an LLM. The pipeline performs:

1. **Retrieval Input Processing**: Accepts top-K retrieved passages from Subtask A outputs
2. **Answer Generation**: Uses LLM (`gpt-4o-mini`) to generate contextually grounded answers
3. **Context Management**: Separates conversation history (for coreference resolution) from retrieved passages (factual source)
4. **Output Formatting**: Produces JSONL submission files with generated answers and metadata


### 4.2 Module Descriptions

| File | Responsibility |
|------|----------------|
| `generate_llm_answers.py` | Main pipeline: loads conversation history + retrieval results, generates answers via LLM, saves to JSONL. Processes one record at a time or in test mode. |
| `check_answerability.py` | Analyzes RAG evaluation data from `human/evaluations/RAG.json` to understand task structure and answerability patterns |
| `data_from_A/create_top5.py` | Trims retrieval results to top-K contexts (default: 5). Converts Subtask A output format to C input format. Accepts `--k` parameter for variable context count. |

### 4.3 Key Models & Technologies

| Component | Technology | Details |
|-----------|-----------|---------|
| **LLM Engine** | `gpt-4o-mini` | OpenAI GPT-4o-mini for grounded answer generation |
| **API Client** | LiteLLM / OpenAI | Via `OpenAI()` client (loads API key from `.env` or environment) |
| **Input Format** | JSONL | Each line: `{"conversation_id", "history", "question", "contexts": [{"text": "..."}, ...]}` |
| **Output Format** | JSONL | Each line: `{"conversation_id", "answer", "model", "contexts_used", ...}` |
| **Framework** | Python (json, argparse, dotenv) | Lightweight, no heavy dependencies |

### 4.4 Answer Generation Process

#### 4.4.1 Input: Conversation History (JSONL)
```json
{
  "conversation_id": "conv_001",
  "history": "User: What is cloud computing?\nAssistant: Cloud computing is...",
  "question": "Tell me more about scalability."
}
```

#### 4.4.2 Input: Retrieved Passages (JSONL)
```json
{
  "conversation_id": "conv_001",
  "contexts": [
    {"text": "Cloud scalability refers to...", "passage_id": "doc_042"},
    {"text": "Horizontal scaling allows...", "passage_id": "doc_156"},
    {"text": "Auto-scaling groups manage...", "passage_id": "doc_201"}
  ]
}
```

#### 4.4.3 Output: Generated Answer Submission (JSONL)
```json
{
  "conversation_id": "conv_001",
  "answer": "Scalability in cloud computing means...",
  "model": "gpt-4o-mini",
  "contexts_used": 3,
  "timestamp": "2026-04-25T10:30:00Z"
}
```

### 4.5 Role Separation in Prompting

The LLM answer generation uses **strict role separation** to ensure faithfulness:

| Role | Source | Purpose | Constraint |
|------|--------|---------|-----------|
| **Conversation History** | History file | Pragmatic context: topic, intent, coreference resolution | NOT a factual source — do not extract facts |
| **Retrieved Passages** | Retrieval file | SOLE source of factual information | Every claim must cite a passage |
| **Question** | History file | Specific question to answer | Must be directly addressed |

This design prevents the LLM from using parametric knowledge and ensures all answers are grounded in retrieved context.

---

