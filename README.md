# Fiduciary Lens Tax Advisor System

An end-to-end AI-powered Indian tax advisor system. It features two primary components:
1. **The RAG Model (`rag_model`)**: A retrieval-augmented generation engine over the **Income-tax Act, 2025** (as amended by the Finance Act, 2026).
2. **The NLP Pipeline (`nlp_pipeline`)**: A document processing pipeline that parses PDFs (like Form 16s), summarizes them, extracts entities, and generates actionable tax insights using the RAG Model.

---

## Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Configure Credentials
Create a `.env` file in the root directory and add your LLM credentials. (e.g., Hugging Face for the embedding model and OpenRouter/OpenAI for the generator).
```env
HF_TOKEN=your_hugging_face_token
OPENAI_API_KEY=your_openrouter_api_key_here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

### 3. Build the FAISS Index
Before querying, you must build the vector index over the tax laws using `ai4bharat/indic-bert`.
```bash
python rag_model/scripts/build_index.py
```

### 4. Run the NLP Pipeline
You can run the end-to-end NLP pipeline on any tax document (PDF or Text). It will parse, summarize, extract entities, and generate rules-based tax insights.
```bash
python nlp_pipeline/src/insight.py "rag_model/data/synthetic_forms/Form16_Synthetic_Arjun_Sharma.pdf"
```

### 5. Interactive RAG Q&A
If you just want to ask general tax questions against the law base:
```bash
python rag_model/scripts/ask.py "What deductions are allowed for life insurance premium?"
```

---

## Directory Structure

```text
Fiduciary-Lens-Tax-Advisor-System/
├── rag_model/                   # Phase 1: The core RAG System
│   ├── data/                    # PDF ingest, chunking, and the JSON KB
│   ├── src/                     
│   │   ├── retrieval/           # embedder.py, lexical.py, index.py, retrieve.py
│   │   ├── generation/          # prompt.py, answer.py
│   │   └── knowledge/           # kb.py
│   └── scripts/                 # ask.py, build_index.py, eval_retrieval.py
│
├── nlp_pipeline/                # Phase 2: The NLP Pipeline
│   └── src/                     
│       ├── parse.py             # PyMuPDF document extraction
│       ├── summarize.py         # LLM-powered summarization
│       ├── extract.py           # Regex + LLM entity extraction
│       └── insight.py           # Pipeline orchestrator
│
├── shared/                      # Shared code
│   ├── llm_client.py            # Provider-agnostic LLM caller
│   └── config.py                # Central configuration
│
├── tests/                       # Unit tests
├── .env
└── README.md
```

## How the RAG Model Works

**Hybrid retrieval.** Dense embeddings (`ai4bharat/indic-bert`) handle paraphrase ("can I write off my work laptop" → depreciation); BM25 handles the tokens embeddings blur over — provision numbers, form numbers, rupee limits. Scores are min-max normalised within each candidate pool and combined 65/35 in favour of dense.

**Fiduciary guardrails.** Grounding is enforced in code: if the best cosine score is below `MIN_SCORE` (0.60), the system refuses to answer before even calling the LLM. Furthermore, questions phrased as personal advice ("should I…") trigger instructions that force the model to provide general rules and direct the user to a Chartered Accountant.

## How the NLP Pipeline Works

The `insight.py` script chains the following steps:
1. **Parse**: Uses `PyMuPDF` to read raw PDF text.
2. **Summarize**: Prompts the Llama 3.3 model to act as an Indian CA and extract action items and deadlines.
3. **Extract**: Uses regex for PAN/TAN/GSTIN/Amounts, and the LLM for complex entities like Company Names or specific Tax Deduction Sections (e.g. "Section 80C").
4. **Insights**: The extracted Tax Deduction Sections are automatically passed into the `rag_model`'s `TaxQA` engine to query the exact rules and limits for those specific deductions.

## Providers

The pipeline never imports a vendor SDK directly in the logic. `LLM_PROVIDER` in `shared/config.py` can be set to:
- `ollama`: Free, offline, private
- `local`: `python download_models.py --with-llm` for in-process transformers
- `openai`: Uses `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL` (currently configured for OpenRouter).

## Scope & Disclaimer
Educational information about Indian tax law. Not professional tax, accounting or financial advice, and not a Chartered Accountant. It explains what the law says; it will not tell you what to do about your own return.
