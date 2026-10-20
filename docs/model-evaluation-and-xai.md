# Model Evaluation and Explainability

Verified against the repository evaluation scripts on 20 October 2026.

## Models and pipeline components

| Component | Role | Available performance evidence |
| --- | --- | --- |
| `BAAI/bge-small-en-v1.5` | Dense query embeddings for semantic retrieval | The current evaluation measures hybrid retrieval with BM25 together; it does not isolate embedding-only performance. On the current index: Recall@6 20/20 (100%) and MRR 0.917. |
| Configured LLM provider/model | Generates Q&A responses and assists document summarization and complex entity extraction | No live-model answer quality, faithfulness, or latency benchmark is recorded. The five `tests/test_llm_evaluation.py` tests use fixed/mock answer payloads and test the evaluator, not the configured model. The Evaluation page reads the configured provider/model identifier from the API without exposing credentials. |
| BM25 | Exact-term retrieval for section numbers, forms, and statutory terms; not a deep-learning model | No standalone BM25 score is recorded. Its contribution is part of the hybrid retrieval benchmark. The default fusion weight is 35%, configurable with `DENSE_WEIGHT`. |
| Regex rules | PAN, TAN, GSTIN, and amount extraction; not a deep-learning model | No precision/recall benchmark is recorded. |

The retrieval evaluation uses 20 in-scope questions and six returned results (`Recall@6`), plus five off-topic controls. It reports 100% refusal on the five off-topic controls. These small fixed sets are useful regression checks, not a broad independent quality study. Changing the embedding model requires rerunning and recalibrating the retrieval benchmark. Do not interpret the hybrid measurements as isolated embedding-model scores.

## Explainability algorithm: Hybrid Retrieval Score Attribution

Each source returned by Q&A includes the actual per-query components used to rank it. Retrieval min-max normalizes dense similarity and BM25 scores within their respective candidate pools, combines them using the configured dense weight, then applies an optional statute applicability weight:

```text
dense contribution = DENSE_WEIGHT × normalized dense score × statute weight
BM25 contribution  = (1 − DENSE_WEIGHT) × normalized BM25 score × statute weight
final retrieval score = dense contribution + BM25 contribution
```

The API returns both contributions for each source. The Sources rail exposes them under “Why this source ranked”; the Evaluation page describes the method. A lower statute weight may demote a passage associated with a superseded Act. The source card also keeps its legal citation and excerpt so the ranking signal is inspectable alongside the retrieved text.

This is an auditable decomposition of the existing retrieval score. It is not SHAP or LIME, does not explain hidden LLM reasoning, does not establish that an answer is legally correct, and does not claim causal feature attribution. The generator’s answers should be judged against retrieved sources and qualified legal review.

## Reproduce

From the repository root:

```powershell
python -m pytest -q tests/test_llm_evaluation.py
python rag_model/scripts/eval_retrieval.py
```

The first command validates the evaluator using fixed examples. The second measures retrieval and refusal behavior against the locally built index; it does not call the answer-generation model.
