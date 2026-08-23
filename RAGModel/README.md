# RAGModel — grounded QA over Indian tax law

A retrieval-augmented question answering system over the **Income-tax Act, 2025**
(as amended by the Finance Act, 2026) and the **Income-tax Rules, 2026**.

Ask a question in plain English; get an answer assembled only from retrieved
statutory text, with a citation to the provision behind every claim — or an
honest refusal when nothing in the knowledge base answers it.

```
python ask.py "What deductions are allowed for life insurance premium?"
```

---

## Quick start

```bash
pip install -r requirements.txt
python download_models.py --with-llm   # embedder (~130 MB) + generator (~3 GB)
python build_index.py                  # ~1 min for 3,152 chunks
python ask.py                          # interactive
```

`--with-llm` is optional. Without any language model the system still answers —
it quotes the retrieved provisions verbatim instead of summarising them (see
*Providers* below).

---

## How it works

```
question
   │
   ├─ embed (BAAI/bge-small-en-v1.5)  ──┐
   │                                    ├─ fuse ─→ dedupe by provision ─→ top-6
   └─ BM25 (provision & form numbers) ──┘
                                              │
                          ┌───────────────────┴───────────────────┐
                          │ best cosine < 0.60?                   │
                          │   yes → refuse, never call the model   │
                          │   no  → grounded prompt → LLM → stream │
                          └───────────────────┬───────────────────┘
                                              │
                              answer + citations + disclaimer
```

**Hybrid retrieval.** Dense embeddings handle paraphrase ("can I write off my
work laptop" → depreciation); BM25 handles the tokens embeddings blur over —
provision numbers, form numbers, rupee limits. Scores are min-max normalised
within each candidate pool and combined 65/35 in favour of dense.

**Dedupe by provision.** A long section spans several adjacent chunks that mostly
repeat each other. Collapsing them means the context window carries six distinct
provisions rather than six slices of two.

**Citations are re-derived, not trusted.** The KB shipped with a
`metadata.section` field guessed by a loose regex, which fires on any *mention*
of a section — "sub-section" yields `TION`, and the definition of "accountant"
yields section 515 because it cross-references it. [`rag/kb.py`](rag/kb.py)
discards that and walks each document in reading order, carrying the current
provision across chunk boundaries, rejecting numbers that run backwards (table
rows, quoted lists), and switching to Schedule/Appendix numbering where the
statute does. Every one of the 3,152 chunks resolves to a citable provision.

---

## Fiduciary guardrails

Grounding is enforced in three places, because prompt instructions alone are not
enforcement:

| Layer | Mechanism | Enforceable? |
|---|---|---|
| Retrieval | Best cosine below `MIN_SCORE` → refuse before any model call | Yes — in code |
| Prompt | System message forbids outside knowledge, requires a cite per claim | No — instruction only |
| Post-hoc | Disclaimer appended by the pipeline, not by the model | Yes — in code |

The threshold was calibrated against this corpus rather than picked: on-topic tax
questions retrieve at cosine 0.70–0.82, off-topic controls (football, baking,
programming) at 0.48–0.54. `MIN_SCORE` sits at 0.60, between the two.

Questions phrased as personal advice ("should I…", "can I claim…") add an
instruction to explain the general rule, decline to prescribe, and direct the
user to a Chartered Accountant.

**Known coverage gap, refused explicitly.** Tax *rates and slabs* are not in this
corpus: section 4 charges income-tax "at the rate or rates specified in the
Finance Act", and the slab table lives in the Finance Act, which is not one of
the ingested PDFs. Left to retrieval these questions still return
plausible-looking provisions at a healthy cosine, and a model then fills the gap
from memory — in testing, one such question produced a confident *"you should not
pay any tax"*. So slab and "how much tax do I owe" questions are intercepted
before retrieval and refused with an explanation of what is missing and why.
Impersonal rate questions ("how much tax is deducted at source on rent") still
answer normally, because TDS rates genuinely are in the Act.

Adding the Finance Act to `Data/tax_pdfs/` and re-running the ingest closes this
gap; the guard in `rag/prompt.py` should then be removed.

---

## Providers

The pipeline never imports a vendor SDK directly. `LLM_PROVIDER=auto` walks:

| Provider | Requires | Notes |
|---|---|---|
| `ollama` | `ollama serve` + a pulled model | Free, offline, private |
| `local` | `python download_models.py --with-llm` | Qwen2.5-1.5B-Instruct in-process; uses the GPU if present |
| `openai` | `OPENAI_API_KEY` | Also honours `OPENAI_BASE_URL` for compatible endpoints |
| `extractive` | nothing | No LLM: quotes retrieved provisions verbatim |

Local backends come first deliberately — free, offline, and users' tax questions
stay off third-party servers. `extractive` is a genuine fallback rather than an
error path: quoting the statute with its citation is the most fiduciary-safe
answer available, just a less readable one. If a chosen backend fails mid-request
(quota, network), the pipeline degrades to `extractive` rather than raising.

```bash
python ask.py --providers          # which backends are reachable right now
python ask.py --provider local -q "..."
```

---

## Commands

```bash
python ask.py "question"                    # one-shot
python ask.py                               # interactive session
python ask.py --retrieval-only -q "..."     # inspect retrieval, no generation
python ask.py --providers                   # backend availability
python build_index.py                       # rebuild after the KB changes
python download_models.py [--with-llm]      # prefetch models
python eval_retrieval.py [--verbose]        # retrieval + refusal metrics
python -m pytest tests/ -q                  # unit tests
```

Tuning knobs are environment variables read in [`rag/config.py`](rag/config.py):
`TOP_K`, `MIN_SCORE`, `DENSE_WEIGHT`, `EMBED_MODEL`, `LLM_PROVIDER`,
`OLLAMA_MODEL`, `LOCAL_MODEL`, `OPENAI_MODEL`, `MAX_TOKENS`, `TEMPERATURE`.

---

## Measured quality

`python eval_retrieval.py`, 20 gold questions + 5 off-topic controls:

| Metric | Result |
|---|---|
| recall@6 | 100% (20/20) |
| MRR | 0.912 |
| Refusal rate on off-topic | 100% (5/5) |

The gold set's expected provisions were read off headings in the knowledge base
itself, not recalled from memory — this Act **renumbered** the familiar sections,
so the old 80C deduction now lives in **section 123 and Schedule XV**, and the
old 80D health-insurance deduction in **section 126**. Answers cite the 2025 Act's
numbering, which will not match pre-2026 guidance found elsewhere.

---

## Limitations

- **Generation quality is bounded by the model you plug in.** The default
  `local` provider is Qwen2.5-1.5B-Instruct, chosen to fit a 6 GB GPU. It
  summarises retrieved text competently but reasons poorly over numbers and
  follows the bracket-citation format loosely — it tends to cite provisions by
  name in prose instead. For anything beyond a demo, run Ollama with an 8B+
  model; retrieval, citations and guardrails are unchanged, only the wording
  improves. Retrieval is the part measured above; generation is not.
- **No rates or slabs** — see the coverage gap above.
- **Renumbered Act.** This is the 2025 Act, not the 1961 one. Section numbers
  will not match pre-2026 guidance found elsewhere.
- **Chunk boundaries come from the existing ingest** (`Data/textChunker.py`,
  1200 chars with 250 overlap), so a long section is split mid-provision.
  Citations are resolved correctly across those splits, but a single retrieved
  passage may reproduce only part of a section — the prompt instructs the model
  to say so when that happens.
- **No conversational memory beyond the last three turns**, and none across runs.

## Updating the knowledge base

Tax law changes yearly, and the entire update path is a re-ingest:

```bash
# 1. drop new/updated PDFs into Data/tax_pdfs/
cd Data && python textChunker.py     # → Data/rag_knowledge_base.json
cd .. && python build_index.py       # → index/
```

`index/` is a derived artifact and is gitignored; nothing important should exist
only as a generated file. Changing `EMBED_MODEL` requires a full re-index, and
`rag/index.py` refuses to load a mismatched index rather than returning silently
wrong neighbours.

---

## Layout

```
rag/
  config.py     tunables, all env-overridable
  kb.py         chunk loading + provision/citation resolution
  embedder.py   sentence-transformers wrapper
  lexical.py    BM25 (tokenizer keeps "80CCD" and "Form No. 154" whole)
  index.py      build/save/load FAISS + BM25, with staleness checks
  retrieve.py   hybrid search, fusion, dedupe, grounding gate
  prompt.py     guardrails, context assembly, refusal + disclaimer text
  llm.py        provider abstraction (ollama / local / openai / extractive)
  answer.py     end-to-end pipeline, streaming
ask.py               CLI
build_index.py       index builder
download_models.py   model prefetch
eval_retrieval.py    retrieval + refusal evaluation
tests/               unit tests
```

---

## Troubleshooting

**`CERTIFICATE_VERIFY_FAILED` on model download.** This machine sits behind a
TLS-intercepting proxy whose root CA lives in the Windows certificate store but
not in certifi's bundle, so `curl` works while Python fails. Export the Windows
store and `rag/config.py` will pick the file up automatically:

```powershell
$out = "win-ca-bundle.pem"; $sb = New-Object System.Text.StringBuilder; $seen = @{}
Get-ChildItem Cert:\LocalMachine\Root, Cert:\CurrentUser\Root, Cert:\LocalMachine\CA | ForEach-Object {
  if (-not $seen.ContainsKey($_.Thumbprint)) { $seen[$_.Thumbprint] = $true
    [void]$sb.AppendLine("-----BEGIN CERTIFICATE-----")
    [void]$sb.AppendLine([Convert]::ToBase64String($_.RawData,'InsertLineBreaks'))
    [void]$sb.AppendLine("-----END CERTIFICATE-----") } }
Set-Content $out $sb.ToString() -Encoding ascii
```

**`Keras 3 ... not yet supported in Transformers`.** `rag/config.py` sets
`USE_TF=0` so transformers never probes TensorFlow. Import `rag.config` before
`transformers` if you write your own entry point.

**Garbled output in the Windows console.** Run with `PYTHONUTF8=1`.

---

## Scope

Educational information about Indian tax law. Not professional tax, accounting or
financial advice, and not a Chartered Accountant. It explains what the law says;
it will not tell you what to do about your own return.
