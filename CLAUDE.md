# CLAUDE.md — Fiduciary-Lens Tax Advisor System

> Operating manual for any Claude (or human) working in this repo. Read this top-to-bottom before starting any task. It encodes **what** we are building, **how** the pieces fit, and **which skills/plugins to invoke when**.

---

## 1. Project overview & "Fiduciary Lens" philosophy

**What:** An AI-powered virtual assistant that helps users understand the **Indian taxation system** through natural-language chat. Instead of reading dense legal text, users ask a question and get a clear, accurate, cited explanation.

**Audience:** individuals, students, salaried employees, freelancers, small business owners.

**Scope:** a college NLP case-study project (NLP + LLM + Retrieval-Augmented Generation). The deliverable is a working, demonstrable chatbot plus the methodology behind it.

**Why "Fiduciary Lens":** A fiduciary acts in the client's best interest and *never misleads*. This chatbot must behave the same way. Every design and engineering choice should be biased toward **accuracy, grounding, and honesty**:

- **Grounded answers only.** Every answer must be derived from retrieved tax-law text, not the model's parametric memory. No retrieved evidence → say you don't know, don't guess.
- **Always cite.** Responses must reference the relevant Income Tax Act section / rule / circular that supports them.
- **Always disclaim.** The UI and answers must make clear this is **educational information, not professional tax/financial/accounting advice**. We are not Chartered Accountants.
- **Refuse personal financial advice.** The bot explains the law; it does not tell a specific user "you should pay ₹X" or "do this tax plan." It can illustrate, but it must not prescribe personalized action.
- **No fabrication of provisions.** If a section number, limit, date, or rate is uncertain, the system says so rather than inventing one. Stale tax data is worse than no data.

These five rules are non-negotiable. When in doubt, the answer that is **honest and limited** beats the answer that is **fluent and speculative**.

---

## 2. Tech stack & architecture

### Stack (decided)
- **Backend:** Python + FastAPI (async, SSE streaming for token-by-token answers)
- **Frontend:** React + Vite (component-driven SPA chat UI)
- **LLM:** **provider-agnostic.** Do NOT pin a provider. Behind an abstraction layer (`backend/app/llm/`) so the system can run on **local Ollama** (free, offline, private) **or a cloud API** (OpenAI / Anthropic / Gemini / Hugging Face Inference). Config selects the provider; code is identical.
- **RAG:** local embedding model (sentence-transformers) + vector store (FAISS or Chroma) + (optional) reranker
- **Knowledge base source:** Income Tax Act + official rules (curated legal text), ingested offline.

### Out of scope (YAGNI — do not let these creep in)
- ❌ User accounts / authentication — single-session demo is fine.
- ❌ Persistent multi-user history database — in-memory session history only.
- ❌ Payment / billing / subscriptions.
- ❌ Fine-tuning a model — RAG is better here: cheaper, more accurate, and trivially updatable when tax law changes. Stick with RAG.
- ❌ Giving definitive personalized tax-planning directives to users.

### Architecture

```
 Browser (React + Vite)                         Backend (Python + FastAPI)
 ┌──────────────────────────┐                  ┌──────────────────────────────────────┐
 │ Chat UI                  │  POST /api/chat  │ API layer (routes + SSE streaming)   │
 │  · question input         │ ──────────────▶ │ Conversation / session manager (hist) │
 │  · streamed answer        │ ◀── SSE tokens ─ │                                        │
 │ Sources panel (citations) │                  │ RAG pipeline:                         │
 │ Disclaimer banner         │                  │  embed query → top-k retrieval         │
 │ "educational, not advice" │                  │  → (rerank) → grounded prompt assembly │
 └──────────────────────────┘                  │ LLM abstraction (Ollama / Cloud swap)  │
                                               │ Guardrails:                            │
                                               │  cite-or-refuse · disclaimer injection │
                                               │  refuse personal advice · no-context→  │
                                               │  "I don't know"                        │
                                               │ Logging/metrics (query, chunks, lat)   │
                                               └───────────────────┬────────────────────┘
                                                                 │ (loads at startup; rebuilt offline)
                       ┌─────────────────────────────────────────┴───────────────────────┐
                       │ Knowledge Base (offline ingestion, versioned)                    │
                       │ Income Tax Act + rules → section-aware chunking                  │
                       │ → embeddings (local) → vector store (FAISS/Chroma)                │
                       │ metadata: §-number, Act, year, source URL                        │
                       └──────────────────────────────────────────────────────────────────┘
```

### Per-query data flow (matches the project workflow)
1. User enters a tax question in the chat UI.
2. Frontend `POST /api/chat` → backend **embeds the query**.
3. Backend **retrieves top-k** relevant passages from the vector store (optionally reranks).
4. Backend **assembles a grounded prompt**: system instructions (fiduciary guardrails) + retrieved chunks + their citation metadata + conversation history.
5. Backend **streams the LLM answer token-by-token** back via SSE.
6. Frontend renders the answer + the **sources/citations panel** + the persistent **disclaimer banner**.

> When building any part of this pipeline, isolate each stage (embed → retrieve → assemble → generate → stream) behind a single-purpose function/module with a clear interface so it can be tested and swapped independently.

---

## 3. Repo layout & commands

Target layout (create as you go; do not pre-scaffold everything at once):

```
/
├── CLAUDE.md                  ← you are here
├── frontend/                  ← React + Vite SPA
│   ├── src/
│   └── package.json
├── backend/
│   ├── app/
│   │   ├── api/               ← FastAPI routes (chat, health, sources)
│   │   ├── rag/               ← retrieval, prompt assembly, guardrails
│   │   ├── llm/               ← provider abstraction (Ollama + cloud adapters)
│   │   ├── ingest/            ← KB build pipeline (chunk, embed, index)
│   │   └── core/              ← config, logging, session/state
│   ├── data/                  ← raw tax source docs + processed KB (gitignore large artifacts)
│   ├── tests/                 ← unit + retrieval/grounding eval
│   └── pyproject.toml
├── docs/
│   └── superpowers/specs/     ← design specs from brainstorming
└── .gitignore
```

### Common commands (fill exact invocations as each piece lands)
- Run backend dev server: `uvicorn` / FastAPI dev entrypoint (backend/)
- Run frontend dev server: `npm run dev` (frontend/, Vite)
- (Re)build the knowledge base: a `python -m app.ingest` entrypoint (backend/)
- Run tests + retrieval eval: `pytest` (backend/tests/)
- Run the whole app: prefer the **`run` skill** (see §5) to launch and screenshot.

> Keep `data/` raw source documents in git. **Gitignore the large generated artifacts** (vector index, embeddings cache) — they must be rebuildable from source via the ingest pipeline. Nothing important should exist only as a generated artifact.

---

## 4. Domain rules (Indian taxation) — non-negotiable

1. **Source of truth = the retrieved Act/rules text.** Answers are anchored to passages the retriever returned. If retriever confidence is low or no passage clearly answers, the bot responds with "I can't find a provision covering that — please consult a Chartered Accountant" rather than improvising.
2. **Mandatory citations.** Every substantive claim cites its §-section / rule / circular. The Sources panel shows these with the originating Act + year + source URL where available.
3. **Section-aware knowledge base.** Chunk by Act/Rule/Section boundaries — not by fixed character count — so retrieval returns coherent legal units and citations are clean.
4. **Accuracy > fluency.** Section numbers, deduction limits (e.g. 80C ₹1.5L, 80D limits), slab rates, due dates, and assessment-year specifics must come from KB text. If the KB is stale, say so.
5. **Currency / extensibility.** Tax law changes yearly (Budget, Finance Acts). The ingest pipeline must be re-runnable when source docs are updated, and the date/version of the source must be surfaced in the UI so users know how current the guidance is.
6. **Refusal behavior (fiduciary).** Decline to: give personalized "you should do X" directives; compute a specific user's tax liability definitively; advise on evasion. May illustrate with examples, must redirect to a CA for personal decisions, always ends/sections with the disclaimer.
7. **Evaluation.** Maintain a small gold set of (question → expected section / acceptable answer traits) and run an automated retrieval + grounding eval as part of tests. Retrieval quality (does the right section surface?) is the metric that matters most.

---

## 5. Skill & plugin usage rules — the core operating instructions

**Rule zero:** If there is even a small chance a skill/plugin applies to the current task, **invoke it before acting** (this is the using-superpowers contract). Announce "Using [skill] to [purpose]" and follow it. The skills below are the standing set for this repo.

### Dispatch table (situation → invoke this first)

| Situation | Skill / plugin to invoke **before** acting |
|---|---|
| Any UI, layout, theme, component, or visual design work | `frontend-design:frontend-design` |
| Starting any new feature, component, or behavior | `superpowers:brainstorming` |
| Multi-step task with a spec, before touching code | `superpowers:writing-plans` |
| Executing a written implementation plan | `superpowers:executing-plans` (or `claude-mem:do`) |
| Any bug, test failure, or unexpected behavior | `superpowers:systematic-debugging` |
| Writing any feature or bugfix code | `superpowers:test-driven-development` |
| About to run unreliable long work in parallel | `superpowers:dispatching-parallel-agents` |
| Implementation done, all tests pass | `superpowers:requesting-code-review` then `superpowers:verification-before-completion` |
| Receiving code-review feedback | `superpowers:receiving-code-review` |
| Branch work complete, ready to integrate | `superpowers:finishing-a-development-branch` |
| Need isolation from current workspace | `superpowers:using-git-worktrees` |
| Before claiming "done/fixed/passing" | `superpowers:verification-before-completion` (evidence before assertion) |
| "Did we solve this / how did we do X before?" | `claude-mem:mem-search` |
| Onboarding / need to understand the codebase | `claude-mem:learn-codebase` (or `Explore` agent) |
| Building a chart / dashboard / any data viz | `dataviz` |
| Deep multi-source fact-check on a tax-law topic | `deep-research` |
| Asked to launch / run / screenshot the app | `run` |
| Reviewing a diff / PR | `/code-review` (or `review`) |
| Security review of pending changes | `security-review` |
| Clean-up pass on recently changed code | `simplify` |
| Recurring task (e.g. "every 5 min check X") | `loop` |

### frontend-design (`frontend-design:frontend-design`) — for ALL design & UI work
- **Invoke for every frontend task**: chat layout, message bubbles, the sources/citations panel, the disclaimer banner, color/type system, light/dark, responsive layout, empty/loading/error states, streaming-token rendering.
- Process order: brainstorm the UX (what should it *do*) first, then `frontend-design` for *how it looks and feels*. frontend-design is an implementation-skill; it carries out a design, it doesn't define product behavior.
- Goal: a UI that does **not** read as a templated default — intentional typography, cohesive palette, accessible contrast (especially important for a finance/advisory product where trust is the product).
- Mail frontend-design decisions through the Sources panel and disclaimer prominently — these are fiduciary affordances, not decoration.

### superpowers — for planning & review
- **`superpowers:brainstorming` — before any creative work.** Creating a feature, component, or new behavior starts here. It produces a design + spec at `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md` and hands off to writing-plans. Do not skip to code.
- **`superpowers:writing-plans`** turns an approved spec into a step-by-step implementation plan.
- **`superpowers:systematic-debugging` — before proposing any fix.** For every bug/test failure/unexpected behavior, use the structured method (reproduce → root-cause → minimal fix → verify). Never patch symptoms.
- **`superpowers:test-driven-development`** when writing features/bugfixes: write the failing retrieval/grounding/API test first, then implement to green.
- **`superpowers:verification-before-completion`** — run the actual commands and show output before ever saying "done", "fixed", or "passing". Evidence or it didn't happen.
- **`superpowers:requesting-code-review` & `superpowers:receiving-code-review`** — review completed work before merge; when receiving feedback, verify it technically rather than performatively agreeing.
- **`superpowers:using-git-worktrees`** for feature work that needs isolation from the current workspace.
- **`superpowers:finishing-a-development-branch`** when implementation is complete and tests pass — choose merge/PR/cleanup deliberately.
- Use the **`Plan` / `feature-dev:*` agents** for architecture exploration and review, and the **`Explore`** agent for broad "where is X in the codebase" sweeps (prefer these over manual grepping for substantial searches).

### claude-mem — for memory & cross-session context management
- **`claude-mem:mem-search` before solving non-trivial problems** — answer "did we already solve this / how did we do X last time" from prior sessions before re-deriving it. Check this before writing a new retrieval strategy, a guardrail, or revisiting a tax-law ingest question.
- Let claude-mem **capture observations** as you work (decisions, bugfixes, discoveries) so future sessions inherit context. Rely on it especially for the RAG pipeline decisions that are easy to lose: chunking rules, embedding model choice, guardrail prompts, eval results.
- **`claude-mem:learn-codebase`** to prime / onboarding when context is large; **`claude-mem:knowledge-agent`** to build a queryable brain over prior work (e.g. a "tax-RAG decisions" corpus).
- **`claude-mem:make-plan` / `claude-mem:do`** are an alternative planning→execution pair you may use for multi-step implementation work; pick writing-plans/executing-plans or make-plan/do consistently for a given task, don't mix mid-task.
- The repo also has a lightweight file-based memory at `C:\Users\mails\.claude\projects\e--Ganesh-Sarathi-College-Coding-NLP-Case-Study-Project-Fiduciary-Lens-Tax-Advisor-System\memory\` with a `MEMORY.md` index. Use it for small durable facts (user prefs, project constraints not in code). For richer cross-session recall, use claude-mem. Don't duplicate the same fact in both.

### Other skills / plugins — invoke when the situation fits
- **`dataviz`** — any chart, retrieval-latency plot, eval dashboard, or KPI tile. Read it before choosing chart colors or layout.
- **`deep-research`** — when a tax-law question needs multi-source verification beyond the KB (e.g. confirming a recent Budget amendment's effective date).
- **`run`** — when asked to launch/start/screenshot the app to confirm a change works in the real app, not just tests.
- **`init`** — to regenerate/refresh this CLAUDE.md if the project evolves.
- **`review` / `security-review` / `/code-review`** — for reviewing PRs and the working diff; `security-review` is worth running before any merge since we handle user-submitted financial questions.
- **`simplify`** — a quality-only cleanup pass on recently changed code (reuse, efficiency, altitude) after a feature lands.
- **`feature-dev:feature-dev`** — guided feature development with codebase understanding when a feature is sizable.
- **`loop`** — only for genuinely recurring tasks; never for one-off work.
- **`claude-mem:babysit`** — to watch a review/PR cycle to readiness if one opens.

---

## 6. Data & knowledge base

- **Sources:** Income Tax Act, allied rules, and CBDT circulars deemed in-scope. Keep source documents under `backend/data/` in git (text/markdown/structured). Record provenance per document (act, year, source URL) — this metadata flows into citations.
- **Chunking:** section-aware. Split at Act / Part / Section / Rule boundaries so a chunk is a coherent legal unit and the citation is clean. Avoid mid-section cuts.
- **Embeddings + index:** run via the ingest pipeline; store the vector index outside git (rebuildable). Pin the embedding model version and record it (changing the model = full reindex).
- **Source freshness:** track a source date/Finance Act year; surface it in the UI so users know how current the guidance is. Re-running ingest after updating source docs must be the entire update path — no manual index edits.
- **Extensibility:** adding a new tax law = drop the source doc in, ingest, done. Don't build anything that requires code changes to add new source material.

---

## 7. Domain safety & legal note

- This system provides **educational information about Indian taxation**, not professional tax, accounting, or financial advice.
- It is **not a Chartered Accountant** and must not be presented as one.
- The disclaimer ("educational, not advice; consult a CA for personal decisions") must appear (a) persistently in the UI and (b) in answer text where a user could mistake an explanation for a directive.
- Refuse to facilitate tax evasion or to give definitive personalized liability figures. Illustrating how a section applies to a hypothetical is fine; asserting what *you* should do is not.
- When retrieved evidence is weak or absent, prefer the honest "I don't know / consult a CA" response over a confident-sounding guess. This is the fiduciary lens applied at the answer layer.

---

## 8. Working agreements (quick reference)

- **Process order for any feature:** brainstorm → (writing-plans) → implement with TDD → request code review → verify before claiming done → finish branch.
- **For any bug:** systematic-debugging first, then fix, then verify.
- **For any UI:** frontend-design.
- **Before claiming completion:** run the real test/infer commands, show output.
- **Commit/push only when the user asks.** When on `main`, branch first. Don't push generated KB artifacts.
- **File hygiene:** keep modules small and single-purpose (one stage of the RAG pipeline per module). If a file grows large, that's a signal to split it.
- **Match surrounding code** in naming, density, and idiom.
- **Memory:** `mem-search` (claude-mem) before non-trivial solves; lightweight file-based memory for durable prefs/constraints.
