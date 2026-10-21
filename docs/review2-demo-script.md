# Review 2 — Demo Recording Script

**Format:** screen recording with voice-over (see "Why a screen recording" below)
**Target length:** 9–10 minutes
**Speakers:** all four members. The instruction says "execute the application **as a team** and explain the working of product", so one narrator does not satisfy it. One person screen-records and drives the mouse; the other three take the sections marked below and speak over the same recording (easiest over a Meet/Teams call with everyone unmuted, recorded locally).
**Upload:** Google Drive, then put the Drive link in the readme alongside the other three URLs

---

## Before you hit record

- [ ] Backend running: `uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000`
- [ ] Frontend running: `python -m http.server 3000 --directory frontend`
- [ ] Index built (`index/manifest.json` present) and PostgreSQL up with `alembic upgrade head`
- [ ] A test account already registered — **do not create an account on camera**, it wastes 40 seconds
- [ ] Synthetic Form 16 ready at a path you can find fast: `rag_model/data/synthetic_forms/Form16_Synthetic_Arjun_Sharma.pdf`
- [ ] Deck open in a second window, slide 1 showing
- [ ] Browser zoomed to ~125% so text is readable in the recording
- [ ] Close Slack/WhatsApp/notifications
- [ ] Do one full dry run. The chat answers take a few seconds each; you need to know where the pauses are so you can talk through them instead of going silent.

**Recording tool:** OBS Studio (free) or Windows Game Bar (`Win+Alt+R`). Record the whole screen, not just the browser, so the terminal is visible when you reference it.

---

## The script

### 0:00 – 0:40 · Opening — Member 1

> "Hi, we're Group 11. This is **Fiduciary Lens** — a tax advisory system for Indian income tax, and our slogan for it is *every answer, traced back to the law*.
>
> The short version of the problem: if you ask a general AI assistant a tax question today, you get a confident paragraph with a section number in it, and absolutely no way to tell whether that section exists or whether the limit it quoted is current. A wrong rate reads exactly like a right one.
>
> So we built a system that can only answer from the actual statute, shows you which passage it used, and refuses when it has nothing to stand on. Let me show you it working first, and then we'll go under the hood."

*Slide 1 → slide 2 while talking. Don't read the slide out loud.*

---

### 0:40 – 1:40 · Problem and landscape — Member 1

*Slides 2 and 3.*

> "Three things make this hard. Tax documents are written for practitioners, not taxpayers. The rules change every year through the Finance Act, so a model's training data is stale in a way it can't detect. And professional advice is priced out of reach for most salaried taxpayers, who are exactly the people most likely to trust a free chatbot.
>
> We looked at what exists. FlyFin, Keeper and TaxDome are mature products, but they're built on the US tax code and they optimise for categorising transactions and filing, not for showing you the law. On the Indian side, GITA, TaxbotGPT and the Income Tax Department's Kar Saathi all target Indian tax, but from what's published they're FAQ and intent driven — none of them hands you the provision behind the answer. That gap is what we're filling."

---

### 1:40 – 4:10 · Live demo 1: grounded tax Q&A — Member 2

*Switch to the browser. This is the most important section — do not rush it.*

**Step 1 — a normal question.** Type:

```
What deductions are allowed from income from house property?
```

> "So I've asked about deductions on house property income. Watch what comes back — every sentence that states a rule has a bracket number and a short label after it, like 'Act s.22'. Those aren't decoration. Scroll down and you can see the actual retrieved provisions."

*Scroll to the sources panel. Expand one.*

> "This is the passage the answer was built from, straight out of the Income-tax Act 2025, with its citation. If I don't trust the answer I can read the law myself. That's the whole point of the product."

**Step 2 — the explainability view.** Point at the score breakdown on a source card.

> "Each passage also shows *why* it was retrieved. We split the ranking score into how much came from semantic similarity and how much from exact keyword matching. If a passage was retrieved almost entirely on keyword match, it means the system found your exact section number. If it's mostly semantic, it matched your meaning and not your wording — and that's the case where you should read the retrieved text more carefully. We call it Hybrid Retrieval Score Attribution, and it's one of our two explainability mechanisms."

**Step 3 — the refusal.** Type something clearly off-topic:

```
Who won the 2018 FIFA World Cup?
```

> "Now the important part. It refuses. And it doesn't refuse because we asked the model nicely in a prompt — it refuses because there's a numeric check in the retrieval layer that runs *before* the model is ever called. If the best match scores below 0.60 cosine similarity, we never call the LLM at all. The model can't override that, because the model isn't consulted.
>
> We measured this. On-topic questions score between 0.73 and 0.84. Off-topic ones top out at 0.52. The threshold sits in an empty band between the two."

**Step 4 — the rate guard.** Type:

```
What are the income tax slab rates?
```

> "And here's a domain-specific guard. Rate questions are the most common thing a taxpayer asks and the most dangerous thing for a model to answer from memory, because a stale rate looks identical to a current one. The system checks whether a Finance Act is actually indexed before it will touch a rate question — and the check reads the indexed documents, not a hardcoded flag, so adding next year's Finance Act removes the restriction by itself."

---

### 4:10 – 6:10 · Live demo 2: document analysis — Member 3

*Go to the Documents tab, already signed in.*

> "The second half of the system starts from a document you upload. I'm using a synthetic Form 16 — we generate these ourselves with fake PANs and salaries, because we're not putting a real person's tax certificate through a demo."

*Upload the file. While it processes:*

> "Four things are happening. We pull the text out with PyMuPDF. We summarise it with the model prompted to behave like an Indian CA, so it leads with action items and deadlines. Then entity extraction — and this is deliberately split: PAN, TAN, GSTIN and rupee amounts go to regular expressions, while employer names and deduction sections go to the language model. A regex will never hallucinate a PAN, and a model will never enumerate every way an Indian company name can be written. Each one gets the job it can't get wrong."

*Results appear. Walk through the three panels.*

> "Summary at the top. Extracted entities here — you can see the PAN and the amounts came out clean. And then the part that ties the two halves of the system together: every deduction section we found in this document gets turned into a question and sent through the *same* grounded retrieval path you just saw. So each finding arrives attached to the provision that governs it, with the sources underneath and a grounded flag.
>
> One limit we're upfront about: we don't add your uploaded document to the retrieval corpus. The system explains the law relevant to your document; it doesn't do free-form Q&A about the document itself."

---

### 6:10 – 8:10 · Under the hood — Member 4

*Switch to the deck, slides 7, 8 and 10. Then the Evaluation page in the app.*

> "Quickly, the architecture. Our corpus is three official documents: the Income-tax Act 2025, the Income-tax Rules 2026, and the Finance Act 2026 — 3,496 passages after chunking at 1,200 characters with 250 overlap.
>
> We kept the Finance Act in deliberately. Section 4 of the Act charges tax at the rate 'specified in the Finance Act', so without it you have the structure of the law and none of the numbers.
>
> Retrieval runs two searches in parallel. Dense embeddings with bge-small handle paraphrase. BM25 handles the things embeddings are bad at — section numbers, form numbers, rupee limits — and we keep tokens like '80CCD' and 's.17' intact so they survive tokenisation. We fuse them 65/35 in favour of dense, demote rate tables belonging to the repealed 1961 Act, deduplicate by provision, and hand six passages to the model."

*Open the Evaluation page.*

> "Everything we claim is measured and visible in the app. Retrieval: 100% recall at six passages on a 20-question gold set, MRR 0.917 — seventeen of twenty questions return the right provision first. Refusal: five out of five on the off-topic controls.
>
> We also benchmarked three models on the same task and the same split, which is the deep-learning comparison: a trigram small language model we wrote from scratch, BERT, and DistilBERT. All three sets of metrics are in the UI here — accuracy, precision, recall, F1, plus perplexity for the SLM and model size and inference latency for all of them."

> "The result surprised us. The trigram SLM won: **0.884 accuracy** against **0.849** for BERT and **0.814** for DistilBERT, with macro F1 following the same order. And it did that with about **92,000 count cells against BERT's 109 million parameters** — roughly one eleven-hundredth — classifying a passage in **0.38 milliseconds on CPU** against 23.5 for BERT on a GPU.
>
> The reason matters more than the ranking. These three classes are separated almost entirely by surface vocabulary: the Rules talk about forms and furnishing, the Finance Act about schedules and rates. A trigram model captures exactly that. A mean-pooled transformer embedding deliberately throws surface form away in favour of meaning, which is the right thing for paraphrase and the wrong thing when the wording *is* the evidence.
>
> We're careful about what that does and doesn't prove. The transformers are frozen probes, not fine-tuned, and an n-gram model is useless at the paraphrase matching our retrieval path lives on. It's the same lesson that made us build hybrid retrieval in the first place, arrived at from the other direction: match the model to where the signal is." 

> "And our second explainability mechanism is here — leave-one-token-out occlusion. We remove each token, re-score, and see how far the decision margin drops. The tokens that matter most turn out to be the structural vocabulary of each document type, which incidentally explains why the shallow model does as well as it does."

---

### 8:10 – 9:10 · Results, honesty, close — Member 1

> "Where we've got to. Hybrid retrieval and the grounding guard both work and are measured. Document analysis runs end to end and ties findings back to provisions. Two explainability mechanisms, both surfaced in the UI.
>
> What we'd flag honestly: the gold set is twenty questions, written by us, so one miss moves recall five points — it's a regression check, not a general performance claim. Five absurd control questions are an easy refusal test; harder controls from adjacent legal areas would be the real test. And we haven't benchmarked the generation model's answer quality live, only the retrieval that feeds it.
>
> It's deployed at **[URL]**, the code's at **[GitHub URL]**, and the dataset is published on IEEE DataPort at **[DataPort URL]**. Thanks for watching."

---

## Why a screen recording, not just the slides

The instruction from 19 and 21 October reads:

> "Execute your application and have a audio recording for the same and place it in your googledrive. --- execute the application as a team and explain the working of product, This can be used for sharing to companies for internship and placement. **Use ppt slides also for support**"

Three things follow from that wording:

1. **"Execute your application"** is the primary instruction — the app has to be seen running. A slide-only presentation does not satisfy this.
2. **"ppt slides also for support"** — the deck is the supporting material, not the main event. So: demo-led, slides for the architecture and landscape bits that are hard to show live.
3. **"for sharing to companies for internship and placement"** — this sets the register. It should look like a product demo you'd send a recruiter, not a viva answer. No reading off slides, no "as you can see in this diagram".

It says "audio recording", but you cannot demonstrate a running application in audio alone. A screen recording with voice-over satisfies both readings, so record the screen. If you want to hedge, also export the audio track as a separate `.m4a` in the same Drive folder — costs you one click.

---

## Where everything goes

**Google Drive** — the demo recording. Create a folder, set it to "Anyone with the link can view", and keep the link.

**SharePoint Review 2 folder** — upload here:

https://amritavishwavidyapeetham-my.sharepoint.com/:f:/g/personal/t_senthilkumar_cb_amrita_edu/IgBWkUNsBc11R5tl6BRhxMYnAZRNYzXRZSVHTo7ejwcnGmg?e=fmXzxW

Create a folder named with your roll number, containing:

```
CB.SC.U4CSE24151/
├── ppt/                      the presentation
├── case study document/      the report, Word format
├── model paper/              the journal paper you referenced for experiments
└── readme.txt
```

`readme.txt` must contain:

```
a. Hosted application URL : <paste your deployed URL>
b. GitHub URL             : https://github.com/Key-to-Code/Fiduciary-Lens-Tax-Advisor-System
c. IEEE DataPort URL      : https://ieee-dataport.org//documents/fiduciary-lens-dataset
d. Demo recording         : <paste your Google Drive link>
e. Reference journal paper: https://www.ijsr.net/getabstract.php?paperid=ES24115141827
```

The model paper to place in `model paper/` is the IJSR paper above:
https://www.ijsr.net/getabstract.php?paperid=ES24115141827

Item (d) isn't in the stated list, but the recording lives in Drive and nothing else tells the instructor where to find it, so add the line.

**Deadline:** 22 October, Thursday, 8:00 pm.

---

## One thing to confirm with the instructor

On 10 September the message was *"Review 2 ppt will be individual ppt one for each member"*, but the 19 and 21 October instructions just say *"create a folder with ppt ---place the presentation"* (singular). These may not be the same requirement any more. Worth a message to ask whether it's one team deck or one per member — cheap to ask, expensive to get wrong the night before.
