/* Verified project details and evaluation snapshots for the public information pages. */
window.FIDUCIARY_PROJECT_DATA = Object.freeze({
  context: {
    group: "11",
    category: "Software",
    courseCode: "23CSE471",
    courseName: "Natural Language Processing",
    repository: "https://github.com/Key-to-Code/Fiduciary-Lens-Tax-Advisor-System",
    productDemo: null,
  },
  faculty: {
    name: "Dr. Senthil Kumar T.",
    role: "Professor",
    department: "Department of Computer Science and Engineering",
    school: "School of Computing, Coimbatore",
    qualification: "Ph.D",
    email: "t_senthilkumar@cb.amrita.edu",
  },
  members: [
    {
      rollNo: "CB.SC.U4CSE24107",
      name: "Anuja Ranganathan",
      email: "cb.sc.u4cse24107@cb.students.amrita.edu",
      phone: "8925477535",
      github: "https://github.com/Anuja1307",
    },
    {
      rollNo: "CB.SC.U4CSE24118",
      name: "Ganesh Sarathi",
      email: "cb.sc.u4cse24118@cb.students.amrita.edu",
      phone: "8778746516",
      github: "https://github.com/GaneshSarathi19",
    },
    {
      rollNo: "CB.SC.U4CSE24119",
      name: "Gaurav Chandrashekar",
      email: "cb.sc.u4cse24119@cb.students.amrita.edu",
      phone: "8825984586",
      github: "https://github.com/Key-to-Code",
    },
    {
      rollNo: "CB.SC.U4CSE24151",
      name: "Sudharshan Sambathkumar",
      email: "cb.sc.u4cse24151@cb.students.amrita.edu",
      phone: "7418523795",
      github: "https://github.com/SL8Entropy",
    },
  ],
  evaluation: {
    verifiedOn: "20 October 2026",
    retrieval: {
      indexChunks: 3496,
      embeddingModel: "BAAI/bge-small-en-v1.5",
      retrievalMethod: "Hybrid dense + BM25 retrieval",
      questions: 20,
      k: 6,
      recallCount: 20,
      recall: 1,
      mrr: 0.917,
      offTopicQuestions: 5,
      refusedOffTopic: 5,
      refusalRate: 1,
    },
    models: [
      {
        role: "Dense retrieval · deep learning",
        name: "BAAI/bge-small-en-v1.5",
        runtime: false,
        measured: true,
        measurementLabel: "Hybrid benchmark",
        description: "Embeds the user question for semantic search. The published benchmark measures the complete dense + BM25 retrieval pipeline, not an isolated embedding-only run.",
        measures: ["Hybrid Recall@6: 100% (20/20)", "Hybrid MRR: 0.917", "Off-topic refusal: 100% (5/5)"],
      },
      {
        role: "Answer generation · LLM",
        name: "Configured by the backend",
        runtime: true,
        measured: false,
        description: "The active provider and model are selected by backend environment settings. Q&A and document summarization/entity extraction share this provider path; the repository has no live-model quality benchmark.",
        measures: ["Quality / faithfulness: not benchmarked", "Latency distribution: not benchmarked", "The 5 passing harness tests use fixed, mocked answer data"],
      },
      {
        role: "Lexical retrieval · not deep learning",
        name: "BM25",
        runtime: false,
        measured: false,
        description: "Retrieves exact terms such as section numbers and form identifiers, then contributes to the measured hybrid retrieval results.",
        measures: ["No standalone BM25 benchmark is recorded", "Fusion weight: 35% by default"],
      },
      {
        role: "Document entity extraction · rules",
        name: "Regex extraction + configured LLM",
        runtime: false,
        measured: false,
        description: "PAN, TAN, GSTIN and amounts use deterministic patterns; complex organization and deduction extraction uses the configured language model.",
        measures: ["Entity precision / recall: not benchmarked"],
      },
    ,
      {
        "role": "Deep-learning comparison · statistical LM",
        "name": "Trigram SLM (from scratch)",
        "runtime": false,
        "measured": true,
        "measurementLabel": "Statutory-source classification",
        "description": "Class-conditional trigram LM, add-k smoothing (k=0.1), vocabulary capped at 8,000 types. Classifies by highest length-normalised log-likelihood.",
        "measures": [
          "Accuracy: 0.884",
          "Precision (macro): 0.888",
          "Recall (macro): 0.884",
          "F1 (macro): 0.884",
          "Perplexity: 552.94",
          "BLEU: 0.0361",
          "ROUGE-L: 0.1321",
          "Parameters: 91,973 count cells",
          "Training: 0.15 s on CPU",
          "Inference: 0.38 ms/passage"
        ]
      },
      {
        "role": "Deep-learning comparison · transformer",
        "name": "BERT (frozen + linear probe)",
        "runtime": false,
        "measured": true,
        "measurementLabel": "Statutory-source classification",
        "description": "Mean-pooled final-layer embeddings from a frozen bert-base-uncased, classified by multinomial logistic regression. Not fine-tuned. Truncated at 256 tokens.",
        "measures": [
          "Accuracy: 0.849",
          "Precision (macro): 0.851",
          "Recall (macro): 0.849",
          "F1 (macro): 0.850",
          "Parameters: 109.48 M",
          "Training: 35.18 s on CUDA",
          "Inference: 23.51 ms/passage"
        ]
      },
      {
        "role": "Deep-learning comparison · transformer",
        "name": "DistilBERT (frozen + linear probe)",
        "runtime": false,
        "measured": true,
        "measurementLabel": "Statutory-source classification",
        "description": "Mean-pooled final-layer embeddings from a frozen distilbert-base-uncased, classified by multinomial logistic regression. Not fine-tuned. Truncated at 256 tokens.",
        "measures": [
          "Accuracy: 0.814",
          "Precision (macro): 0.819",
          "Recall (macro): 0.814",
          "F1 (macro): 0.816",
          "Parameters: 66.37 M",
          "Training: 8.55 s on CUDA",
          "Inference: 18.21 ms/passage"
        ]
      },
    ],
    xai: {
      name: "Hybrid Retrieval Score Attribution (HRSA)",
      description: "For each retrieved passage, the UI shows how much of its ranking score came from semantic similarity and exact-term matching, including a statute applicability penalty where one was applied.",
      formula: "score = (dense weight × normalized cosine + BM25 weight × normalized BM25) × statute weight",
    },
    xaiOcclusion: {
      "name": "Leave-one-token-out occlusion attribution",
      "appliesTo": "SLM, BERT and DistilBERT classifiers",
      "description": "Each token is removed in turn and the drop in the margin between the predicted class and its closest rival is recorded. Tokens with the largest drop are the ones carrying the decision.",
      "examples": [
        {
          "excerpt": "Eligible international transactions for safe harbour. 88. An \"eligible international transaction\" means an international transaction between the eligible assessee and its associated enterprise, one of which is necessaril",
          "trueLabel": "Rules",
          "predictedLabel": "Rules",
          "topTokens": [
            {
              "token": "harbour",
              "contribution": 0.14607
            },
            {
              "token": "assessee",
              "contribution": -0.12997
            },
            {
              "token": "loan",
              "contribution": 0.11436
            },
            {
              "token": "research",
              "contribution": 0.09548
            },
            {
              "token": "88.",
              "contribution": 0.08596
            },
            {
              "token": "necessarily",
              "contribution": -0.07673
            }
          ]
        },
        {
          "excerpt": "Furnishing of authorisation and maintenance of documents, etc. for the purposes of section 176. 125. (1) For the purposes of section 176(3)(a), the authorisation to be submitted by the assessee, shall be in Form No. 61.",
          "trueLabel": "Rules",
          "predictedLabel": "Act",
          "topTokens": [
            {
              "token": "125",
              "contribution": -0.10776
            },
            {
              "token": "61",
              "contribution": -0.0858
            },
            {
              "token": "filled",
              "contribution": 0.07515
            },
            {
              "token": "1",
              "contribution": -0.06308
            },
            {
              "token": "3",
              "contribution": 0.05889
            },
            {
              "token": "jurisdiction",
              "contribution": -0.05005
            }
          ]
        },
        {
          "excerpt": "made to the Transfer Pricing Officer for determining the arm's length price under section 166(1), shall be extended by an additional period of twelve months. (b) In terms of provisions of sub-section (1) [Table: Sl. Nos.",
          "trueLabel": "Act",
          "predictedLabel": "Act",
          "topTokens": [
            {
              "token": "'s",
              "contribution": 0.22102
            },
            {
              "token": "additional",
              "contribution": -0.14313
            },
            {
              "token": "no",
              "contribution": -0.13606
            },
            {
              "token": "5",
              "contribution": -0.11693
            },
            {
              "token": "length",
              "contribution": 0.10067
            },
            {
              "token": "3",
              "contribution": 0.09359
            }
          ]
        }
      ]
    },
    modelComparison: {
      "task": "Statutory-source classification (3 classes)",
      "taskDescription": "Given a statutory passage, predict which indexed document it came from: the Income-tax Act 2025, the Income-tax Rules 2026, or the Finance Act 2026.",
      "dataset": {
        "source": "Project tax-law corpus (rag_model/data/rag_knowledge_base.json)",
        "classes": [
          "Act",
          "Finance Act",
          "Rules"
        ],
        "passagesPerClass": 344,
        "trainSize": 774,
        "testSize": 258,
        "preprocessing": "NLTK word_tokenize, punctuation and 198 English stop-words removed, WordNet lemmatisation",
        "balanced": true
      },
      "rows": [
        {
          "metric": "Accuracy",
          "slm": "0.884",
          "bert": "0.849",
          "distilbert": "0.814"
        },
        {
          "metric": "Precision (macro)",
          "slm": "0.888",
          "bert": "0.851",
          "distilbert": "0.819"
        },
        {
          "metric": "Recall (macro)",
          "slm": "0.884",
          "bert": "0.849",
          "distilbert": "0.814"
        },
        {
          "metric": "F1 (macro)",
          "slm": "0.884",
          "bert": "0.850",
          "distilbert": "0.816"
        },
        {
          "metric": "Perplexity",
          "slm": "552.94",
          "bert": "n/a",
          "distilbert": "n/a"
        },
        {
          "metric": "BLEU",
          "slm": "0.0361",
          "bert": "n/a",
          "distilbert": "n/a"
        },
        {
          "metric": "ROUGE-L",
          "slm": "0.1321",
          "bert": "n/a",
          "distilbert": "n/a"
        },
        {
          "metric": "Parameters",
          "slm": "91,973 counts",
          "bert": "109.48 M",
          "distilbert": "66.36 M"
        },
        {
          "metric": "Training time",
          "slm": "0.15 s",
          "bert": "35.18 s",
          "distilbert": "8.55 s"
        },
        {
          "metric": "Inference",
          "slm": "0.38 ms",
          "bert": "23.51 ms",
          "distilbert": "18.21 ms"
        }
      ],
      "inference": "The trigram SLM wins every classification metric while using about one eleven-hundredth of BERT's parameters and running roughly 60 times faster. The three classes are separable on surface vocabulary, which is what an n-gram model captures and what a mean-pooled sentence embedding deliberately discards. This is a narrow result for this task, not a general ranking: the transformers were probed rather than fine-tuned, and a shallow count-based model offers nothing on the paraphrase matching the retrieval path depends on.",
      "caveat": "BERT and DistilBERT are used as frozen encoders with a logistic-regression head, not fine-tuned. Perplexity, BLEU and ROUGE-L apply only to the SLM, the one generative model in the comparison; masked-LM pseudo-perplexity is not the same quantity and is omitted rather than printed in the same column.",
      "chart": "../experiments/model_comparison.png",
      "generatedAt": "2026-10-21 17:57:08"
    },
    harness: {
      command: ".venv/Scripts/python.exe -m pytest -q tests/test_llm_evaluation.py",
      testCount: 5,
      passed: 5,
      failed: 0,
      usesLiveModel: false,
      checks: [
        "Detects an answer that is unrelated to the retrieved tax-law context",
        "Scores a grounded answer against its retrieved context",
        "Ranks a relevant answer above an unrelated answer",
        "Accepts a fixed Section 80D sample with citation and grounding checks",
        "Rejects an ungrounded answer to an out-of-scope question",
      ],
    },
    caveat: "These are narrow regression checks. Retrieval was measured on 20 hand-authored in-scope questions and five unrelated controls against the current index; the result does not establish legal correctness or generalize to every tax question. The five LLM tests use fixed examples and a mocked answer payload, and the scoring functions use token overlap and string markers. No live-model answer quality or latency, standalone embedding/BM25 performance, or document-extraction precision/recall has been measured. A live LLM benchmark needs a reviewed question set, reference answers and citations, real provider calls, and reported quality and latency metrics.",
    historicalNote: "A comment in shared/config.py describes an earlier, retired IndicBERT dense-only run as both 35% recall@6 and 1/20. Those figures conflict, so that historical result is excluded from the current benchmark.",
  },
});
