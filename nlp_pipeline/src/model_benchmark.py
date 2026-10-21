"""SLM vs BERT vs DistilBERT on the project's own statutory corpus.

The task is statutory-source classification: given a passage, decide which of the
three indexed documents it came from (Income-tax Act 2025, Income-tax Rules 2026,
Finance Act 2026). That is the labelling step the retriever's citation layer
depends on, so it is a task the product actually needs rather than a borrowed
benchmark.

Three models are compared on one split:
  slm         - a class-conditional trigram language model written from scratch
  bert        - bert-base-uncased, frozen encoder + logistic-regression head
  distilbert  - distilbert-base-uncased, frozen encoder + logistic-regression head

The transformers are used as linear probes, not fine-tuned. That is a deliberate
choice: it runs on CPU in minutes and it isolates the quality of the pretrained
representation, which is what the comparison is about. It is reported as a probe
everywhere so the numbers are not mistaken for fine-tuned ones.

Run:  python nlp_pipeline/src/model_benchmark.py
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from shared import config

config.apply_tls_workaround()

SEED = 20261021
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "experiments"
RESULTS_JSON = RESULTS_DIR / "model_benchmark.json"
CHART_PNG = RESULTS_DIR / "model_comparison.png"

# Short, stable class names. The corpus filenames are long and dated.
LABELS = {
    "Income-tax-Act-2025": "Act",
    "Income-tax-Rules-2026": "Rules",
    "Finance-Act-2026": "Finance Act",
}
TEST_FRACTION = 0.25
MAX_TOKENS = 256  # transformer truncation length


# --------------------------------------------------------------- a. the dataset

def _label_for(document_name: str) -> str | None:
    for prefix, label in LABELS.items():
        if document_name.startswith(prefix):
            return label
    return None


@dataclass
class Dataset:
    train_text: list[str]
    train_y: list[str]
    test_text: list[str]
    test_y: list[str]
    train_tokens: list[list[str]] = field(default_factory=list)
    test_tokens: list[list[str]] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    per_class: int = 0
    raw_counts: dict = field(default_factory=dict)


def preprocess(text: str) -> list[str]:
    """NLTK pipeline: tokenise, drop stop-words and punctuation, lemmatise.

    Lemmatising rather than stemming keeps statutory vocabulary readable
    ("deductions" -> "deduction", not "deduct"), which matters because the XAI
    stage reports these tokens back to a human.
    """
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    stop = set(stopwords.words("english"))
    lemmatiser = WordNetLemmatizer()

    out = []
    for token in word_tokenize(text.lower()):
        if not any(ch.isalnum() for ch in token):
            continue
        if token in stop:
            continue
        out.append(lemmatiser.lemmatize(token))
    return out


def load_dataset() -> Dataset:
    raw = json.loads(Path(config.KB_JSON).read_text(encoding="utf-8"))

    by_class: dict[str, list[str]] = defaultdict(list)
    for record in raw:
        label = _label_for(record["document_name"])
        if label and record["content"].strip():
            by_class[label].append(record["content"])

    raw_counts = {k: len(v) for k, v in by_class.items()}

    # Balance to the smallest class so accuracy is not inflated by the majority
    # document, which supplies over half the corpus on its own.
    per_class = min(len(v) for v in by_class.values())
    rng = random.Random(SEED)

    train_text, train_y, test_text, test_y = [], [], [], []
    for label in sorted(by_class):
        items = by_class[label][:]
        rng.shuffle(items)
        items = items[:per_class]
        cut = int(per_class * (1 - TEST_FRACTION))
        train_text += items[:cut]
        train_y += [label] * cut
        test_text += items[cut:]
        test_y += [label] * (per_class - cut)

    data = Dataset(train_text, train_y, test_text, test_y,
                   classes=sorted(by_class), per_class=per_class,
                   raw_counts=raw_counts)
    print(f"  preprocessing {len(train_text) + len(test_text)} passages with NLTK ...")
    data.train_tokens = [preprocess(t) for t in train_text]
    data.test_tokens = [preprocess(t) for t in test_text]
    return data


# ------------------------------------------------------- b. the SLM, from scratch

class TrigramSLM:
    """A trigram language model with add-k smoothing.

    Architecture: counts of (w1,w2) -> w3 over the preprocessed token stream,
    backing off to bigram and unigram counts. Vocabulary is capped and rare
    tokens map to <unk>. One model is trained per class; classification picks the
    class whose model assigns the passage the highest log-likelihood, so the same
    object serves as both a language model (perplexity, generation) and a
    classifier (accuracy, precision, recall, F1).

    Parameters: order=3, add_k=0.1, vocab_size=8000.
    """

    def __init__(self, order: int = 3, add_k: float = 0.1, vocab_size: int = 8000):
        self.order = order
        self.add_k = add_k
        self.vocab_size = vocab_size
        self.vocab: set[str] = set()
        self.trigram: dict = defaultdict(Counter)
        self.bigram: dict = defaultdict(Counter)
        self.unigram: Counter = Counter()
        self.total = 0

    @property
    def n_params(self) -> int:
        """Stored conditional-count cells, the SLM's analogue of a weight count."""
        return (sum(len(v) for v in self.trigram.values())
                + sum(len(v) for v in self.bigram.values())
                + len(self.unigram))

    def _pad(self, tokens: list[str]) -> list[str]:
        mapped = [t if t in self.vocab else "<unk>" for t in tokens]
        return ["<s>", "<s>"] + mapped + ["</s>"]

    def fit(self, token_lists: list[list[str]]) -> "TrigramSLM":
        counts = Counter(t for tokens in token_lists for t in tokens)
        self.vocab = {w for w, _ in counts.most_common(self.vocab_size)}
        self.vocab |= {"<unk>", "<s>", "</s>"}
        for tokens in token_lists:
            padded = self._pad(tokens)
            for i in range(2, len(padded)):
                w1, w2, w3 = padded[i - 2], padded[i - 1], padded[i]
                self.trigram[(w1, w2)][w3] += 1
                self.bigram[w2][w3] += 1
                self.unigram[w3] += 1
                self.total += 1
        return self

    def _logp(self, w1: str, w2: str, w3: str) -> float:
        v = len(self.vocab)
        tri = self.trigram.get((w1, w2))
        if tri:
            return math.log((tri.get(w3, 0) + self.add_k) / (sum(tri.values()) + self.add_k * v))
        bi = self.bigram.get(w2)
        if bi:
            return math.log((bi.get(w3, 0) + self.add_k) / (sum(bi.values()) + self.add_k * v))
        return math.log((self.unigram.get(w3, 0) + self.add_k) / (self.total + self.add_k * v))

    def log_likelihood(self, tokens: list[str]) -> tuple[float, int]:
        padded = self._pad(tokens)
        total = 0.0
        for i in range(2, len(padded)):
            total += self._logp(padded[i - 2], padded[i - 1], padded[i])
        return total, len(padded) - 2

    def perplexity(self, token_lists: list[list[str]]) -> float:
        log_sum, n = 0.0, 0
        for tokens in token_lists:
            ll, count = self.log_likelihood(tokens)
            log_sum += ll
            n += count
        return float(math.exp(-log_sum / max(n, 1)))

    def generate(self, prompt: list[str], max_new: int = 30) -> list[str]:
        """Greedy continuation, used only to give BLEU/ROUGE something to score."""
        mapped = [t if t in self.vocab else "<unk>" for t in prompt]
        history = ["<s>", "<s>"] + mapped
        out = []
        for _ in range(max_new):
            candidates = self.trigram.get((history[-2], history[-1]))
            if not candidates:
                candidates = self.bigram.get(history[-1])
            if not candidates:
                break
            nxt = max(candidates.items(), key=lambda kv: kv[1])[0]
            if nxt == "</s>":
                break
            out.append(nxt)
            history.append(nxt)
        return out


class ClassConditionalSLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.models: dict[str, TrigramSLM] = {}
        self.classes: list[str] = []

    def fit(self, token_lists: list[list[str]], y: list[str]) -> "ClassConditionalSLM":
        grouped = defaultdict(list)
        for tokens, label in zip(token_lists, y):
            grouped[label].append(tokens)
        self.classes = sorted(grouped)
        for label in self.classes:
            self.models[label] = TrigramSLM(**self.kwargs).fit(grouped[label])
        return self

    def predict(self, token_lists: list[list[str]]) -> list[str]:
        out = []
        for tokens in token_lists:
            best, best_score = None, -math.inf
            for label, model in self.models.items():
                ll, n = model.log_likelihood(tokens)
                score = ll / max(n, 1)  # length-normalised, passages vary in length
                if score > best_score:
                    best, best_score = label, score
            out.append(best)
        return out

    @property
    def n_params(self) -> int:
        return sum(m.n_params for m in self.models.values())


# --------------------------------------------------- c. metrics (no extra deps)

def classification_metrics(y_true: list[str], y_pred: list[str],
                           classes: list[str]) -> dict:
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average="macro", zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "confusion_labels": classes,
    }


def rouge_l(reference: list[str], hypothesis: list[str]) -> float:
    """ROUGE-L F-measure from the longest common subsequence."""
    if not reference or not hypothesis:
        return 0.0
    prev = [0] * (len(hypothesis) + 1)
    for r in reference:
        curr = [0]
        for j, h in enumerate(hypothesis):
            curr.append(prev[j] + 1 if r == h else max(curr[j], prev[j + 1]))
        prev = curr
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(hypothesis), lcs / len(reference)
    return float(2 * p * r / (p + r))


def generation_metrics(slm: ClassConditionalSLM, data: Dataset,
                       prompt_len: int = 12, continue_len: int = 20) -> dict:
    """BLEU and ROUGE-L for the SLM's greedy continuation of held-out passages."""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    smoothing = SmoothingFunction().method1
    bleus, rouges, used = [], [], 0
    for tokens, label in zip(data.test_tokens, data.test_y):
        if len(tokens) < prompt_len + continue_len:
            continue
        prompt = tokens[:prompt_len]
        reference = tokens[prompt_len:prompt_len + continue_len]
        hypothesis = slm.models[label].generate(prompt, max_new=continue_len)
        if not hypothesis:
            continue
        bleus.append(sentence_bleu([reference], hypothesis,
                                   smoothing_function=smoothing))
        rouges.append(rouge_l(reference, hypothesis))
        used += 1
    return {
        "bleu": round(float(np.mean(bleus)), 4) if bleus else None,
        "rouge_l": round(float(np.mean(rouges)), 4) if rouges else None,
        "generated_samples": used,
        "prompt_tokens": prompt_len,
        "continuation_tokens": continue_len,
    }


# ------------------------------------------- d. BERT and DistilBERT linear probes

def embed(model_name: str, texts: list[str], batch_size: int = 16):
    """Mean-pooled final-layer embeddings from a frozen encoder."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokeniser = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokeniser(batch, padding=True, truncation=True,
                                max_length=MAX_TOKENS, return_tensors="pt").to(device)
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vectors.append(pooled.cpu().numpy())
    n_params = sum(p.numel() for p in model.parameters())
    del model
    return np.vstack(vectors), n_params, device


def run_transformer_probe(model_name: str, data: Dataset) -> dict:
    from sklearn.linear_model import LogisticRegression

    t0 = time.perf_counter()
    train_x, n_params, device = embed(model_name, data.train_text)
    embed_train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    head = LogisticRegression(max_iter=2000, random_state=SEED)
    head.fit(train_x, data.train_y)
    head_fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    test_x, _, _ = embed(model_name, data.test_text)
    predictions = head.predict(test_x).tolist()
    inference_s = time.perf_counter() - t0

    result = classification_metrics(data.test_y, predictions, data.classes)
    result.update({
        "encoder_params": n_params,
        "head_params": int(head.coef_.size + head.intercept_.size),
        "train_seconds": round(embed_train_s + head_fit_s, 2),
        "inference_seconds": round(inference_s, 2),
        "inference_ms_per_passage": round(inference_s * 1000 / len(data.test_text), 2),
        "device": device,
        "perplexity": None,
        "bleu": None,
        "rouge_l": None,
    })
    return result, head


# ----------------------------------------------------------------------- e. XAI

def occlusion_attribution(slm: ClassConditionalSLM, tokens: list[str],
                          predicted: str, top_n: int = 8) -> list[dict]:
    """Leave-one-token-out attribution for the SLM's class decision.

    Drop each token in turn and measure how far the margin between the predicted
    class and the best rival falls. A large drop means the token was carrying the
    decision. This is model-agnostic occlusion, the same idea LIME approximates
    with a surrogate, but computed exactly here because the model is cheap.
    """
    def margin(seq: list[str]) -> float:
        scores = {}
        for label, model in slm.models.items():
            ll, n = model.log_likelihood(seq)
            scores[label] = ll / max(n, 1)
        rival = max((s for l, s in scores.items() if l != predicted), default=0.0)
        return scores[predicted] - rival

    base = margin(tokens)
    contributions = []
    seen = set()
    for i, token in enumerate(tokens):
        if token in seen:
            continue
        seen.add(token)
        without = tokens[:i] + tokens[i + 1:]
        contributions.append({"token": token,
                              "contribution": round(base - margin(without), 5)})
    contributions.sort(key=lambda c: -abs(c["contribution"]))
    return contributions[:top_n]


def build_xai_examples(slm: ClassConditionalSLM, data: Dataset,
                       n_examples: int = 3) -> list[dict]:
    predictions = slm.predict(data.test_tokens)
    examples = []
    rng = random.Random(SEED)
    indices = list(range(len(data.test_tokens)))
    rng.shuffle(indices)
    for i in indices:
        if len(examples) >= n_examples:
            break
        tokens = data.test_tokens[i][:60]  # cap: attribution is O(n^2)
        if len(tokens) < 20:
            continue
        examples.append({
            "excerpt": data.test_text[i][:220].replace("\n", " ").strip(),
            "true_label": data.test_y[i],
            "predicted_label": predictions[i],
            "top_tokens": occlusion_attribution(slm, tokens, predictions[i]),
        })
    return examples


# ------------------------------------------------------------ f/g. run and report

def chart(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["display_name"] for r in results["models"]]
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]

    x = np.arange(len(metrics))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, model in enumerate(results["models"]):
        values = [model[m] for m in metrics]
        bars = ax.bar(x + i * width - 0.4 + width / 2, values, width,
                      label=model["display_name"])
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Macro score")
    ax.set_title("Statutory-source classification on the project corpus")
    ax.legend(loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHART_PNG, dpi=160)
    print(f"  chart -> {CHART_PNG}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and preprocessing the corpus ...")
    data = load_dataset()
    print(f"  {len(data.classes)} classes, {data.per_class} passages each "
          f"(balanced from {data.raw_counts})")
    print(f"  train {len(data.train_text)} / test {len(data.test_text)}")

    models = []

    print("\nTraining the trigram SLM ...")
    t0 = time.perf_counter()
    slm = ClassConditionalSLM(order=3, add_k=0.1, vocab_size=8000)
    slm.fit(data.train_tokens, data.train_y)
    slm_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    slm_predictions = slm.predict(data.test_tokens)
    slm_inference = time.perf_counter() - t0

    slm_result = classification_metrics(data.test_y, slm_predictions, data.classes)
    pooled = TrigramSLM(order=3, add_k=0.1, vocab_size=8000).fit(data.train_tokens)
    slm_result.update({
        "key": "slm",
        "display_name": "Trigram SLM (from scratch)",
        "kind": "Statistical language model",
        "encoder_params": slm.n_params,
        "head_params": 0,
        "perplexity": round(pooled.perplexity(data.test_tokens), 2),
        "train_seconds": round(slm_train, 2),
        "inference_seconds": round(slm_inference, 2),
        "inference_ms_per_passage": round(slm_inference * 1000 / len(data.test_text), 2),
        "device": "cpu",
        "notes": "Class-conditional trigram LM, add-k smoothing (k=0.1), "
                 "vocabulary capped at 8,000 types. Classifies by highest "
                 "length-normalised log-likelihood.",
    })
    slm_result.update(generation_metrics(slm, data))
    models.append(slm_result)
    print(f"  accuracy {slm_result['accuracy']:.3f} | "
          f"perplexity {slm_result['perplexity']} | "
          f"BLEU {slm_result['bleu']} | ROUGE-L {slm_result['rouge_l']}")

    for key, model_name, display in [
        ("bert", "bert-base-uncased", "BERT (frozen + linear probe)"),
        ("distilbert", "distilbert-base-uncased", "DistilBERT (frozen + linear probe)"),
    ]:
        print(f"\nRunning {model_name} ...")
        result, _ = run_transformer_probe(model_name, data)
        result.update({
            "key": key,
            "display_name": display,
            "kind": "Pretrained transformer encoder",
            "hf_model": model_name,
            "notes": f"Mean-pooled final-layer embeddings from a frozen "
                     f"{model_name}, classified by multinomial logistic "
                     f"regression. Not fine-tuned. Truncated at {MAX_TOKENS} tokens.",
        })
        models.append(result)
        print(f"  accuracy {result['accuracy']:.3f} | "
              f"F1 {result['f1_macro']:.3f} | "
              f"{result['inference_ms_per_passage']} ms/passage | "
              f"{result['encoder_params'] / 1e6:.1f}M params")

    print("\nComputing XAI attributions ...")
    xai = build_xai_examples(slm, data)

    results = {
        "task": "Statutory-source classification (3 classes)",
        "task_description": (
            "Given a statutory passage, predict which indexed document it came "
            "from: the Income-tax Act 2025, the Income-tax Rules 2026, or the "
            "Finance Act 2026."),
        "dataset": {
            "source": "rag_model/data/rag_knowledge_base.json",
            "classes": data.classes,
            "passages_per_class": data.per_class,
            "train_size": len(data.train_text),
            "test_size": len(data.test_text),
            "test_fraction": TEST_FRACTION,
            "raw_class_counts": data.raw_counts,
            "preprocessing": ("NLTK word_tokenize, punctuation and 198 English "
                              "stop-words removed, WordNet lemmatisation"),
            "balanced": True,
            "seed": SEED,
        },
        "models": models,
        "xai": {
            "algorithm": "Leave-one-token-out occlusion attribution",
            "applies_to": "slm",
            "description": (
                "Each token is removed in turn and the drop in the margin "
                "between the predicted class and its closest rival is recorded. "
                "Tokens with the largest drop are the ones carrying the "
                "decision."),
            "examples": xai,
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    RESULTS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  results -> {RESULTS_JSON}")
    chart(results)

    print("\n" + "-" * 76)
    header = f"{'Model':<36}{'Acc':>7}{'F1':>7}{'PPL':>9}{'ms/psg':>9}{'Params':>9}"
    print(header)
    print("-" * 76)
    for m in models:
        ppl = f"{m['perplexity']:.1f}" if m["perplexity"] else "n/a"
        params = f"{(m['encoder_params'] + m['head_params']) / 1e6:.1f}M"
        print(f"{m['display_name']:<36}{m['accuracy']:>7.3f}{m['f1_macro']:>7.3f}"
              f"{ppl:>9}{m['inference_ms_per_passage']:>9.1f}{params:>9}")
    print("-" * 76)


if __name__ == "__main__":
    main()
