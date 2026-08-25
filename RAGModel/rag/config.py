"""Central configuration. Everything the pipeline needs to find or tune lives here."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- data ---------------------------------------------------------------
KB_JSON = Path(os.getenv("KB_JSON", ROOT / "Data" / "rag_knowledge_base.json"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "index"))

# --- retrieval ----------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# bge models are trained with an instruction prefix on the query side only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))

# The Finance Act sets rates for both the superseded Income-tax Act, 1961 and the
# Income-tax Act, 2025 that this corpus is built on, in near-identical wording -
# embeddings cannot tell the two tables apart, so the 1961 one often outranks the
# right one. Demote passages governed by an Act other than the principal statute
# rather than dropping them, so an explicit question about the old Act still works.
PRINCIPAL_ACT_YEAR = os.getenv("PRINCIPAL_ACT_YEAR", "2025")
SUPERSEDED_PENALTY = float(os.getenv("SUPERSEDED_PENALTY", "0.55"))

TOP_K = int(os.getenv("TOP_K", "6"))          # passages handed to the LLM
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "30"))  # per-retriever candidate pool
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.65"))  # vs. lexical, in fusion
# Below this cosine we treat retrieval as "nothing relevant found" and refuse.
# Calibrated on this KB: on-topic tax questions land at 0.70-0.82, off-topic
# controls (football, baking, programming) at 0.48-0.54.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.60"))

# --- generation ---------------------------------------------------------
# "auto" probes ollama, then a cached local model, then openai, then falls back
# to extractive (no LLM at all).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# Runs in-process via transformers. ~3.1 GB in fp16, so it fits a 6 GB GPU.
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "800"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

# --- TLS ----------------------------------------------------------------
# This machine sits behind a TLS-intercepting proxy whose root CA is in the
# Windows store but not in certifi's bundle. If we exported that store to
# win-ca-bundle.pem, point Python's HTTPS stack at it.
CA_BUNDLE = ROOT / "win-ca-bundle.pem"


def apply_tls_workaround() -> None:
    if CA_BUNDLE.exists():
        for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            os.environ.setdefault(var, str(CA_BUNDLE))
    # transformers probes TensorFlow on import; Keras 3 makes that probe explode.
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
