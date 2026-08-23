"""Build, save and load the two indexes the retriever searches.

Both are derived artifacts: delete `index/` and `build_index.py` reproduces it
from `Data/rag_knowledge_base.json`. Nothing here belongs in git.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from . import config, embedder
from .kb import Chunk, index_text, load_chunks
from .lexical import BM25Index

_MANIFEST = "manifest.json"
_VECTORS = "dense.faiss"
_LEXICAL = "bm25.pkl"


@dataclass
class SearchIndex:
    chunks: list[Chunk]
    dense: faiss.Index
    lexical: BM25Index
    embed_model: str


def build(kb_path: Path | None = None, index_dir: Path | None = None,
          verbose: bool = True) -> SearchIndex:
    index_dir = Path(index_dir or config.INDEX_DIR)
    index_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(kb_path)
    texts = [index_text(chunk) for chunk in chunks]
    if verbose:
        print(f"Loaded {len(chunks)} chunks from {kb_path or config.KB_JSON}")
        print(f"Embedding with {config.EMBED_MODEL} ...")

    vectors = embedder.encode_passages(texts, show_progress=verbose)
    dense = faiss.IndexFlatIP(vectors.shape[1])  # vectors are L2-normalised -> cosine
    dense.add(vectors)
    faiss.write_index(dense, str(index_dir / _VECTORS))

    if verbose:
        print("Building BM25 index ...")
    lexical = BM25Index.build(texts)
    lexical.save(index_dir / _LEXICAL)

    (index_dir / _MANIFEST).write_text(json.dumps({
        "embed_model": config.EMBED_MODEL,
        "kb_path": str(kb_path or config.KB_JSON),
        "n_chunks": len(chunks),
        "dim": int(vectors.shape[1]),
        "documents": sorted({c.document_name for c in chunks}),
    }, indent=2), encoding="utf-8")

    if verbose:
        print(f"Index written to {index_dir}")
    return SearchIndex(chunks, dense, lexical, config.EMBED_MODEL)


def load(index_dir: Path | None = None, kb_path: Path | None = None) -> SearchIndex:
    index_dir = Path(index_dir or config.INDEX_DIR)
    manifest_path = index_dir / _MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No index at {index_dir}. Build it first:  python build_index.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    chunks = load_chunks(kb_path or manifest["kb_path"])
    if len(chunks) != manifest["n_chunks"]:
        raise RuntimeError(
            f"Knowledge base changed ({len(chunks)} chunks vs {manifest['n_chunks']} "
            "indexed). Re-run: python build_index.py"
        )
    if manifest["embed_model"] != config.EMBED_MODEL:
        raise RuntimeError(
            f"Index was built with {manifest['embed_model']} but EMBED_MODEL is "
            f"{config.EMBED_MODEL}. Changing the model requires a full re-index."
        )

    dense = faiss.read_index(str(index_dir / _VECTORS))
    lexical = BM25Index.load(index_dir / _LEXICAL)
    return SearchIndex(chunks, dense, lexical, manifest["embed_model"])
