"""Thin wrapper around the sentence-transformers encoder.

Loading the model is slow, so it is cached per process. Queries get the bge
instruction prefix; passages must not, or the two spaces stop lining up.
"""

from __future__ import annotations

import functools

import numpy as np

from . import config

config.apply_tls_workaround()


@functools.lru_cache(maxsize=1)
def _model(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def encode_passages(texts: list[str], show_progress: bool = False) -> np.ndarray:
    vectors = _model(config.EMBED_MODEL).encode(
        texts,
        batch_size=config.EMBED_BATCH,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return vectors.astype("float32")


def encode_query(text: str) -> np.ndarray:
    vector = _model(config.EMBED_MODEL).encode(
        [config.QUERY_PREFIX + text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vector.astype("float32")
