from __future__ import annotations

import functools
import numpy as np
from shared import config

config.apply_tls_workaround()


@functools.lru_cache(maxsize=1)
def _model(name: str):
    """Loads a sentence transformer model by name."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def encode_passages(texts: list[str], show_progress: bool = False) -> np.ndarray:
    """Encodes a list of passages into vectors."""
    vectors = _model(config.EMBED_MODEL).encode(
        texts,
        batch_size=config.EMBED_BATCH,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return vectors.astype("float32")


def encode_query(text: str) -> np.ndarray:
    """Encodes a query string into a vector."""
    vector = _model(config.EMBED_MODEL).encode(
        [config.QUERY_PREFIX + text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vector.astype("float32")