"""A compact BM25 index.

Dense retrieval alone is weak on exactly the tokens that matter most in tax law:
provision numbers, form numbers and rupee limits ("80CCD", "Form No. 154").
BM25 catches those, so the two are fused in `retrieve.py`.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

_TOKEN_RE = r"(?u)\b\w[\w.\-]*\b"  # keeps "80CCD", "2025-26", "s.17" intact

K1 = 1.5
B = 0.75


class BM25Index:
    def __init__(self, vectorizer: CountVectorizer, matrix: sparse.csr_matrix):
        self.vectorizer = vectorizer
        counts = matrix.tocsc()
        n_docs = matrix.shape[0]
        doc_freq = np.diff(counts.indptr)
        self.idf = np.log(1 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5)).astype("float32")
        lengths = np.asarray(matrix.sum(axis=1)).ravel()
        avg_len = float(lengths.mean()) or 1.0

        # Precompute the BM25 term weight of every (doc, term) cell once.
        weighted = matrix.tocoo(copy=True).astype("float32")
        norm = K1 * (1 - B + B * lengths[weighted.row] / avg_len)
        weighted.data = weighted.data * (K1 + 1) / (weighted.data + norm)
        self.matrix = weighted.tocsc()

    @classmethod
    def build(cls, texts: list[str]) -> "BM25Index":
        vectorizer = CountVectorizer(lowercase=True, token_pattern=_TOKEN_RE, min_df=1)
        matrix = vectorizer.fit_transform(texts)
        return cls(vectorizer, matrix)

    def search(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (doc_indices, scores) for the k best-matching documents."""
        vocabulary = self.vectorizer.vocabulary_
        terms = re.findall(_TOKEN_RE, query.lower())
        columns = [vocabulary[t] for t in terms if t in vocabulary]
        if not columns:
            return np.empty(0, dtype=int), np.empty(0, dtype="float32")

        scores = np.zeros(self.matrix.shape[0], dtype="float32")
        for column in columns:
            block = self.matrix.getcol(column).tocoo()
            scores[block.row] += block.data * self.idf[column]

        k = min(k, int((scores > 0).sum()))
        if k == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype="float32")
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return top, scores[top]

    def save(self, path: Path) -> None:
        with open(path, "wb") as handle:
            pickle.dump({"vectorizer": self.vectorizer, "matrix": self.matrix,
                         "idf": self.idf}, handle)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with open(path, "rb") as handle:
            state = pickle.load(handle)
        index = cls.__new__(cls)
        index.vectorizer = state["vectorizer"]
        index.matrix = state["matrix"]
        index.idf = state["idf"]
        return index
