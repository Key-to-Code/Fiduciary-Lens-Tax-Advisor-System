"""The end-to-end pipeline: question in, grounded and cited answer out."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

from . import index as index_module, prompt, retrieve
from .llm import ExtractiveProvider, Provider, get_provider
from .retrieve import Hit


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit] = field(default_factory=list)
    grounded: bool = True
    provider: str = ""
    latency_ms: int = 0

    @property
    def sources(self) -> list[dict]:
        return [
            {
                "n": position,
                "citation": hit.chunk.citation,
                "short": hit.chunk.short_citation,
                "document": hit.chunk.document_name,
                "as_of": hit.chunk.source_note,
                "chunk_id": hit.chunk.chunk_id,
                "score": hit.score,
                "cosine": hit.dense_score,
            }
            for position, hit in enumerate(self.hits, start=1)
        ]


class TaxQA:
    """Holds the loaded index and provider so repeated questions stay cheap."""

    def __init__(self, provider: Provider | str | None = None, index_dir=None):
        self.index = index_module.load(index_dir)
        self.provider = provider if isinstance(provider, Provider) else get_provider(provider)
        self.last: Answer | None = None

    def retrieve(self, question: str, top_k: int | None = None) -> list[Hit]:
        return retrieve.search(self.index, question, top_k=top_k)

    def stream(self, question: str, history=None,
               top_k: int | None = None) -> Iterator[str]:
        """Yield the answer in pieces. Final state lands on `self.last`."""
        started = time.perf_counter()

        # A known coverage gap is caught before retrieval: these questions do
        # retrieve something plausible-looking, which is exactly the trap.
        uncovered = prompt.uncovered_topic(question)
        if uncovered:
            text = uncovered + "\n\n" + prompt.DISCLAIMER
            self.last = Answer(question, text, hits=[], grounded=False,
                               provider=self.provider.name,
                               latency_ms=int((time.perf_counter() - started) * 1000))
            yield text
            return

        hits = self.retrieve(question, top_k=top_k)

        if not retrieve.is_grounded(hits):
            # Cite-or-refuse: nothing retrieved clears the relevance bar, so we
            # never reach the model. This is the guardrail that actually holds.
            text = prompt.REFUSAL + "\n\n" + prompt.DISCLAIMER
            self.last = Answer(question, text, hits=[], grounded=False,
                               provider=self.provider.name,
                               latency_ms=int((time.perf_counter() - started) * 1000))
            yield text
            return

        system, user = prompt.build_messages(question, hits, history)
        collected: list[str] = []
        try:
            for piece in self.provider.generate(system, user):
                collected.append(piece)
                yield piece
        except Exception as exc:
            # A dead backend (quota, network, unloaded model) must not cost the
            # user their answer: the retrieved law is still sound, so fall back
            # to quoting it rather than surfacing a traceback.
            note = (f"\n\n_[{self.provider.name} backend failed: "
                    f"{type(exc).__name__}: {exc}]_\n\n")
            collected.append(note)
            yield note
            if not any(piece.strip() for piece in collected[:-1]):
                for piece in ExtractiveProvider().generate(system, user):
                    collected.append(piece)
                    yield piece

        tail = "\n\n" + prompt.DISCLAIMER
        yield tail
        self.last = Answer(
            question=question,
            text="".join(collected) + tail,
            hits=hits,
            grounded=True,
            provider=self.provider.name,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    def ask(self, question: str, history=None, top_k: int | None = None) -> Answer:
        for _ in self.stream(question, history, top_k):
            pass
        return self.last
