"""Ask the tax knowledge base a question.

    python ask.py "What deductions are allowed for life insurance premium?"
    python ask.py                      # interactive session
    python ask.py --providers          # show which LLM backends are usable
    python ask.py -q "..." --retrieval-only   # show retrieved provisions, no LLM
"""

from __future__ import annotations

import argparse
import sys

from rag import llm
from rag.answer import TaxQA
from rag.prompt import DISCLAIMER

BANNER = """Fiduciary-Lens Tax QA - Income-tax Act, 2025 + Income-tax Rules, 2026
Educational information only, not professional tax advice.
Type a question, or 'exit' to quit."""


def _print_sources(answer) -> None:
    if not answer.sources:
        return
    print("\nSources")
    for source in answer.sources:
        print(f"  [{source['n']}] {source['citation']}")
        print(f"      {source['document']}")
        print(f"      as of: {source['as_of']}  (relevance {source['cosine']:.2f})")


def run_once(engine: TaxQA, question: str, history, show_sources: bool) -> None:
    print()
    for piece in engine.stream(question, history=history):
        sys.stdout.write(piece)
        sys.stdout.flush()
    print()
    answer = engine.last
    if show_sources:
        _print_sources(answer)
    print(f"\n[{answer.provider} | {len(answer.hits)} passages | {answer.latency_ms} ms]")
    history.append((question, answer.text))


def retrieval_only(engine: TaxQA, question: str) -> None:
    hits = engine.retrieve(question)
    print(f"\nTop {len(hits)} passages for: {question}\n")
    for position, hit in enumerate(hits, start=1):
        print(f"[{position}] {hit.chunk.citation}")
        print(f"    cosine {hit.dense_score:.3f} | bm25 {hit.lexical_score:.1f} "
              f"| fused {hit.score:.3f} | {hit.chunk.chunk_id}")
        body = " ".join(hit.chunk.content.split())
        print(f"    {body[:320]}...\n")
    print(DISCLAIMER)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="*", help="question to ask")
    parser.add_argument("-q", "--query", help="question to ask (alternative to positional)")
    parser.add_argument("--provider", default=None,
                        help="auto (default), ollama, openai or extractive")
    parser.add_argument("--providers", action="store_true",
                        help="list LLM backends and whether they are reachable")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="show retrieved provisions without generating an answer")
    parser.add_argument("--no-sources", action="store_true", help="hide the sources panel")
    parser.add_argument("--top-k", type=int, default=None, help="passages to retrieve")
    args = parser.parse_args()

    if args.providers:
        print("LLM providers")
        for name, ok, detail in llm.describe_providers():
            print(f"  {'OK ' if ok else '-- '} {name:<11} {detail}")
        print("\nSet LLM_PROVIDER to pin one; 'auto' takes the first available.")
        return

    question = args.query or " ".join(args.question)

    engine = TaxQA(provider=args.provider)

    if args.retrieval_only:
        if not question:
            parser.error("--retrieval-only needs a question")
        retrieval_only(engine, question)
        return

    if question:
        run_once(engine, question, [], not args.no_sources)
        return

    print(BANNER)
    print(f"LLM provider: {engine.provider.name}\n")
    history: list[tuple[str, str]] = []
    while True:
        try:
            question = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q"}:
            break
        run_once(engine, question, history, not args.no_sources)
        print()


if __name__ == "__main__":
    main()
