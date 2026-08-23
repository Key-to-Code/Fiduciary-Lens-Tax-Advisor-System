"""Retrieval evaluation against a gold set.

Retrieval quality is the metric that matters most here: the generator can only
be as honest as the passages it is handed, so if the right provision never
surfaces, no amount of prompting saves the answer.

    python eval_retrieval.py            # summary
    python eval_retrieval.py --verbose  # per-question detail

Two things are measured:
  recall@k  - did an acceptable provision appear in the top-k passages?
  refusal   - are off-topic questions correctly scored below the grounding bar?

Expected provisions were read off the headings in the knowledge base itself, not
recalled from memory, because this Act renumbered the familiar sections (the old
80C deduction now lives in section 123 and Schedule XV).
"""

from __future__ import annotations

import argparse

from rag import index, retrieve

# question -> provisions that would be a correct citation, as (unit, number)
GOLD: list[tuple[str, list[tuple[str, str]]]] = [
    ("How is the residential status of an individual determined in India?",
     [("Section", "6")]),
    ("What counts as a perquisite for salary purposes?",
     [("Section", "17")]),
    ("What deductions are allowed from income from house property?",
     [("Section", "22")]),
    ("What are the instalments and due dates for paying advance tax?",
     [("Section", "408")]),
    ("Who is liable to pay advance tax?",
     [("Section", "404"), ("Section", "405")]),
    ("Is there a deduction for health insurance premium?",
     [("Section", "126")]),
    ("Can I deduct interest on a loan taken for higher education?",
     [("Section", "129")]),
    ("How is depreciation on business assets deducted?",
     [("Section", "33"), ("Rule", "25"), ("Appendix", "I")]),
    ("What interest is charged for filing the return of income late?",
     [("Section", "423")]),
    ("Is there a deduction for donations to charitable institutions?",
     [("Section", "133")]),
    ("Are capital gains on the sale of a residential house exempt?",
     [("Section", "82")]),
    ("What is the penalty for under-reporting income?",
     [("Section", "439")]),
    ("Is interest on savings deposits deductible?",
     [("Section", "153")]),
    ("Who is required to file a return of income?",
     [("Section", "263"), ("Rule", "164")]),
    ("What deduction is available for life insurance premium and provident fund?",
     [("Section", "123"), ("Schedule", "XV")]),
    ("How are capital gains computed?",
     [("Section", "72"), ("Section", "67")]),
    ("What relief is available when salary is received in arrears?",
     [("Section", "157")]),
    ("How do I appeal against an order to the Commissioner (Appeals)?",
     [("Section", "357"), ("Section", "358"), ("Rule", "167")]),
    ("Can a loss from house property be carried forward?",
     [("Section", "110")]),
    ("How is total income rounded off?",
     [("Section", "516")]),
]

# Must fall below the grounding bar so the bot refuses instead of improvising.
OFF_TOPIC = [
    "Who won the 2018 FIFA World Cup?",
    "How do I bake a sourdough loaf?",
    "What is the best programming language for web development?",
    "Give me a recipe for butter chicken.",
    "What is the capital of Brazil?",
]


def evaluate(engine_index, k: int, verbose: bool) -> tuple[float, float]:
    hit_at_k = 0
    reciprocal_ranks = []

    for question, accepted in GOLD:
        hits = retrieve.search(engine_index, question, top_k=k)
        found_rank = None
        for rank, hit in enumerate(hits, start=1):
            if (hit.chunk.unit, hit.chunk.number) in accepted:
                found_rank = rank
                break
        if found_rank:
            hit_at_k += 1
            reciprocal_ranks.append(1 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

        if verbose or not found_rank:
            status = f"OK  @{found_rank}" if found_rank else "MISS   "
            want = ", ".join(f"{u} {n}" for u, n in accepted)
            print(f"  {status}  {question}")
            print(f"          want: {want}")
            if not found_rank or verbose:
                for hit in hits[:k]:
                    print(f"          got : {hit.chunk.unit} {hit.chunk.number} "
                          f"(cos {hit.dense_score:.2f}) {(hit.chunk.heading or '')[:52]}")

    recall = hit_at_k / len(GOLD)
    mrr = sum(reciprocal_ranks) / len(GOLD)
    return recall, mrr


def evaluate_refusals(engine_index, k: int, verbose: bool) -> float:
    refused = 0
    for question in OFF_TOPIC:
        hits = retrieve.search(engine_index, question, top_k=k)
        grounded = retrieve.is_grounded(hits)
        best = max((h.dense_score for h in hits), default=0.0)
        if not grounded:
            refused += 1
        if verbose or grounded:
            status = "REFUSED" if not grounded else "LEAKED "
            print(f"  {status}  (best cos {best:.2f})  {question}")
    return refused / len(OFF_TOPIC)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-k", type=int, default=6, help="passages retrieved (default 6)")
    parser.add_argument("--verbose", action="store_true", help="show every question")
    args = parser.parse_args()

    engine_index = index.load()
    print(f"Index: {len(engine_index.chunks)} chunks, {engine_index.embed_model}\n")

    print(f"Grounding: {len(GOLD)} questions, recall@{args.k}")
    recall, mrr = evaluate(engine_index, args.k, args.verbose)

    print(f"\nRefusal: {len(OFF_TOPIC)} off-topic questions")
    refusal_rate = evaluate_refusals(engine_index, args.k, args.verbose)

    print("\n" + "-" * 52)
    print(f"  recall@{args.k}      {recall:.0%}  ({round(recall * len(GOLD))}/{len(GOLD)})")
    print(f"  MRR            {mrr:.3f}")
    print(f"  refusal rate   {refusal_rate:.0%}  "
          f"({round(refusal_rate * len(OFF_TOPIC))}/{len(OFF_TOPIC)})")
    print("-" * 52)


if __name__ == "__main__":
    main()
