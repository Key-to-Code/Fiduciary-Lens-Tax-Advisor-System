"""(Re)build the vector + BM25 indexes from the knowledge base JSON.

    python build_index.py

Run this whenever Data/rag_knowledge_base.json changes.
"""

import argparse
from pathlib import Path

from rag import index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", type=Path, default=None, help="path to knowledge base JSON")
    parser.add_argument("--out", type=Path, default=None, help="index output directory")
    args = parser.parse_args()
    index.build(kb_path=args.kb, index_dir=args.out)


if __name__ == "__main__":
    main()
