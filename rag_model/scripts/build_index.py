import argparse
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from rag_model.src.retrieval import index

def main() -> None:
    """Builds the vector and BM25 indexes from the knowledge base JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    index.build(kb_path=args.kb, index_dir=args.out)

if __name__ == "__main__":
    main()