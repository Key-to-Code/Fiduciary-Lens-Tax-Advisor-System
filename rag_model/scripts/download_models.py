from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared import config

config.apply_tls_workaround()


def main() -> None:
    """Downloads the necessary models into the local Hugging Face cache."""
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-llm", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    print(f"Fetching embedder {config.EMBED_MODEL} ...")
    snapshot_download(config.EMBED_MODEL)
    print("  done")

    if args.with_llm:
        print(f"Fetching generator {config.LOCAL_MODEL} ...")
        snapshot_download(
            config.LOCAL_MODEL,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        )
        print("  done")


if __name__ == "__main__":
    main()