"""Pre-download the models the pipeline needs, into the local Hugging Face cache.

    python download_models.py            # embedder only (needed to build the index)
    python download_models.py --with-llm # also the local generator (~3 GB)

Afterwards everything runs offline. Run this before `build_index.py` on a fresh
machine, so the first question is not also a 3 GB download.
"""

from __future__ import annotations

import argparse

from rag import config

config.apply_tls_workaround()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-llm", action="store_true",
                        help=f"also fetch {config.LOCAL_MODEL} (~3 GB)")
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
