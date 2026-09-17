"""Provider-agnostic text generation.

The pipeline never imports a vendor SDK directly: it asks for `get_provider()`
and calls `.generate()`. Swapping Ollama for a cloud API is a config change.

Providers, in the order `auto` probes them:
  ollama     - local, free, offline (needs `ollama serve` + a pulled model)
  local      - in-process transformers model, free and offline, once downloaded
  openai     - OPENAI_API_KEY, optionally OPENAI_BASE_URL for a compatible endpoint
  extractive - no LLM at all; quotes the retrieved provisions verbatim

Local backends come first on purpose: they cost nothing, run offline, and keep
users' tax questions off third-party servers.

`extractive` is a real fallback, not an error path: quoting the statute with its
citation is the most fiduciary-safe answer available, just a less readable one.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Iterator

from . import config

config.apply_tls_workaround()


class Provider(ABC):
    name: str

    @abstractmethod
    def generate(self, system: str, user: str) -> Iterator[str]:
        """Yield answer text in chunks, so callers can stream it."""

    def complete(self, system: str, user: str) -> str:
        return "".join(self.generate(system, user))


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or config.OLLAMA_MODEL
        self.host = (host or config.OLLAMA_HOST).rstrip("/")

    @staticmethod
    def available(host: str | None = None) -> tuple[bool, str]:
        import requests
        host = (host or config.OLLAMA_HOST).rstrip("/")
        try:
            response = requests.get(f"{host}/api/tags", timeout=2)
            models = [m["name"] for m in response.json().get("models", [])]
        except Exception as exc:
            return False, f"not reachable at {host} ({type(exc).__name__})"
        if not models:
            return False, f"running at {host} but no models pulled"
        return True, ", ".join(models)

    def generate(self, system: str, user: str) -> Iterator[str]:
        import json
        import requests
        response = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "stream": True,
                "options": {"temperature": config.TEMPERATURE,
                            "num_predict": config.MAX_TOKENS},
            },
            stream=True,
            timeout=300,
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            payload = json.loads(line)
            piece = payload.get("message", {}).get("content", "")
            if piece:
                yield piece
            if payload.get("done"):
                break


class LocalProvider(Provider):
    """A small instruct model run in-process with transformers.

    Loaded once per process and kept on the GPU if there is one. Weights are
    resolved from the Hugging Face cache, so this works with no network.
    """

    name = "local"
    _cache: dict[str, tuple] = {}

    def __init__(self, model: str | None = None):
        self.model_id = model or config.LOCAL_MODEL

    @staticmethod
    def available(model: str | None = None) -> tuple[bool, str]:
        model = model or config.LOCAL_MODEL
        try:
            from huggingface_hub import snapshot_download
            path = snapshot_download(model, local_files_only=True)
        except Exception:
            return False, f"{model} not downloaded (python download_models.py --with-llm)"
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
        return True, f"{model} cached, will run on {device} ({path[:40]}...)"

    def _load(self):
        if self.model_id not in self._cache:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            use_cuda = torch.cuda.is_available()
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=torch.float16 if use_cuda else torch.float32,
            )
            model.to("cuda" if use_cuda else "cpu").eval()
            self._cache[self.model_id] = (tokenizer, model)
        return self._cache[self.model_id]

    def generate(self, system: str, user: str) -> Iterator[str]:
        import threading

        import torch
        from transformers import TextIteratorStreamer

        tokenizer, model = self._load()
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True,
                                        skip_special_tokens=True)
        kwargs = dict(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            streamer=streamer,
            max_new_tokens=config.MAX_TOKENS,
            do_sample=config.TEMPERATURE > 0,
            temperature=max(config.TEMPERATURE, 1e-5),
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
        worker = threading.Thread(target=model.generate, kwargs=kwargs, daemon=True)
        worker.start()
        yield from streamer
        worker.join()


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, model: str | None = None):
        self.model = model or config.OPENAI_MODEL
        self._client = None

    @staticmethod
    def available() -> tuple[bool, str]:
        if not os.getenv("OPENAI_API_KEY"):
            return False, "OPENAI_API_KEY not set"
        return True, os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def client(self):
        if self._client is None:
            import httpx
            from openai import OpenAI
            # httpx pins certifi's bundle, which misses this machine's proxy root CA.
            verify = str(config.CA_BUNDLE) if config.CA_BUNDLE.exists() else True
            self._client = OpenAI(
                base_url=os.getenv("OPENAI_BASE_URL") or None,
                http_client=httpx.Client(verify=verify, timeout=120),
            )
        return self._client

    def generate(self, system: str, user: str) -> Iterator[str]:
        stream = self.client().chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            stream=True,
        )
        for event in stream:
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content


class ExtractiveProvider(Provider):
    """No LLM: hand back the retrieved provisions as-is.

    The prompt builder puts the passages in the user message under [1], [2], ...
    so echoing that section is already a cited, fully grounded answer.
    """

    name = "extractive"

    def generate(self, system: str, user: str) -> Iterator[str]:
        marker = "RETRIEVED PROVISIONS"
        body = user.split(marker, 1)[-1]
        body = body.split("QUESTION:", 1)[0].strip().lstrip("-").strip()
        yield (
            "No language model is configured, so I cannot summarise. "
            "Here are the provisions retrieved for your question, quoted verbatim "
            "from the knowledge base:\n\n" + body
        )


_PROVIDERS = {
    "ollama": OllamaProvider,
    "local": LocalProvider,
    "openai": OpenAIProvider,
    "extractive": ExtractiveProvider,
}

# Order `auto` walks. Local and free first, paid API next, no-LLM last.
_AUTO_ORDER = ("ollama", "local", "openai")


def describe_providers() -> list[tuple[str, bool, str]]:
    rows = [(name,) + _PROVIDERS[name].available() for name in _AUTO_ORDER]
    rows.append(("extractive", True, "always available (quotes provisions verbatim)"))
    return rows


def get_provider(name: str | None = None) -> Provider:
    name = (name or config.LLM_PROVIDER).lower()
    if name in _PROVIDERS:
        return _PROVIDERS[name]()
    if name != "auto":
        raise ValueError(
            f"Unknown LLM_PROVIDER {name!r}; expected one of "
            f"{', '.join(_PROVIDERS)} or 'auto'"
        )

    for candidate in _AUTO_ORDER:
        if _PROVIDERS[candidate].available()[0]:
            return _PROVIDERS[candidate]()
    return ExtractiveProvider()
