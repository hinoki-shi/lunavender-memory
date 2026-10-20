"""Optional Ollama embedding adapter for the vector-sidecar experiment.

This is an example adapter, not part of the stdlib-only core. It uses the
Ollama HTTP API, so it does not require torch/transformers in the caller's
Python environment.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class OllamaEmbedder:
    model: str
    dim: int

    def __init__(self, model: str = "qwen3-embedding:latest",
                 base_url: str = "http://127.0.0.1:11434",
                 timeout: float = 120.0) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.dim = 0

    def _request(self, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + "/api/embed",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            # Older Ollama versions expose /api/embeddings with a single prompt.
            text = payload.get("input")
            if not isinstance(text, str):
                raise
            req = urllib.request.Request(
                self.base_url + "/api/embeddings",
                data=json.dumps({"model": self.model, "prompt": text},
                                ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                one = json.loads(resp.read().decode("utf-8"))
            return {"embeddings": [one["embedding"]]}

    def embed(self, text: str) -> tuple[float, ...]:
        data = self._request({"model": self.model, "input": str(text or "")})
        embeddings = data.get("embeddings") or []
        if not embeddings:
            raise RuntimeError("Ollama returned no embedding: %r" % (data,))
        vec = tuple(float(x) for x in embeddings[0])
        if not self.dim:
            self.dim = len(vec)
        elif len(vec) != self.dim:
            raise RuntimeError("embedding dimension changed: %d -> %d"
                               % (self.dim, len(vec)))
        return vec

    def embed_batch(self, texts: list[str]) -> list[tuple[float, ...]]:
        data = self._request({"model": self.model, "input": list(texts)})
        embeddings = data.get("embeddings") or []
        if len(embeddings) != len(texts):
            raise RuntimeError("Ollama returned %d embeddings for %d inputs"
                               % (len(embeddings), len(texts)))
        out = [tuple(float(x) for x in vec) for vec in embeddings]
        if out:
            self.dim = len(out[0])
        return out
