"""Dense retriever over KB chunks (bge-small-en-v1.5, cosine similarity, exact search)."""
import threading
import time

import numpy as np

from . import config, llm
from .kb import chunk_docs, load_docs


class Retriever:
    def __init__(self, model_name: str = config.EMBED_MODEL, top_k: int = config.TOP_K):
        from sentence_transformers import SentenceTransformer
        import torch

        self.top_k = top_k
        self._lock = threading.Lock()  # one encoder shared by worker threads
        self.chunks = chunk_docs(load_docs())
        self.model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        self.emb = self.model.encode([c["text"] for c in self.chunks], normalize_embeddings=True,
                                     convert_to_numpy=True)

    def search(self, query: str, k: int | None = None, log: bool = True) -> list[dict]:
        t0 = time.perf_counter()
        with self._lock:
            q = self.model.encode([config.EMBED_QUERY_PREFIX + query], normalize_embeddings=True,
                                  convert_to_numpy=True)[0]
        scores = self.emb @ q
        idx = np.argsort(-scores)[: k or self.top_k]
        hits = [{**self.chunks[i], "score": float(scores[i])} for i in idx]
        if log:
            llm.log_event("retrieve", time.perf_counter() - t0, kind="retrieval")
        return hits


def format_context(hits: list[dict]) -> str:
    return "\n\n".join(f"[{i + 1}] {h['text']}" for i, h in enumerate(hits))
