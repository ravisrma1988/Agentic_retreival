"""Thin OpenAI-compatible client with per-call telemetry (latency and token counts)."""
import threading
import time
from functools import lru_cache

from openai import OpenAI

from . import config

# Per-thread log of calls made since the last reset_log(); one entry per LLM call or retrieval.
_tls = threading.local()


def _log() -> list[dict]:
    if not hasattr(_tls, "calls"):
        _tls.calls = []
    return _tls.calls


def reset_log() -> None:
    _tls.calls = []


def get_log() -> list[dict]:
    return list(_log())


def log_event(node: str, latency_s: float, prompt_tokens: int = 0, completion_tokens: int = 0, kind: str = "llm") -> None:
    _log().append({"node": node, "kind": kind, "latency_s": latency_s,
                   "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens})


@lru_cache(maxsize=None)
def _client(base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="EMPTY", timeout=120)


def chat(messages: list[dict], *, node: str, aux: bool = False, temperature: float | None = None,
         max_tokens: int | None = None, seed: int | None = None) -> str:
    base_url = config.AUX_BASE_URL if aux else config.AGENT_BASE_URL
    model = config.AUX_MODEL if aux else config.AGENT_MODEL
    t0 = time.perf_counter()
    r = _client(base_url).chat.completions.create(
        model=model,
        messages=messages,
        temperature=config.TEMPERATURE if temperature is None else temperature,
        top_p=config.TOP_P,
        max_tokens=max_tokens or config.MAX_TOKENS,
        seed=config.SEED if seed is None else seed,
    )
    u = r.usage
    log_event(node, time.perf_counter() - t0, u.prompt_tokens if u else 0, u.completion_tokens if u else 0)
    return (r.choices[0].message.content or "").strip()


def summarize(calls: list[dict]) -> dict:
    llm = [c for c in calls if c["kind"] == "llm"]
    return {
        "llm_calls": len(llm),
        "prompt_tokens": sum(c["prompt_tokens"] for c in llm),
        "completion_tokens": sum(c["completion_tokens"] for c in llm),
        "llm_latency_s": sum(c["latency_s"] for c in llm),
        "retrieval_latency_s": sum(c["latency_s"] for c in calls if c["kind"] == "retrieval"),
    }
