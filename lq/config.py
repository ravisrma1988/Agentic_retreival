"""Central configuration. Every value used in the paper is defined here so it can be reported verbatim."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "kb"
DATA_DIR = ROOT / "data"
RUNS_DIR = Path(os.getenv("RUNS_DIR", ROOT / "runs"))

# Agent LLM (system under test and all baselines), served by vLLM with an OpenAI-compatible API.
AGENT_BASE_URL = os.getenv("AGENT_BASE_URL", "http://localhost:8000/v1")
AGENT_MODEL = os.getenv("AGENT_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# Auxiliary LLM from a different model family: test-data generation, user simulation and LLM-as-judge.
# Using a different family than the agent reduces self-preference bias in judging and simulation.
AUX_BASE_URL = os.getenv("AUX_BASE_URL", "http://localhost:8001/v1")
AUX_MODEL = os.getenv("AUX_MODEL", "microsoft/phi-4")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# bge v1.5 recommends this instruction prefix for short retrieval queries.
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Decoding settings for the agent (deterministic).
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 384
SEED = 42

# Retrieval settings.
CHUNK_WORDS = 120
CHUNK_OVERLAP = 30
TOP_K = 4

# User simulator settings (stochastic on purpose, seeded per dialogue).
SIM_TEMPERATURE = 0.7
SIM_MAX_TURNS = 12
# Personas for the reported test run. Two development sets were used while building the system:
# seed 42 (personas_dev.jsonl) and seed 2026 (personas_dev2.jsonl). The test set (seed 7) was not inspected.
TEST_PERSONA_SEED = 7

LEAD_FIELDS = ("name", "email", "platform")
INTENTS = ("greeting", "product_inquiry", "high_intent", "general")
