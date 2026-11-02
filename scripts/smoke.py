"""Smoke test against the live servers on DEVELOPMENT personas only (never the test personas)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lq import config  # noqa: E402
from lq.baselines import make_system  # noqa: E402
from lq.evaluate import read_jsonl, score_dialogue  # noqa: E402
from lq.retriever import Retriever  # noqa: E402
from lq.simulate import simulate  # noqa: E402

r = Retriever()
dev = {p["id"]: p for p in read_jsonl(config.DATA_DIR / "personas_dev.jsonl")}
ids = sys.argv[1].split(",") if len(sys.argv) > 1 else ["D054", "D095", "D100", "D101", "D126"]
systems = sys.argv[2].split(",") if len(sys.argv) > 2 else ["agentic_rag"]
for name in systems:
    for i in ids:
        p = dev[i]
        d = simulate(make_system(name, r), p)
        s = score_dialogue(d)
        print(f"\n=== {name} {i} {p['behaviour']} | truth: {p['name']} | {p['email']} | {p['platform']} -> success={s['success']}")
        for t in d["turns"]:
            print(f"  U: {t['user'][:140]}\n  B: {t['bot'][:140]!r} {t.get('trace')}")
        print("  regs:", [{k: v for k, v in r_.items() if k != 'ts'} for r_ in d["registrations"]])
