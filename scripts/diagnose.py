"""Diagnostics on the dialogue runs: ReAct turn errors, subset comparison with the sequential run, browse_first failures."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lq.evaluate import read_jsonl, score_dialogue  # noqa: E402

R = Path(__file__).resolve().parents[1] / "runs"
react = read_jsonl(R / "dialogues" / "react.jsonl")
errs = Counter(str(t.get("error"))[:90] for d in react for t in d["turns"] if t.get("error"))
print("react turns with errors:", sum(errs.values()), "of", sum(len(d["turns"]) for d in react))
for e, n in errs.most_common(5):
    print(f"  {n:4d}  {e}")
print("react dialogues with >=1 error turn:", sum(any(t.get("error") for t in d["turns"]) for d in react))
print("react empty replies:", sum(1 for d in react for t in d["turns"] if not t["bot"]))

seq_ids = {d["persona"]["id"] for d in read_jsonl(R / "dialogues_latency" / "react.jsonl")}
for name in ("react", "agentic_rag"):
    main = [score_dialogue(d) for d in read_jsonl(R / "dialogues" / f"{name}.jsonl") if d["persona"]["id"] in seq_ids]
    seq = [score_dialogue(d) for d in read_jsonl(R / "dialogues_latency" / f"{name}.jsonl")]
    f = lambda xs: f"{sum(x['success'] for x in xs if x['expect_lead'])}/{sum(x['expect_lead'] for x in xs)} success, " \
                   f"{sum(x['unsupported_registrations'] for x in xs)} unsupported"
    print(f"{name}: same 24 personas -> main(8 workers): {f(main)} | sequential: {f(seq)}")

ag = read_jsonl(R / "dialogues" / "agentic_rag.jsonl")
shown = 0
for d in ag:
    s = score_dialogue(d)
    if s["behaviour"] == "browse_first" and not s["success"] and shown < 3:
        shown += 1
        print(f"\n=== {d['persona']['id']} end={d['end_reason']} regs={len(d['registrations'])}")
        for t in d["turns"]:
            print(f"  U: {t['user'][:130]}\n  B: {t['bot'][:200]!r} {t.get('trace')}")
reasons = Counter()
for d in ag:
    s = score_dialogue(d)
    if s["behaviour"] == "browse_first" and not s["success"]:
        reasons[(d["end_reason"], s["sim_gave_all"], len(d["registrations"]))] += 1
print("\nbrowse_first failure (end_reason, sim_gave_all, n_regs):", dict(reasons))
