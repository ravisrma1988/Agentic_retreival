"""Print failed agentic_rag dialogues per behaviour (run from paper48/)."""
import json
import sys

sys.path.insert(0, ".")
from lq.evaluate import score_dialogue  # noqa: E402

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
want = sys.argv[2].split(",")
n_each = int(sys.argv[3]) if len(sys.argv) > 3 else 3
shown = {}
for d in rows:
    s = score_dialogue(d)
    b = s["behaviour"]
    if b not in want or s["success"] or shown.get(b, 0) >= n_each:
        continue
    shown[b] = shown.get(b, 0) + 1
    p = d["persona"]
    print(f"\n=== {p['id']} {b} | truth: {p['name']} | {p['email']} | {p['platform']} | end={d['end_reason']}")
    print("    regs:", d["registrations"])
    print("    fields:", {k: (v["pred"], v["ok"]) for k, v in s["fields"].items()})
    for t in d["turns"]:
        print(f"  U: {t['user'][:150]}\n  B: {t['bot'][:110]!r} {t.get('trace')}")
