"""Render one real held-out agentic_rag dialogue as a LaTeX table (Figure 2). Prints the chosen id and the table."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lq.evaluate import read_jsonl, score_dialogue  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
rows = read_jsonl(ROOT / "runs" / "dialogues" / "agentic_rag.jsonl")
want = sys.argv[1] if len(sys.argv) > 1 else None


def nodes(t):
    return set(t.get("trace") or [])


def ok(d):
    used = set().union(*(nodes(t) for t in d["turns"])) if d["turns"] else set()
    return score_dialogue(d)["success"] and {"rag", "lead", "confirm", "tool"} <= used and len(d["turns"]) <= 7


cands = [d for d in rows if (d["persona"]["id"] == want if want else ok(d))]
d = min(cands, key=lambda d: (len(d["turns"]), sum(len(t["bot"]) for t in d["turns"])))


def tex(s, limit=None):
    s = re.sub(r"\s+", " ", s).strip()
    if limit and len(s) > limit:
        s = s[:limit].rsplit(" ", 1)[0] + " \\ldots"
    for a, b in [("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"),
                 ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\^{}")]:
        if a == "\\" and "\\ldots" in s:
            continue
        s = s.replace(a, b)
    return s.replace("\\\\ldots", "\\ldots")


print("% dialogue", d["persona"]["id"], d["persona"]["behaviour"], file=sys.stderr)
for t in d["turns"]:
    path = r" $\rightarrow$ ".join(tex(x) for x in t["trace"])
    print(f"{t['t'] + 1} & {tex(t['user'])} & {tex(t['bot'], 230)} & \\texttt{{\\scriptsize {path}}} \\\\")
