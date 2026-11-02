"""Experiment runner and metrics.

  python -m lq.evaluate finalize-intent        merge annotator sheets A/B (+ adjudication) -> data/intent_gold.csv
  python -m lq.evaluate intent [--no-context]  intent accuracy, macro-F1, confusion matrix
  python -m lq.evaluate retrieval              Recall@k / MRR of the retriever on gold sources
  python -m lq.evaluate qa SYSTEM...           run the QA set (sequential, so latency is clean)
  python -m lq.evaluate judge SYSTEM...        LLM-as-judge on QA outputs
  python -m lq.evaluate dialogues SYSTEM... [--workers N] [--per-behaviour K] [--tag T]
  python -m lq.evaluate annotation-export      stratified QA sample sheets for two human annotators
  python -m lq.evaluate agreement              Cohen's kappa: human-human and human-judge
  python -m lq.evaluate report                 all tables -> runs/report.md
"""
import argparse
import csv
import json
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config, llm, prompts

RUNS = config.RUNS_DIR
QA_SYSTEMS = ["llm_only", "full_context", "naive_rag", "react", "agentic_rag"]
ANNOT_DIR = config.DATA_DIR / "annotation"


# ------------------------------------------------------------------ helpers

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def boot_ci(values, n: int = 2000, seed: int = config.SEED) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt_ci(values, pct: bool = True) -> str:
    m, lo, hi = boot_ci(values)
    k = 100 if pct else 1
    return f"{m * k:.1f} [{lo * k:.1f}, {hi * k:.1f}]" if pct else f"{m:.2f} [{lo:.2f}, {hi:.2f}]"


def norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower().replace(",", "").replace("–", "-").replace("‑", "-"))


def keyfact_coverage(answer: str, must: list[list[str]]) -> float | None:
    if not must:
        return None
    a = norm_text(answer)
    return sum(any(alt in a for alt in grp) for grp in must) / len(must)


REFUSAL_RE = re.compile(r"(don't|do not|doesn't|does not) (have|contain|include|mention)|not (available|mentioned|specified|"
                        r"listed|provided) in|no information|not sure|unable to (find|provide)|support@autostream", re.I)


def load_retriever():
    from .retriever import Retriever
    return Retriever()


def env_snapshot():
    import subprocess
    d = RUNS / "env"
    d.mkdir(parents=True, exist_ok=True)
    (d / "pip_freeze.txt").write_text(subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout)
    (d / "config.json").write_text(json.dumps({k: str(getattr(config, k)) for k in dir(config) if k.isupper()}, indent=2))


# ------------------------------------------------------------------ intent

def finalize_intent(_):
    pool = {r["id"]: r for r in csv.DictReader(open(config.DATA_DIR / "intent_pool.csv", encoding="utf-8"))}
    lab = lambda who: {r["id"]: list(r.values())[3].strip() for r in
                       csv.DictReader(open(ANNOT_DIR / f"intent_sheet_{who}.csv", encoding="utf-8-sig"))}
    A, B = lab("A"), lab("B")
    adj_path = ANNOT_DIR / "intent_adjudication.csv"
    adj = {r["id"]: r["label"].strip() for r in csv.DictReader(open(adj_path, encoding="utf-8-sig"))} if adj_path.exists() else {}
    from sklearn.metrics import cohen_kappa_score
    ids = [i for i in pool if A.get(i) in config.INTENTS and B.get(i) in config.INTENTS]
    kappa = cohen_kappa_score([A[i] for i in ids], [B[i] for i in ids])
    disagree = [i for i in ids if A[i] != B[i]]
    rows, unresolved = [], []
    for i, r in pool.items():
        if A.get(i) == B.get(i) and A.get(i) in config.INTENTS:
            gold = A[i]
        elif adj.get(i) in config.INTENTS:
            gold = adj[i]
        else:
            unresolved.append(i)
            continue
        rows.append({"id": i, "prev_bot": r["prev_bot"], "text": r["text"], "gold": gold, "gen_label": r["gen_label"],
                     "source": r["source"]})
    with open(config.DATA_DIR / "intent_gold.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    changed = sum(r["gold"] != r["gen_label"] for r in rows)
    summary = {"n_labelled_both": len(ids), "kappa_A_B": kappa, "n_disagree": len(disagree),
               "n_unresolved": len(unresolved), "n_gold": len(rows), "gold_differs_from_generated": changed}
    (ANNOT_DIR / "intent_agreement.json").write_text(json.dumps(summary, indent=2))
    if unresolved:
        with open(ANNOT_DIR / "intent_adjudication_todo.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["id", "prev_bot", "text", "label_A", "label_B", "label"])
            w.writerows([[i, pool[i]["prev_bot"], pool[i]["text"], A.get(i, ""), B.get(i, ""), ""] for i in unresolved])
    print(json.dumps(summary, indent=2))


def eval_intent(args):
    from sklearn.metrics import classification_report, confusion_matrix
    from .agent import classify_intent

    gold_path = config.DATA_DIR / "intent_gold.csv"
    if not gold_path.exists():
        if not args.allow_unverified:
            raise SystemExit("intent_gold.csv missing: run finalize-intent after annotation (or pass --allow-unverified)")
        rows = [{**r, "gold": r["gen_label"]} for r in csv.DictReader(open(config.DATA_DIR / "intent_pool.csv", encoding="utf-8"))]
    else:
        rows = list(csv.DictReader(open(gold_path, encoding="utf-8")))

    def one(r):
        label, ok = classify_intent(r["text"], "" if args.no_context else r["prev_bot"])
        return {**r, "pred": label, "parsed": ok}

    with ThreadPoolExecutor(args.workers) as ex:
        out = list(ex.map(one, rows))
    tag = "intent_nocontext" if args.no_context else "intent"
    write_jsonl(RUNS / f"{tag}.jsonl", out)
    y, p = [r["gold"] for r in out], [r["pred"] for r in out]
    labels = list(config.INTENTS)
    rep = classification_report(y, p, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, p, labels=labels)
    acc = [int(a == b) for a, b in zip(y, p)]
    ctx = [int(r["gold"] == r["pred"]) for r in out if r["source"] == "context_handwritten"]
    res = {"n": len(out), "accuracy": fmt_ci(acc), "macro_f1": rep["macro avg"]["f1-score"],
           "per_class": {l: rep[l] for l in labels}, "confusion": cm.tolist(), "labels": labels,
           "context_subset_accuracy": fmt_ci(ctx) if ctx else None,
           "unparsed": sum(not r["parsed"] for r in out)}
    (RUNS / f"{tag}_metrics.json").write_text(json.dumps(res, indent=2))
    _plot_confusion(cm, labels, RUNS / f"{tag}_confusion.png")
    print(json.dumps(res, indent=2))


def _plot_confusion(cm, labels, path):
    import os
    os.environ["MPLBACKEND"] = "Agg"  # an inherited Jupyter inline backend breaks headless runs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["Greeting", "Product\ninquiry", "High\nintent", "General"]
    fig, ax = plt.subplots(figsize=(3.4, 3.0), dpi=300)
    ax.imshow(cm, cmap="Blues")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(labels)), names, fontsize=7)
    ax.set_yticks(range(len(labels)), names, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Gold", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))


# ------------------------------------------------------------------ retrieval

def eval_retrieval(_):
    r = load_retriever()
    items = [q for q in read_jsonl(config.DATA_DIR / "qa.jsonl") if q["answerable"] and q["category"] != "followup"]
    ks = [1, 3, 4, 5, 10]
    hits = {k: [] for k in ks}
    rr = []
    for q in items:
        docs = [h["doc_id"] for h in r.search(q["question"], k=max(ks), log=False)]
        rank = next((i + 1 for i, d in enumerate(docs) if d in q["sources"]), None)
        rr.append(1 / rank if rank else 0.0)
        for k in ks:
            hits[k].append(int(rank is not None and rank <= k))
    res = {"n": len(items), "n_chunks": len(r.chunks), **{f"recall@{k}": fmt_ci(hits[k]) for k in ks},
           "mrr": fmt_ci(rr, pct=False)}
    (RUNS / "retrieval_metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


# ------------------------------------------------------------------ QA

def run_qa(args):
    from .baselines import make_system
    env_snapshot()
    retriever = load_retriever()
    items = read_jsonl(config.DATA_DIR / "qa.jsonl")
    for name in args.systems:
        sys_ = make_system(name, retriever)
        for _ in range(3):  # warm-up turns (CUDA graphs, caches); not recorded
            sys_.reset()
            sys_.turn("What plans do you offer?")
        out = []
        t0 = time.time()
        for q in items:
            sys_.reset()
            for prior in q["prior_turns"]:
                sys_.turn(prior)
            answer, tel = sys_.turn(q["question"])
            out.append({"id": q["id"], "system": name, "answer": answer, **tel})
        write_jsonl(RUNS / "qa" / f"{name}.jsonl", out)
        print(f"{name}: {len(out)} items in {time.time() - t0:.0f}s")


def _evidence(q, docs_by_id):
    return "\n\n".join(docs_by_id[d] for d in q["sources"]) if q["sources"] else "(no relevant text in the knowledge base)"


def run_judge(args):
    from .kb import load_docs
    docs_by_id = {d["doc_id"]: d["text"] for d in load_docs()}
    items = {q["id"]: q for q in read_jsonl(config.DATA_DIR / "qa.jsonl")}

    def one(row):
        q = items[row["id"]]
        raw = llm.chat([{"role": "system", "content": prompts.JUDGE_SYSTEM},
                        {"role": "user", "content": prompts.JUDGE_USER.format(
                            question=" / ".join(q["prior_turns"] + [q["question"]]),
                            answerable="yes" if q["answerable"] else "no (UNANSWERABLE)",
                            gold=q["gold"], evidence=_evidence(q, docs_by_id), answer=row["answer"] or "(empty)")}],
                       node="judge", aux=True, max_tokens=150)
        m = re.search(r"\{.*\}", raw, flags=re.S)
        try:
            j = json.loads(m.group(0))
            c, h = int(j["correctness"]), int(j["hallucination"])
            assert c in (0, 1, 2) and h in (0, 1)
            return {**row, "judge_correctness": c, "judge_hallucination": h, "judge_reason": j.get("reason", "")}
        except Exception:
            return {**row, "judge_correctness": None, "judge_hallucination": None, "judge_raw": raw}

    for name in args.systems:
        rows = read_jsonl(RUNS / "qa" / f"{name}.jsonl")
        with ThreadPoolExecutor(args.workers) as ex:
            out = list(ex.map(one, rows))
        write_jsonl(RUNS / "qa" / f"{name}.judged.jsonl", out)
        print(f"{name}: judged {len(out)}, unparsed {sum(r['judge_correctness'] is None for r in out)}")


def annotation_export(_):
    """Stratified sample: 3 items per category x every QA system, shuffled and blinded."""
    rng = random.Random(config.SEED)
    items = read_jsonl(config.DATA_DIR / "qa.jsonl")
    by_cat = defaultdict(list)
    for q in items:
        by_cat[q["category"]].append(q)
    chosen = [q for cat in sorted(by_cat) for q in rng.sample(by_cat[cat], 3)]
    from .kb import load_docs
    docs_by_id = {d["doc_id"]: d["text"] for d in load_docs()}
    rows = []
    for name in QA_SYSTEMS:
        ans = {r["id"]: r["answer"] for r in read_jsonl(RUNS / "qa" / f"{name}.jsonl")}
        rows += [{"q": q, "system": name, "answer": ans[q["id"]]} for q in chosen]
    rng.shuffle(rows)
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    for who in ("A", "B"):
        with open(ANNOT_DIR / f"qa_sheet_{who}.csv", "w", newline="", encoding="utf-8-sig") as f:  # BOM for Excel
            w = csv.writer(f)
            w.writerow(["sid", "question", "answerable", "reference_answer", "kb_evidence", "assistant_answer",
                        "correctness (0/1/2)", "hallucination (0/1)"])
            for k, r in enumerate(rows):
                q = r["q"]
                w.writerow([f"S{k:03d}", " / ".join(q["prior_turns"] + [q["question"]]),
                            "yes" if q["answerable"] else "no", q["gold"], _evidence(q, docs_by_id), r["answer"], "", ""])
    key = [{"sid": f"S{k:03d}", "id": r["q"]["id"], "system": r["system"]} for k, r in enumerate(rows)]
    write_jsonl(ANNOT_DIR / "qa_sheet_key.jsonl", key)
    print(f"wrote {len(rows)} rows per annotator (rubric: lq/prompts.py JUDGE_SYSTEM)")


def agreement(_):
    from sklearn.metrics import cohen_kappa_score
    key = {r["sid"]: r for r in read_jsonl(ANNOT_DIR / "qa_sheet_key.jsonl")}

    def sheet(who):
        out = {}
        for r in csv.DictReader(open(ANNOT_DIR / f"qa_sheet_{who}.csv", encoding="utf-8-sig")):
            v = list(r.values())
            if v[6].strip() and v[7].strip():
                out[r["sid"]] = (int(v[6]), int(v[7]))
        return out

    A, B = sheet("A"), sheet("B")
    judged = {}
    for name in QA_SYSTEMS:
        for r in read_jsonl(RUNS / "qa" / f"{name}.judged.jsonl"):
            judged[(r["id"], name)] = (r["judge_correctness"], r["judge_hallucination"])
    sids = [s for s in key if s in A and s in B]
    res = {"n_items": len(sids)}
    for idx, metric, w in ((0, "correctness", "quadratic"), (1, "hallucination", None)):
        a, b = [A[s][idx] for s in sids], [B[s][idx] for s in sids]
        res[f"{metric}_kappa_A_B"] = cohen_kappa_score(a, b, weights=w)
        res[f"{metric}_raw_agreement_A_B"] = float(np.mean(np.array(a) == np.array(b)))
        cons = [s for s in sids if A[s][idx] == B[s][idx]]
        j = [judged[(key[s]["id"], key[s]["system"])][idx] for s in cons]
        h = [A[s][idx] for s in cons]
        ok = [(x, y) for x, y in zip(h, j) if y is not None]
        res[f"{metric}_kappa_consensus_judge"] = cohen_kappa_score(*zip(*ok), weights=w) if ok else None
        res[f"{metric}_n_consensus"] = len(ok)
    (ANNOT_DIR / "qa_agreement.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


# ------------------------------------------------------------------ dialogues

def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def _strip_emails(s: str) -> str:
    from .extractor import EMAIL_RE
    return EMAIL_RE.sub(" ", s)


def _supported(reg: dict, user_text: str) -> bool:
    """Every registered value is traceable to what the user actually typed before the call."""
    from .extractor import PLATFORM_ALIASES
    u = user_text.lower()
    email_ok = (reg.get("email") or "").lower() in u and bool(reg.get("email"))
    name_ok = bool(reg.get("name")) and all(t in _tokens(_strip_emails(u)) for t in _tokens(reg["name"]))
    plat = (reg.get("platform") or "").lower()
    aliases = [a for a, c in PLATFORM_ALIASES.items() if c.lower() == plat] + [plat]
    plat_ok = bool(plat) and any(re.search(rf"(?<![a-z]){re.escape(a)}(?![a-z])", u) for a in aliases if a)
    return email_ok and name_ok and plat_ok


def _norm_platform(p):
    from .extractor import PLATFORM_ALIASES
    p = (p or "").strip().lower()
    return PLATFORM_ALIASES.get(p, p).lower() if p else ""


def _field_ok(field, pred, truth, user_text=""):
    if not pred:
        return False
    if field == "email":
        return pred.strip().lower() == truth.lower()
    if field == "platform":
        return _norm_platform(pred) == truth.lower()
    # name: no token outside the true name, and every true-name token the user typed (outside emails) is kept,
    # so a user who only types their first name is credited when the first name is captured.
    pred_t, truth_t = set(_tokens(pred)), set(_tokens(truth))
    typed = truth_t & set(_tokens(_strip_emails(user_text)))
    return bool(pred_t) and pred_t <= truth_t and typed <= pred_t


def score_dialogue(d: dict) -> dict:
    p, turns, regs = d["persona"], d["turns"], d["registrations"]
    user_all = " ".join(t["user"] for t in turns).lower()
    user_names = _tokens(_strip_emails(user_all))
    sim_gave = {"name": any(t in user_names for t in _tokens(p["name"])),
                "email": p["email"].lower() in user_all,
                "platform": bool(re.search(rf"\b({re.escape(p['platform'].lower())}|{re.escape(p['platform_alias'])})\b", user_all))}
    # registration turn index: first turn whose n_registrations reached k
    reg_turns = []
    for k in range(1, len(regs) + 1):
        reg_turns.append(next((t["t"] for t in turns if t["n_registrations"] >= k), None))
    supported = []
    for reg, rt in zip(regs, reg_turns):
        upto = " ".join(t["user"] for t in turns if rt is not None and t["t"] <= rt)
        supported.append(_supported(reg, upto))
    final = regs[-1] if regs else {}
    fields = {f: _field_ok(f, final.get(f), p[f], user_all) for f in config.LEAD_FIELDS}
    correct = bool(regs) and all(fields.values())
    return {
        "id": p["id"], "behaviour": p["behaviour"], "expect_lead": p["expect_lead"],
        # diagnostic only: did the simulated user state every persona field at some point?
        "sim_gave_all": all(sim_gave.values()) if p["expect_lead"] else True,
        "n_turns": len(turns), "n_registrations": len(regs),
        "success": (correct and all(supported) and len(regs) == 1) if p["expect_lead"] else not regs,
        "correct_final_lead": correct,
        "unsupported_registrations": sum(not s for s in supported),
        "false_registration": (not p["expect_lead"]) and bool(regs),
        "duplicate_registrations": max(len(regs) - 1, 0),
        "turns_to_lead": (reg_turns[-1] + 1) if correct and reg_turns[-1] is not None else None,
        "fields": {f: {"pred": final.get(f), "truth": p[f], "ok": fields[f]} for f in config.LEAD_FIELDS},
    }


def run_dialogues(args):
    from .baselines import make_system
    from .simulate import simulate
    env_snapshot()
    retriever = load_retriever()
    personas = read_jsonl(config.DATA_DIR / "personas.jsonl")
    if args.per_behaviour:
        seen = Counter()
        personas = [p for p in personas if (seen.update([p["behaviour"]]) or seen[p["behaviour"]] <= args.per_behaviour)]
    for name in args.systems:
        t0 = time.time()

        def one(p):
            return simulate(make_system(name, retriever), p)

        with ThreadPoolExecutor(args.workers) as ex:
            out = list(ex.map(one, personas))
        write_jsonl(RUNS / f"dialogues{args.tag}" / f"{name}.jsonl", out)
        print(f"{name}: {len(out)} dialogues in {time.time() - t0:.0f}s")


def dialogue_metrics(rows):
    s = [score_dialogue(d) for d in rows]
    lead = [x for x in s if x["expect_lead"]]  # all lead dialogues count; nothing is excluded
    dec = [x for x in s if not x["expect_lead"]]
    res = {"n": len(s), "n_lead_dialogues": len(lead),
           "sim_gave_all_fields": f"{sum(x['sim_gave_all'] for x in lead)}/{len(lead)}",
           "completion_rate": fmt_ci([x["success"] for x in lead]),
           "correct_final_lead": fmt_ci([x["correct_final_lead"] for x in lead]),
           "unsupported_registrations": sum(x["unsupported_registrations"] for x in s),
           "duplicate_registrations": sum(x["duplicate_registrations"] for x in s),
           "false_registrations_on_decline": f"{sum(x['false_registration'] for x in dec)}/{len(dec)}",
           "avg_turns_to_lead": float(np.mean([x["turns_to_lead"] for x in lead if x["turns_to_lead"]] or [np.nan])),
           "by_behaviour": {}}
    # product questions asked while lead collection was active, and how many got a retrieval-grounded answer
    q_turns = [t for d in rows for t in d["turns"] if "extract" in (t.get("trace") or [])
               and "?" in t["user"] and t.get("intent") == "product_inquiry"]
    res["mid_collection_questions"] = len(q_turns)
    res["mid_collection_answered"] = sum("rag" in t["trace"] for t in q_turns)
    for b in prompts.SIM_BEHAVIOURS:
        xs = [x for x in s if x["behaviour"] == b]
        res["by_behaviour"][b] = f"{sum(x['success'] for x in xs)}/{len(xs)}"
    for f in config.LEAD_FIELDS:  # field-level P/R/F1 over valid lead dialogues
        tp = sum(x["fields"][f]["ok"] for x in lead)
        pred = sum(bool(x["fields"][f]["pred"]) for x in lead)
        prec, rec = tp / max(pred, 1), tp / max(len(lead), 1)
        res[f"field_{f}"] = {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9)}
    return res, s


# ------------------------------------------------------------------ report

def _turn_cost_rows(rows):
    lat = [r["latency_s"] for r in rows]
    return {"turns": len(rows), "latency_mean_s": float(np.mean(lat)), "latency_p50_s": float(np.percentile(lat, 50)),
            "latency_p95_s": float(np.percentile(lat, 95)), "llm_calls": float(np.mean([r["llm_calls"] for r in rows])),
            "prompt_tokens": float(np.mean([r["prompt_tokens"] for r in rows])),
            "completion_tokens": float(np.mean([r["completion_tokens"] for r in rows]))}


def report(_):
    lines = ["# Results", ""]
    items = {q["id"]: q for q in read_jsonl(config.DATA_DIR / "qa.jsonl")}
    cats = ["pricing", "feature", "policy", "troubleshooting", "multi_hop", "followup", "unanswerable"]
    lines += ["## QA (judge correctness / 2, % [95% CI]; hallucination %; key-fact coverage %)", "",
              "| System | " + " | ".join(cats) + " | Overall | Halluc. | Key-fact |", "|" + "---|" * (len(cats) + 4)]
    qa_cost = {}
    for name in QA_SYSTEMS:
        p = RUNS / "qa" / f"{name}.judged.jsonl"
        if not p.exists():
            continue
        rows = [r for r in read_jsonl(p) if r["judge_correctness"] is not None]
        sc = lambda rs: [r["judge_correctness"] / 2 for r in rs]
        per = [fmt_ci(sc([r for r in rows if items[r["id"]]["category"] == c])) for c in cats]
        kf = [k for r in rows if (k := keyfact_coverage(r["answer"], items[r["id"]]["must_include"])) is not None]
        lines.append(f"| {name} | " + " | ".join(per) + f" | {fmt_ci(sc(rows))} | "
                     f"{fmt_ci([r['judge_hallucination'] for r in rows])} | {fmt_ci(kf)} |")
        qa_cost[name] = _turn_cost_rows(read_jsonl(RUNS / "qa" / f"{name}.jsonl"))
    lines += ["", "## Per-turn latency and tokens (QA run, sequential)", "",
              "| System | mean s | p50 s | p95 s | LLM calls | prompt tok | completion tok |", "|---|---|---|---|---|---|---|"]
    for n, c in qa_cost.items():
        lines.append(f"| {n} | {c['latency_mean_s']:.2f} | {c['latency_p50_s']:.2f} | {c['latency_p95_s']:.2f} | "
                     f"{c['llm_calls']:.2f} | {c['prompt_tokens']:.0f} | {c['completion_tokens']:.0f} |")
    for tag in ("", "_seq"):
        d = RUNS / f"dialogues{tag}"
        if not d.exists():
            continue
        lines += ["", f"## Dialogues{tag}", ""]
        for p in sorted(x for x in d.glob("*.jsonl") if not x.name.endswith(".scored.jsonl")):
            rows = read_jsonl(p)
            m, scored = dialogue_metrics(rows)
            write_jsonl(d / f"{p.stem}.scored.jsonl", scored)
            turns = [t for r in rows for t in r["turns"]]
            m["per_turn"] = _turn_cost_rows(turns) if turns else None
            lines += [f"### {p.stem}", "```", json.dumps(m, indent=2), "```"]
    for f in ("intent_metrics.json", "intent_nocontext_metrics.json", "retrieval_metrics.json"):
        if (RUNS / f).exists():
            lines += ["", f"## {f}", "```", (RUNS / f).read_text(), "```"]
    for f in ("intent_agreement.json", "qa_agreement.json"):
        if (ANNOT_DIR / f).exists():
            lines += ["", f"## {f}", "```", (ANNOT_DIR / f).read_text(), "```"]
    (RUNS / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("systems", nargs="*")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--per-behaviour", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--allow-unverified", action="store_true")
    a = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    {"finalize-intent": finalize_intent, "intent": eval_intent, "retrieval": eval_retrieval, "qa": run_qa,
     "judge": run_judge, "dialogues": run_dialogues, "annotation-export": annotation_export,
     "agreement": agreement, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
