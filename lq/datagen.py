"""Synthetic test-data generation.

  python -m lq.datagen intent     -> data/intent_pool.csv (generated labels) + data/annotation/intent_sheet.csv (blind)
  python -m lq.datagen personas   -> data/personas.jsonl
"""
import csv
import json
import random
import re
import sys

from . import config, llm, prompts

ANNOT_DIR = config.DATA_DIR / "annotation"


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def _jaccard(a: str, b: str) -> float:
    x, y = set(a.split()), set(b.split())
    return len(x & y) / max(len(x | y), 1)


def _parse_list(raw: str) -> list[str]:
    m = re.search(r"\[.*\]", raw, flags=re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        items = []
    return [s.strip() for s in items if isinstance(s, str) and 2 <= len(s.strip()) <= 300]


def gen_intent(per_call: int = 12, per_label: int = 90) -> None:
    from concurrent.futures import ThreadPoolExecutor

    rng = random.Random(config.SEED)
    pool = []
    jobs = [(label, si, style, rep) for label in prompts.INTENT_LABEL_DESC
            for si, style in enumerate(prompts.INTENT_GEN_STYLES) for rep in range(2)]

    def call(job):
        label, si, style, rep = job
        raw = llm.chat([{"role": "system", "content": prompts.INTENT_GEN_SYSTEM},
                        {"role": "user", "content": prompts.INTENT_GEN_USER.format(
                            n=per_call, label_desc=prompts.INTENT_LABEL_DESC[label], style=style)}],
                       node="gen_intent", aux=True, temperature=0.9, max_tokens=1200,
                       seed=config.SEED + 100 * si + rep)
        return [(t, style) for t in _parse_list(raw)]

    with ThreadPoolExecutor(8) as ex:
        results = dict(zip(jobs, ex.map(call, jobs)))
    for label in prompts.INTENT_LABEL_DESC:
        cands: list[tuple[str, str]] = [c for j in jobs if j[0] == label for c in results[j]]
        kept: list[tuple[str, str]] = []
        for t, style in cands:  # drop exact and near duplicates (token Jaccard >= 0.8)
            n = _norm(t)
            if n and all(_jaccard(n, _norm(k)) < 0.8 for k, _ in kept):
                kept.append((t, style))
        rng.shuffle(kept)
        print(f"{label}: {len(cands)} generated, {len(kept)} unique, keeping {min(per_label, len(kept))}")
        pool += [{"text": t, "prev_bot": "", "gen_label": label, "source": "generated", "style": s}
                 for t, s in kept[:per_label]]
    for line in (config.DATA_DIR / "intent_context.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        pool.append({"text": d["text"], "prev_bot": d["prev_bot"], "gen_label": d["label"],
                     "source": "context_handwritten", "style": "context"})
    rng.shuffle(pool)
    for i, row in enumerate(pool):
        row["id"] = f"I{i:03d}"

    cols = ["id", "prev_bot", "text", "gen_label", "source", "style"]
    with open(config.DATA_DIR / "intent_pool.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(pool)
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    for who in ("A", "B"):  # blind sheets: no generated label shown
        with open(ANNOT_DIR / f"intent_sheet_{who}.csv", "w", newline="", encoding="utf-8-sig") as f:  # BOM for Excel
            w = csv.writer(f)
            w.writerow(["id", "prev_bot", "text", "label (greeting/product_inquiry/high_intent/general)"])
            w.writerows([[r["id"], r["prev_bot"], r["text"], ""] for r in pool])
    print(f"wrote {len(pool)} items")


FIRST = ["Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Sneha", "Arjun", "Kavya", "Ishaan", "Meera", "Aditya",
         "Pooja", "Siddharth", "Neha", "Rahul", "Emma", "Liam", "Sofia", "Lucas", "Chloe", "Mateo", "Hannah",
         "Noah", "Aisha", "Yuki", "Omar", "Olivia", "Ethan", "Zara", "Daniel", "Mia", "Jean-Luc", "Fatima",
         "Wei", "Isabella", "Kwame", "Ana", "Tom", "Nina", "Leo"]
LAST = ["Mehta", "Nair", "Gupta", "Iyer", "Singh", "Reddy", "Kapoor", "Menon", "Verma", "Joshi", "Rao", "Bhatt",
        "Das", "Kulkarni", "Chauhan", "Wilson", "O'Connor", "Garcia", "Martin", "Dubois", "Rossi", "Schmidt", "Kim",
        "Bello", "Tanaka", "Haddad", "Brown", "Clarke", "Ahmed", "Mensah", "Novak", "Moreau", "Zahra", "Chen",
        "Costa", "Asante", "Lima", "Becker", "Petrova", "Fischer"]
DOMAINS = ["gmail.com", "outlook.com", "yahoo.co.in", "proton.me", "hotmail.com", "icloud.com"]
PLATFORMS = {"YouTube": "yt", "Instagram": "insta", "TikTok": "tiktok", "Facebook": "fb",
             "LinkedIn": "linkedin", "Twitch": "twitch", "X": "twitter"}


def _email(first: str, last: str, rng: random.Random) -> str:
    f, l = re.sub(r"[^a-z]", "", first.lower()), re.sub(r"[^a-z]", "", last.lower())
    local = rng.choice([f"{f}.{l}", f"{f}{l}{rng.randint(1, 99)}", f"{f[0]}.{l}", f"{f}_{l[0]}", f"{l}.{f}"])
    return f"{local}@{rng.choice(DOMAINS)}"


def _typo(email: str, rng: random.Random) -> str:
    local, dom = email.split("@")
    i = rng.randrange(1, len(local) - 1)
    return f"{local[:i]}{local[i + 1]}{local[i]}{local[i + 2:]}@{dom}" if local[i] != local[i + 1] else f"{local}x@{dom}"


def gen_personas(seed: int = config.TEST_PERSONA_SEED, filename: str = "personas.jsonl", per_behaviour: int = 25) -> None:
    """Test personas use TEST_PERSONA_SEED; the pilot/development set used seed 42 (data/personas_dev.jsonl)."""
    rng = random.Random(seed)
    out = []
    for b in prompts.SIM_BEHAVIOURS:
        for k in range(per_behaviour):
            first, last = rng.choice(FIRST), rng.choice(LAST)
            name = first if rng.random() < 0.1 else f"{first} {last}"  # ~10% single-word names
            email = _email(first, last, rng)
            platform = rng.choice(list(PLATFORMS))
            out.append({"id": f"D{len(out):03d}", "behaviour": b, "name": name, "email": email,
                        "platform": platform, "platform_alias": PLATFORMS[platform], "typo_email": _typo(email, rng),
                        "expect_lead": b != "decline", "seed": seed + len(out)})
    with open(config.DATA_DIR / filename, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(p) + "\n" for p in out)
    print(f"wrote {len(out)} personas to {filename} (seed {seed})")


if __name__ == "__main__":
    if sys.argv[1] == "personas":
        gen_personas(*(int(a) if i == 0 else a for i, a in enumerate(sys.argv[2:4])))
    else:
        gen_intent()
