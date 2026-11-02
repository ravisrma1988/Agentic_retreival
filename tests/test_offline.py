"""Offline checks that need no GPU or LLM: KB consistency with the QA gold set, chunking, extractor."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lq import config  # noqa: E402
from lq.evaluate import keyfact_coverage, norm_text, score_dialogue  # noqa: E402
from lq.extractor import extract_fields, is_decline  # noqa: E402
from lq.kb import chunk_docs, load_docs  # noqa: E402


def test_qa_gold_consistent_with_kb():
    docs = {d["doc_id"]: norm_text(d["text"]) for d in load_docs()}
    items = [json.loads(l) for l in (config.DATA_DIR / "qa.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len({q["id"] for q in items}) == len(items)
    computed = {"M01", "M02", "M03", "M04"}  # answers require arithmetic, not verbatim in KB
    for q in items:
        for s in q["sources"]:
            assert s in docs, (q["id"], s)
        if not q["answerable"] or q["id"] in computed:
            continue
        src = " ".join(docs[s] for s in q["sources"])
        for grp in q["must_include"]:
            # at least one alternative of each key fact must be stated in the gold source documents,
            # or be an answer-polarity word ("yes"/"no,") that the KB expresses differently
            ok = any(a in src for a in grp) or any(a in ("yes", "no,") for a in grp) or q["category"] in ("multi_hop",)
            assert ok or keyfact_coverage(q["gold"], [grp]) == 1, (q["id"], grp)


def test_chunks():
    ch = chunk_docs(load_docs())
    assert len(ch) > 60
    assert all(len(c["text"].split()) <= config.CHUNK_WORDS + 12 for c in ch)


def test_extractor():
    cases = [
        ("Ravi Sharma", "name", {"name": "Ravi Sharma"}),
        ("sure, it's priya nair", "name", {"name": "Priya Nair"}),
        ("my name is Liam O'Connor and I make cooking videos", None, {"name": "Liam O'Connor"}),
        ("I'm interested in the Pro plan", None, {"name": None}),
        ("it's youtube", "platform", {"name": None, "platform": "YouTube"}),
        ("mostly on insta", "platform", {"platform": "Instagram"}),
        ("X", "platform", {"platform": "X"}),
        ("x", "name", {"platform": None}),
        ("sorry, my email is actually priya.nair@gmail.com", "platform", {"email": "priya.nair@gmail.com"}),
        ("I want to subscribe to Pro for my YouTube channel", None, {"platform": "YouTube", "name": None}),
        # lower-case "i'm <x>" is only read as a name when the assistant has just asked for the name
        ("i'm aarav from delhi, email aarav.m@outlook.com, i post on tiktok", "name",
         {"name": "Aarav", "email": "aarav.m@outlook.com", "platform": "TikTok"}),
        ("i'm aarav from delhi", None, {"name": None}),
    ]
    for text, awaiting, expect in cases:
        got = extract_fields(text, awaiting)
        for k, v in expect.items():
            assert got[k] == v, (text, k, got[k], v)
    assert is_decline("I'd rather not share my email")
    assert not is_decline("sure, here it is")
    # regressions found in the pilot (development) run
    more = [
        ("I create content on X.", "platform", {"platform": "X"}),
        ("I create content on platform X.", None, {"platform": "X"}),
        ("on x mostly", "platform", {"platform": "X"}),
        ("I'm super excited to sign up! I stream on Twitch", None, {"name": None, "platform": "Twitch"}),
        ("Hey! I'm super excited. My name is Mateo O'Connor and I vlog", None, {"name": "Mateo O'Connor"}),
        ("I'm Zara and I post on LinkedIn", None, {"name": "Zara", "platform": "LinkedIn"}),
    ]
    for text, awaiting, expect in more:
        got = extract_fields(text, awaiting)
        for k, v in expect.items():
            assert got[k] == v, (text, k, got[k], v)
    from lq.extractor import is_affirmative, is_negative
    assert is_affirmative("Yes, that's correct.") and is_affirmative("yep looks good")
    assert not is_affirmative("no, the email is wrong") and is_negative("no, the email is wrong")
    assert not is_affirmative("that's incorrect")


def test_score_dialogue():
    p = {"id": "D0", "behaviour": "cooperative", "name": "Priya Nair", "email": "p.nair@gmail.com",
         "platform": "Instagram", "platform_alias": "insta", "expect_lead": True}
    turns = [{"t": 0, "user": "I want Pro", "n_registrations": 0},
             {"t": 1, "user": "Priya Nair", "n_registrations": 0},
             {"t": 2, "user": "p.nair@gmail.com", "n_registrations": 0},
             {"t": 3, "user": "insta", "n_registrations": 1}]
    good = score_dialogue({"persona": p, "turns": turns,
                           "registrations": [{"name": "Priya Nair", "email": "p.nair@gmail.com", "platform": "Instagram"}]})
    assert good["success"] and good["turns_to_lead"] == 4 and good["unsupported_registrations"] == 0
    early = [dict(t, n_registrations=1) if t["t"] == 1 else t for t in turns]
    bad = score_dialogue({"persona": p, "turns": early,
                          "registrations": [{"name": "Priya Nair", "email": "p.nair@gmail.com", "platform": "Instagram"}]})
    assert bad["unsupported_registrations"] == 1 and not bad["success"]
    # user typed only a first name; the surname appears only inside the email address
    p2 = dict(p, name="Priya Haddad", email="priya.haddad@hotmail.com", platform="Twitch", platform_alias="twitch")
    t2 = [{"t": 0, "user": "want pro for twitch", "n_registrations": 0},
          {"t": 1, "user": "priya", "n_registrations": 0},
          {"t": 2, "user": "priya.haddad@hotmail.com", "n_registrations": 1}]
    first_only = score_dialogue({"persona": p2, "turns": t2, "registrations": [
        {"name": "Priya", "email": "priya.haddad@hotmail.com", "platform": "Twitch"}]})
    assert first_only["success"], first_only
    from_email = score_dialogue({"persona": p2, "turns": t2, "registrations": [
        {"name": "Priya Haddad", "email": "priya.haddad@hotmail.com", "platform": "Twitch"}]})
    assert from_email["unsupported_registrations"] == 1  # surname was never typed as a name


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
