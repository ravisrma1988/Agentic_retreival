"""Rule-based lead-field extractor (name, email, platform) plus decline / confirmation detectors."""
import re

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Alias -> canonical platform. Longer aliases are matched first.
PLATFORM_ALIASES = {
    "youtube": "YouTube", "you tube": "YouTube", "yt": "YouTube",
    "instagram": "Instagram", "insta": "Instagram", "ig": "Instagram", "reels": "Instagram",
    "tiktok": "TikTok", "tik tok": "TikTok", "tt": "TikTok",
    "facebook": "Facebook", "fb": "Facebook",
    "linkedin": "LinkedIn", "linked in": "LinkedIn",
    "twitter": "X", "x.com": "X", "x (twitter)": "X",
    "twitch": "Twitch",
    "snapchat": "Snapchat", "snap": "Snapchat",
    "vimeo": "Vimeo",
}
_PLATFORM_RE = re.compile(
    r"(?<![\w.@])(" + "|".join(re.escape(a) for a in sorted(PLATFORM_ALIASES, key=len, reverse=True)) + r")(?![\w@])",
    re.I,
)
# "X" is ambiguous as a word, so it is accepted only as a capitalised standalone token,
# or in lower case when the assistant has just asked for the platform.
_UPPER_X_RE = re.compile(r"(?<![\w.@'])X(?![\w@'])")
_LOWER_X_RE = re.compile(r"(?:^|\bon\s+|\bplatform\s+|\bit'?s\s+)x\b(?![\w@'])", re.I)

_NAME_TAIL = r"\s+([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,2})"
_NAME_IS_RE = re.compile(r"\b(?:my name is|my name's|name is|call me)" + _NAME_TAIL, re.I)
# "I'm <Name>" is accepted only when the name is capitalised (rules out "I'm super excited"),
# unless the assistant has just asked for the name.
_I_AM_RE = re.compile(r"\b(?:i am|i'm|im)" + _NAME_TAIL, re.I)
_WEAK_NAME_RE = re.compile(r"\b(?:this is|it's|its|it is)" + _NAME_TAIL, re.I)
_NOT_NAMES = {
    "interested", "looking", "a", "an", "the", "here", "ready", "on", "not", "just", "fine", "good", "sure",
    "glad", "happy", "keen", "planning", "thinking", "trying", "new", "from", "with", "using", "creating",
    "making", "posting", "still", "also", "really", "very", "so", "currently", "mostly", "mainly", "actually",
    "sorry", "okay", "ok", "yes", "no", "in", "at", "into", "going", "want", "wanting", "hoping", "excited",
    "super", "considering", "curious", "wondering", "a", "pretty", "quite", "totally", "definitely", "all",
    "back", "done", "set", "after", "after", "about", "currently", "based", "one", "your", "my", "great",
}
_FILLER_RE = re.compile(r"^(?:sure|ok(?:ay)?|yes|yeah|yep|hi|hello|hey|sorry)[,.!\s]+", re.I)
_STOP_AFTER_NAME = re.compile(r"(?:\s+(?:and|but|from|i|my|here|on|by|of|who|what|is|the)\b|[,.!;:?]).*$", re.I)

_DECLINE_RE = re.compile(
    r"\b(?:rather not|don'?t want to (?:share|give)|won'?t (?:share|give)|not comfortable|no thanks|"
    r"not now|never mind|nevermind|prefer not|not interested|skip (?:it|this)|i'?ll pass)\b",
    re.I,
)
_AFFIRM_RE = re.compile(r"\b(?:yes|yep|yeah|yup|correct|confirm(?:ed)?|right|looks good|all good|perfect|"
                        r"go ahead|sure|ok(?:ay)?|exactly|accurate)\b", re.I)
_NEGATE_RE = re.compile(r"\b(?:no|nope|not|wrong|incorrect|change|mistake|typo|actually|fix)\b", re.I)


def extract_email(text: str) -> str | None:
    m = EMAIL_RE.findall(text)
    return m[-1].rstrip(".").lower() if m else None  # last one wins, so corrections override typos


def extract_platform(text: str, awaiting: str | None = None) -> str | None:
    t = EMAIL_RE.sub(" ", text)
    m = _PLATFORM_RE.findall(t)
    if m:
        return PLATFORM_ALIASES[m[-1].lower()]
    if _UPPER_X_RE.search(t) or (awaiting == "platform" and _LOWER_X_RE.search(t.strip())):
        return "X"
    return None


def _clean_name(raw: str) -> str | None:
    raw = _STOP_AFTER_NAME.sub("", raw.strip())
    words = [w for w in re.split(r"\s+", raw) if w]
    if not words or len(words) > 3 or words[0].lower() in _NOT_NAMES:
        return None
    if _PLATFORM_RE.fullmatch(raw) or raw == "X":
        return None
    if any(not re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", w) for w in words):
        return None
    return " ".join(w[:1].upper() + w[1:] for w in words)


def extract_name(text: str, awaiting: str | None = None) -> str | None:
    t = EMAIL_RE.sub(" ", text)
    for m in _NAME_IS_RE.finditer(t):
        if name := _clean_name(m.group(1)):
            return name
    for m in _I_AM_RE.finditer(t):
        if (awaiting == "name" or m.group(1)[0].isupper()) and (name := _clean_name(m.group(1))):
            return name
    if awaiting == "name":
        for m in _WEAK_NAME_RE.finditer(t):
            if name := _clean_name(m.group(1)):
                return name
        # A bare reply to "Could you share your name?" is taken as the name.
        bare = _FILLER_RE.sub("", t.strip()).strip(" .!")
        if bare and not _PLATFORM_RE.search(bare) and len(bare.split()) <= 3:
            return _clean_name(bare)
    return None


def is_decline(text: str) -> bool:
    return bool(_DECLINE_RE.search(text))


def is_affirmative(text: str) -> bool:
    return bool(_AFFIRM_RE.search(text)) and not _NEGATE_RE.search(text)


def is_negative(text: str) -> bool:
    return bool(_NEGATE_RE.search(text)) and not _AFFIRM_RE.search(text)


def extract_fields(text: str, awaiting: str | None = None) -> dict:
    return {
        "name": extract_name(text, awaiting),
        "email": extract_email(text),
        "platform": extract_platform(text, awaiting),
    }
