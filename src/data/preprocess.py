import re
import pandas as pd
from typing import Optional

def normalize_text(s: str) -> str:
    """
    Fix broken encodings and normalize whitespace.
    Must be applied before any regex-based cleaning.
    """
    if s is None:
        return ""

    s = str(s)
    s = (
        s.replace("â€™", "'")
         .replace("â€œ", '"')
         .replace("â€", '"')
         .replace("â€“", "-")
         .replace("â€”", "-")
    )

    s = re.sub(r"\s+", " ", s).strip()
    return s

SECTION_START = re.compile(
    r"\b(responsibilities|what you(?:'| a)?ll do|duties|role|job description|requirements|qualifications|what we(?:'| a)?re looking for|skills|experience)\b",
    flags=re.IGNORECASE
)

SECTION_END = re.compile(
    r"\b(equal opportunity|eeo|diversity|accommodation|reasonable accommodation|background check|drug test|right to work|i-9|privacy|disclaimer|terms of employment)\b",
    flags=re.IGNORECASE
)

def keep_job_sections(text: str, min_chars: int = 400) -> str:
    """
    Keep the main job-related section and discard legal/EEO parts.
    If extracted section is too short, fall back to full text.
    """
    t = normalize_text(text)

    m = SECTION_START.search(t)
    if not m:
        return t

    tail = t[m.start():]
    e = SECTION_END.search(tail)
    cut = tail[:e.start()] if e else tail
    cut = cut.strip()

    return cut if len(cut) >= min_chars else t

BOILER_PATTERNS = [
    r"\bequal opportunity employer\b.*",
    r"\beeo\b.*",
    r"\bwe are an equal opportunity\b.*",
    r"\bdiversity\b.*\binclusion\b.*",
    r"\breasonable accommodation\b.*",
    r"\baccommodation\b.*\bdisability\b.*",
    r"\bbackground check\b.*",
    r"\bdrug test\b.*",
    r"\b401k\b.*",
    r"\bhealth insurance\b.*",
    r"\bdental\b.*\bvision\b.*",
    r"\bbenefits\b.*",
    r"\babout (us|the company)\b.*",
    r"\bcompany overview\b.*",
    r"\bour mission\b.*",
    r"\bwho we are\b.*",
    r"\bprivacy policy\b.*",
    r"\bdisclaimer\b.*",
]

BOILER_REGEX = re.compile(
    "|".join(f"(?:{p})" for p in BOILER_PATTERNS),
    flags=re.IGNORECASE
)

def remove_boilerplate(text: str) -> str:
    """
    Remove common company, legal, and benefits boilerplate.
    """
    t = normalize_text(text)
    t = BOILER_REGEX.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

CUTOFF_HEADERS = [
    r"\beeoc?\b",
    r"\bequal opportunity\b",
    r"\bhow to apply\b",
    r"\bbenefits\b",
    r"\bcompensation\b",
    r"\badditional information\b",
    r"\baccommodation\b",
    r"\blegal\b",
]

def clean_for_embedding(text: Optional[str]) -> str:
    """
    Light preprocessing for SentenceTransformer embeddings.
    Preserves semantic content while removing repetitive boilerplate.
    """
    if text is None:
        return ""

    t = str(text)

    # Remove HTML
    t = re.sub(r"<[^>]+>", " ", t)

    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()

    # Keep very short texts unchanged
    if len(t) < 50:
        return t

    # Cut off after boilerplate-style headers
    lower = t.lower()
    for h in CUTOFF_HEADERS:
        m = re.search(h, lower)
        if m and m.start() > 200:
            t = t[: m.start()].strip()
            lower = t.lower()
            break

    # Remove boilerplate patterns
    for pat in BOILERPLATE_PATTERNS:
        t = re.sub(pat, " ", t, flags=re.IGNORECASE | re.DOTALL)

    return re.sub(r"\s+", " ", t).strip()

def clean_job_text(text: str) -> str:
    """
    Full cleaning pipeline for job descriptions:
    section trimming → boilerplate removal → length cap.
    """
    t = keep_job_sections(text)
    t = remove_boilerplate(t)
    t = t[:3000]   # hard cap for transformer stability
    return t

def build_embed_text(
    df: pd.DataFrame,
    title_col: str = "title",
    desc_col: str = "description"
) -> pd.Series:
    """
    Build embedding input text by combining title + cleaned description.
    """
    title = df[title_col].fillna("").astype(str)
    desc = df[desc_col].fillna("").astype(str)

    combined = (title.str.strip() + ". " + desc.str.strip()).str.strip()
    return combined.apply(clean_for_embedding)
