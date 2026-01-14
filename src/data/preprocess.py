import re
import pandas as pd
from typing import Optional, Literal


# =========================
# 0) Common normalization
# =========================

def normalize_text(s: str) -> str:
    """
    Normalize corrupted characters and collapse whitespace.
    """
    if s is None:
        return ""
    s = str(s)
    s = (s.replace("â€™", "'")
           .replace("â€œ", '"')
           .replace("â€", '"')
           .replace("â€“", "-")
           .replace("â€”", "-"))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def strip_html(t: str) -> str:
    """
    Remove HTML tags.
    """
    return re.sub(r"<[^>]+>", " ", t)


def apply_regex_sub(t: str, rx: re.Pattern) -> str:
    """
    Apply a compiled regex and normalize whitespace.
    """
    t = rx.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


# =========================
# 1) JOBS cleaning logic
# =========================

# If your project already defines SECTION_START / SECTION_END, use those.
# These defaults work for most job postings.
SECTION_START = re.compile(
    r"\b(job description|responsibilities|what you will do|duties|role|position summary)\b",
    flags=re.IGNORECASE
)

SECTION_END = re.compile(
    r"\b(equal opportunity|eeo|how to apply|benefits|compensation|accommodation|legal|privacy)\b",
    flags=re.IGNORECASE
)


def keep_job_sections(text: str, min_chars: int = 400) -> str:
    """
    Keep the main job-related section and remove legal/EEO and footer blocks.
    If the extracted section is too short, fall back to the full text.
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


# Boilerplate patterns commonly found in job ads
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
    flags=re.IGNORECASE | re.DOTALL
)

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


def clean_jobs_text(text: Optional[str], cap: int = 3000) -> str:
    """
    Full cleaning pipeline for job postings:
    section trimming → boilerplate removal → length cap.
    """
    if text is None:
        return ""

    t = str(text)
    t = strip_html(t)
    t = normalize_text(t)

    # Preserve very short postings
    if len(t) < 50:
        return t

    # Keep main job content
    t = keep_job_sections(t)

    # Cut off at legal/boilerplate headers if they appear late in the text
    lower = t.lower()
    for h in CUTOFF_HEADERS:
        m = re.search(h, lower)
        if m and m.start() > 200:
            t = t[: m.start()].strip()
            lower = t.lower()
            break

    # Remove boilerplate blocks
    t = apply_regex_sub(t, BOILER_REGEX)

    return t[:cap]


# =========================
# 2) O*NET cleaning logic
# =========================

ONET_BOILER_PATTERNS = [
    r"\bexamples? of\b.*",
    r"\bmay include\b.*",
    r"\brelated occupations?\b.*",
    r"\bfor more information\b.*",
    r"\bvisit\b.*\bwww\..*",
    r"https?://\S+",
]

ONET_BOILER_REGEX = re.compile(
    "|".join(f"(?:{p})" for p in ONET_BOILER_PATTERNS),
    flags=re.IGNORECASE | re.DOTALL
)


def clean_onet_description(desc: Optional[str], cap: int = 2000) -> str:
    """
    Light cleaning for O*NET descriptions.
    Removes reference-style noise while preserving semantic content.
    """
    if desc is None:
        return ""

    t = str(desc)
    t = strip_html(t)
    t = normalize_text(t)

    # Remove O*NET reference boilerplate
    t = apply_regex_sub(t, ONET_BOILER_REGEX)

    return t[:cap]


def clean_onet_title(title: Optional[str]) -> str:
    """
    Clean O*NET job titles (optional).
    Removes parenthetical clarifications.
    """
    if title is None:
        return ""

    t = normalize_text(title)
    t = re.sub(r"\s*\([^)]*\)\s*", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# =========================
# 3) Unified entry point
# =========================

CleanMode = Literal["jobs", "onet_desc", "onet_title"]

def clean_text(text: Optional[str], mode: CleanMode = "jobs") -> str:
    """
    Unified text cleaning interface.

    - mode="jobs"       → job postings
    - mode="onet_desc"  → O*NET descriptions
    - mode="onet_title" → O*NET titles
    """
    if mode == "jobs":
        return clean_jobs_text(text)
    if mode == "onet_desc":
        return clean_onet_description(text)
    if mode == "onet_title":
        return clean_onet_title(text)
    raise ValueError(f"Unknown mode: {mode}")