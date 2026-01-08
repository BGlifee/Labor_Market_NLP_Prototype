import re
import pandas as pd
from typing import Optional

# Common boilerplate patterns in job postings
BOILERPLATE_PATTERNS = [
    r"equal opportunity employer.*",
    r"we are an equal opportunity employer.*",
    r"e-?verify.*",
    r"background check.*",
    r"drug[- ]?free workplace.*",
    r"reasonable accommodation.*",
    r"accommodations? .* disabilities.*",
    r"applicants? .* disabilities.*",
    r"affirmative action.*",
    r"we do not discriminate.*",
    r"compensation.* benefits.*(package|include).*",
    r"benefits include.*",
    r"apply (now|today).*",
    r"how to apply.*",
    r"click (here|apply).*",
    r"visit our website.*",
    r"any unsolicited resumes.*",
    r"by applying.* you agree.*",
    r"privacy policy.*",
]

# Section headers after which content is often boilerplate
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

    # Remove HTML tags
    t = re.sub(r"<[^>]+>", " ", t)

    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()

    # Keep very short texts unchanged
    if len(t) < 50:
        return t

    # Cut off text after boilerplate-style headers
    lower = t.lower()
    for h in CUTOFF_HEADERS:
        m = re.search(h, lower)
        if m and m.start() > 200:
            t = t[: m.start()].strip()
            lower = t.lower()
            break

    # Remove common boilerplate patterns
    for pat in BOILERPLATE_PATTERNS:
        t = re.sub(pat, " ", t, flags=re.IGNORECASE | re.DOTALL)

    return re.sub(r"\s+", " ", t).strip()

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
