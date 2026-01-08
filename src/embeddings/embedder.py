from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from ..config import load_config
from pathlib import Path
from typing import Iterable
from typing import Optional, List


MODEL = None  # lazy load

def get_model() -> SentenceTransformer:
    global MODEL
    if MODEL is None:
        cfg = load_config()["embeddings"]
        MODEL = SentenceTransformer(cfg["model_name"], device=cfg["device"])
    return MODEL

def embed_texts(texts: Iterable[str]) -> np.ndarray:
    cfg = load_config()["embeddings"]
    model = get_model()
    emb = model.encode(
        list(texts),
        batch_size=cfg["batch_size"],
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb

def build_and_save_embeddings(
    df: pd.DataFrame,
    out_path: str,
    texts: Optional[List[str]] = None
) -> np.ndarray:
    """
    Build embeddings and save them to disk.

    - If `texts` is provided, it will be embedded directly (recommended for preprocessed text).
    - Otherwise, uses cfg["jobs"]["text_column"] from config.toml.
    """
    cfg = load_config()

    if texts is None:
        text_col = cfg["jobs"]["text_column"]
        texts = df[text_col].fillna("").astype(str).tolist()

    embeddings = embed_texts(texts)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)

    return embeddings
