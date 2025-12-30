import numpy as np
import pandas as pd
from sentence_transformers import util
from ..config import load_config
from ..embeddings.embedder import embed_texts

def load_onet() -> pd.DataFrame:
    cfg = load_config()["taxonomy"]
    df = pd.read_csv(cfg["onet_path"])
    needed = [cfg["title_column"], cfg["desc_column"], cfg["code_column"]]
    return df[needed].dropna()

def build_onet_embeddings(onet_df: pd.DataFrame) -> np.ndarray:
    cfg = load_config()["taxonomy"]
    texts = (
        onet_df[cfg["title_column"]] + " - " + onet_df[cfg["desc_column"]]
    ).tolist()
    emb = embed_texts(texts)
    return emb

def map_clusters_to_soc(
    cluster_centers: np.ndarray,
    onet_df: pd.DataFrame,
    onet_emb: np.ndarray,
) -> pd.DataFrame:
    cfg = load_config()["taxonomy"]
    top_k = cfg["top_k"]
    thr = cfg["similarity_threshold"]

    sims = util.cos_sim(cluster_centers, onet_emb).cpu().numpy()
    codes = onet_df[cfg["code_column"]].values

    results = []
    for i, row in enumerate(sims):
        idx = np.argsort(row)[::-1][:top_k]
        top_codes = codes[idx]
        top_scores = row[idx]
        filtered = [
            (c, float(s)) for c, s in zip(top_codes, top_scores) if s >= thr
        ]
        results.append({"cluster_id": i, "soc_matches": filtered})
    return pd.DataFrame(results)
