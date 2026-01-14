import numpy as np
import pandas as pd
import re

def pick_medoids_simple(
    jobs: pd.DataFrame,
    embeddings: np.ndarray,
    labels: np.ndarray,
    text_col: str = "description"
) -> pd.DataFrame:
    """
    Pick 1 medoid (closest job to centroid) per cluster.
    """
    rows = []

    for cid in np.unique(labels):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue

        emb_c = embeddings[idx]
        centroid = emb_c.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        sims = emb_c @ centroid
        best = idx[np.argmax(sims)]

        row = jobs.iloc[best].to_dict()
        row.update({
            "cluster_id": int(cid),
            "job_row_index": int(best),
            "sim_to_centroid": float(np.max(sims)),
            "rep_text": jobs.iloc[best][text_col]
        })
        rows.append(row)

    return pd.DataFrame(rows)

def map_medoids_to_onet(
    medoids_df: pd.DataFrame,
    medoid_emb: np.ndarray,
    onet_df: pd.DataFrame,
    onet_emb: np.ndarray,
    top_k: int = 3
) -> pd.DataFrame:
    rows = []

    for i, row in medoids_df.iterrows():
        sims = onet_emb @ medoid_emb[i]
        top_idx = np.argsort(sims)[::-1][:top_k]

        for rank, j in enumerate(top_idx):
            rows.append({
                "cluster_id": int(row["cluster_id"]),
                "rank": rank + 1,
                "similarity": float(sims[j]),
                "soc_code": onet_df.iloc[j]["O*NET-SOC Code"],
                "soc_title": onet_df.iloc[j]["Title"],
                "onet_description": onet_df.iloc[j]["Description"],
                "job_id": row.get("job_id"),
                "rep_text": row["rep_text"],
            })

    return pd.DataFrame(rows)