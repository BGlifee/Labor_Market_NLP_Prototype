import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

def internal_cluster_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    metric: str = "cosine",   # cosine recommended for sentence embeddings
) -> dict:
    """Compute internal clustering metrics (ignores noise label -1)."""
    mask = labels >= 0
    X = embeddings[mask]
    y = labels[mask]

    if len(np.unique(y)) < 2:
        return {"silhouette": None, "davies_bouldin": None, "n_clusters": int(len(np.unique(y)))}

    # silhouette supports cosine; DB supports only euclidean in sklearn
    sil = silhouette_score(X, y, metric=metric) if metric else silhouette_score(X, y)
    db = davies_bouldin_score(X, y)

    return {"silhouette": float(sil), "davies_bouldin": float(db), "n_clusters": int(len(np.unique(y)))}



def mapping_metrics(mapping_df, labels: np.ndarray, similarity_threshold: float = 0.0) -> dict:
    """
    Evaluate taxonomy mapping quality.
    Expects mapping_df with columns:
      - cluster_id
      - soc_matches: list[(soc_code, similarity)] OR empty list / None
    """
    # clusters that exist (excluding noise)
    valid = labels[labels >= 0]
    cluster_ids = np.unique(valid)
    n_clusters = len(cluster_ids)

    # noise rate
    noise_rate = float(np.mean(labels < 0))

    # coverage + similarity stats
    top1_sims = []
    top1_margins = []
    matched = 0

    for _, row in mapping_df.iterrows():
        matches = row.get("soc_matches", None) or []
        if len(matches) == 0:
            continue

        # assume matches already sorted desc by similarity
        top1 = matches[0][1]
        top2 = matches[1][1] if len(matches) > 1 else None

        if top1 is not None and top1 >= similarity_threshold:
            matched += 1
            top1_sims.append(top1)
            if top2 is not None:
                top1_margins.append(top1 - top2)

    coverage = matched / n_clusters if n_clusters else 0.0

    return {
        "n_clusters": int(n_clusters),
        "noise_rate": noise_rate,
        "coverage": float(coverage),
        "avg_top1_similarity": float(np.mean(top1_sims)) if top1_sims else None,
        "avg_top1_margin": float(np.mean(top1_margins)) if top1_margins else None,
    }