import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

def internal_cluster_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict:

    mask = labels != -1
    X = embeddings[mask]
    y = labels[mask]

    if len(np.unique(y)) < 2:
        return {"silhouette": None, "davies_bouldin": None}

    sil = silhouette_score(X, y)
    db = davies_bouldin_score(X, y)
    return {"silhouette": sil, "davies_bouldin": db}
