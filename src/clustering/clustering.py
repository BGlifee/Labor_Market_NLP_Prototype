import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import normalize
import hdbscan
from ..config import load_config

def kmeans_clusters(embeddings: np.ndarray) -> np.ndarray:
    cfg = load_config()["clustering"]
    k = cfg["n_clusters"]
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    return labels

def hdbscan_clusters(embeddings: np.ndarray) -> np.ndarray:
    cfg = load_config()["clustering"]

    X = normalize(embeddings, norm="l2")   

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg["min_cluster_size"],
        min_samples=cfg.get("min_samples", 10),
        metric="euclidean",
        prediction_data=True,
    )
    return clusterer.fit_predict(X)

def add_cluster_labels(df: pd.DataFrame, labels: np.ndarray, col_name: str) -> pd.DataFrame:
    df = df.copy()
    df[col_name] = labels
    return df


def agglomerative_clusters(embeddings: np.ndarray, k: int | None = None) -> np.ndarray:
    cfg = load_config()["clustering"]
    k = k or cfg["n_clusters"]

    X = normalize(embeddings, norm="l2")

    model = AgglomerativeClustering(
        n_clusters=k,
        metric="cosine",
        linkage="average",
    )
    return model.fit_predict(X)

