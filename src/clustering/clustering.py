import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
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
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg["min_cluster_size"],
        metric="euclidean",
        cluster_selection_epsilon=0.0,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(embeddings)
    return labels

def add_cluster_labels(df: pd.DataFrame, labels: np.ndarray, col_name: str) -> pd.DataFrame:
    df = df.copy()
    df[col_name] = labels
    return df
