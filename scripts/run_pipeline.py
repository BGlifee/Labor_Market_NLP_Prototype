from pathlib import Path
import numpy as np

from src.data.loader import load_jobs
from src.embeddings.embedder import build_and_save_embeddings
from src.clustering.clustering import kmeans_clusters, add_cluster_labels
from src.evaluation.metrics import internal_cluster_metrics
from src.taxonomy.mapping import (
    load_onet,
    build_onet_embeddings,
    map_clusters_to_soc,
)

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def main():
    # 1. Load jobs
    jobs = load_jobs(limit=None)
    print(f"Loaded {len(jobs)} job postings")

    # 2. Embeddings
    emb_path = RESULTS_DIR / "embeddings" / "job_embeddings.npy"
    emb_path.parent.mkdir(parents=True, exist_ok=True)

    if emb_path.exists():
        embeddings = np.load(emb_path)
        print(f"Loaded existing embeddings from {emb_path}")
    else:
        embeddings = build_and_save_embeddings(jobs, emb_path)
        print(f"Saved embeddings to {emb_path}")

    # 3. Clustering (K-means example)
    labels = kmeans_clusters(embeddings)
    jobs_with_clusters = add_cluster_labels(jobs, labels, "cluster_kmeans")

    clusters_path = RESULTS_DIR / "clusters"
    clusters_path.mkdir(exist_ok=True)
    jobs_with_clusters.to_parquet(
        clusters_path / "jobs_with_kmeans_clusters.parquet",
        index=False,
    )

    # 4. Internal metrics
    metrics = internal_cluster_metrics(embeddings, labels)
    print("Internal metrics:", metrics)

    # 5. SOC / O*NET mapping
    onet_df = load_onet()
    onet_emb = build_onet_embeddings(onet_df)

    centers = []
    for c in sorted(set(labels)):
        mask = labels == c
        centers.append(embeddings[mask].mean(axis=0))
    centers = np.vstack(centers)

    mapping_df = map_clusters_to_soc(centers, onet_df, onet_emb)

    mappings_dir = RESULTS_DIR / "mappings"
    mappings_dir.mkdir(exist_ok=True)
    mapping_df.to_json(
        mappings_dir / "cluster_soc_mapping.json",
        orient="records",
        indent=2,
    )

    print("Pipeline finished successfully.")

if __name__ == "__main__":
    main()
