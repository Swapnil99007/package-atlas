"""Stage 04: Dimensionality reduction (UMAP) + clustering (HDBSCAN).

Reads data/embeddings.npy and data/embed_meta.parquet.
Writes:
  data/umap_2d.npy      — float32 [N, 2], UMAP 2D projection
  data/clusters.parquet — name, ecosystem, cluster_id, x, y, download_count, upload_date

UMAP and HDBSCAN are both seed-sensitive; random_state=42 is fixed for reproducibility.
Noise points (unclustered by HDBSCAN) are labelled cluster_id=-1 and kept in output.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import umap

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

UMAP_RANDOM_STATE = 42

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="UMAP reduction + HDBSCAN clustering")
    parser.add_argument("--force", action="store_true", help="Rerun even if output exists")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.05)
    parser.add_argument("--min-cluster-size", type=int, default=20)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s", stream=sys.stderr)

    umap_path = DATA / "umap_2d.npy"
    clusters_path = DATA / "clusters.parquet"
    if umap_path.exists() and clusters_path.exists() and not args.force:
        print(f"Already done: {clusters_path}. Use --force to rerun.")
        return

    log.info("Loading embeddings and metadata …")
    embeddings = np.load(DATA / "embeddings.npy")
    meta = pd.read_parquet(DATA / "embed_meta.parquet")
    assert len(embeddings) == len(meta), "embeddings and meta row count mismatch"
    n = len(embeddings)

    # --- UMAP ---
    log.info(
        "Running UMAP: %d × %d → 2D (n_neighbors=%d, min_dist=%.2f, metric=cosine, seed=%d) …",
        n, embeddings.shape[1], args.n_neighbors, args.min_dist, UMAP_RANDOM_STATE,
    )
    t0 = time.perf_counter()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        random_state=UMAP_RANDOM_STATE,
        low_memory=False,
        verbose=False,
    )
    umap_2d = reducer.fit_transform(embeddings).astype(np.float32)
    umap_elapsed = time.perf_counter() - t0
    log.info("UMAP done in %.0fs", umap_elapsed)

    np.save(umap_path, umap_2d)
    log.info("Saved %s", umap_path)

    # --- HDBSCAN ---
    log.info(
        "Running HDBSCAN: min_cluster_size=%d, min_samples=5 …",
        args.min_cluster_size,
    )
    t1 = time.perf_counter()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(umap_2d)
    hdbscan_elapsed = time.perf_counter() - t1
    log.info("HDBSCAN done in %.0fs", hdbscan_elapsed)

    # --- Build output parquet ---
    df = meta.copy()
    df["cluster_id"] = labels.astype(int)
    df["x"] = umap_2d[:, 0]
    df["y"] = umap_2d[:, 1]
    df = df[["name", "ecosystem", "cluster_id", "x", "y", "download_count", "upload_date"]]
    df.to_parquet(clusters_path, index=False)
    log.info("Saved %s", clusters_path)

    # --- Summary ---
    n_clusters = int(labels.max()) + 1
    n_noise = int((labels == -1).sum())
    noise_pct = 100 * n_noise / n
    cluster_sizes = pd.Series(labels[labels >= 0]).value_counts()

    print(f"\nUMAP:   {n} points → 2D in {umap_elapsed:.0f}s")
    print(f"HDBSCAN: {n_clusters} clusters | {n_noise} noise points ({noise_pct:.1f}%)")
    if len(cluster_sizes):
        print(f"Largest cluster: {cluster_sizes.iloc[0]} packages | Smallest: {cluster_sizes.iloc[-1]} packages")

    for eco in ["pypi", "npm"]:
        mask = df["ecosystem"] == eco
        eco_noise = int((df.loc[mask, "cluster_id"] == -1).sum())
        eco_clustered = int(mask.sum()) - eco_noise
        print(f"  {eco}: {eco_clustered} clustered, {eco_noise} noise")


if __name__ == "__main__":
    main()
