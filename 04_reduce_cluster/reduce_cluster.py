"""Stage 04: Dimensionality reduction (UMAP) + clustering (HDBSCAN).

Reads data/embeddings.npy and data/embed_meta.parquet.
Writes data/umap_2d.npy and data/clusters.parquet (name, ecosystem, cluster_id, x, y).
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

log = logging.getLogger(__name__)

UMAP_RANDOM_SEED = 42
HDBSCAN_MIN_CLUSTER_SIZE = 10


def main() -> None:
    parser = argparse.ArgumentParser(description="UMAP reduction + HDBSCAN clustering")
    parser.add_argument("--force", action="store_true", help="Rerun even if output exists")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s", stream=sys.stderr)

    out_path = DATA / "clusters.parquet"
    if out_path.exists() and not args.force:
        print(f"Already done: {out_path}. Use --force to rerun.")
        return

    raise NotImplementedError("Stage 04 not yet implemented")


if __name__ == "__main__":
    main()
