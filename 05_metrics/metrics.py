"""Stage 05: Compute within-ecosystem redundancy metrics.

Reads data/embeddings.npy and data/embed_meta.parquet.
Computes the headline finding: share of packages with cosine similarity >= threshold T
to at least one other package in the same ecosystem.
Writes data/metrics_summary.json and appends numbers to FINDINGS.md.
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

log = logging.getLogger(__name__)

# Calibrate this threshold on a dev sample before the full run.
# A reasonable starting point for "near-duplicate" descriptions is 0.95.
SIMILARITY_THRESHOLD = 0.95


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute within-ecosystem redundancy metrics")
    parser.add_argument("--force", action="store_true", help="Rerun even if output exists")
    parser.add_argument(
        "--threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help="Cosine similarity threshold for near-duplicate detection",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s", stream=sys.stderr)

    out_path = DATA / "metrics_summary.json"
    if out_path.exists() and not args.force:
        print(f"Already done: {out_path}. Use --force to rerun.")
        return

    raise NotImplementedError("Stage 05 not yet implemented")


if __name__ == "__main__":
    main()
