"""Stage 06: Build the interactive embedding atlas visualization.

Reads data/clusters.parquet and data/metrics_summary.json.
Prototype: export to a format compatible with Apple Embedding Atlas or Nomic Atlas.
Week-2 upgrade: custom deck.gl / regl-scatterplot frontend (decide after finding is confirmed).
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build interactive visualization")
    parser.add_argument("--force", action="store_true", help="Rerun even if output exists")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s", stream=sys.stderr)

    raise NotImplementedError("Stage 06 not yet implemented")


if __name__ == "__main__":
    main()
