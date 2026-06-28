"""Single entry point to run all pipeline stages end-to-end.

Usage:
    python run_pipeline.py                    # run all stages
    python run_pipeline.py --stages 01,02     # run specific stages
    python run_pipeline.py --limit 100        # dev subset (passed to each stage)
    python run_pipeline.py --force            # bypass cache guards in every stage
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

STAGE_MODULES = {
    "01_pypi": "01_fetch.fetch_pypi",
    "01_npm":  "01_fetch.fetch_npm",
    "02":      "02_clean.clean",
    "03":      "03_embed.embed",
    "04":      "04_reduce_cluster.reduce_cluster",
    "05":      "05_metrics.metrics",
    "06":      "06_visualize.visualize",
}

# Map user-facing stage keys to argument lists each module's main() accepts.
# Each stage's main() reads sys.argv, so we patch it before calling.
STAGE_ARGS: dict[str, list[str]] = {k: [] for k in STAGE_MODULES}


def run_stage(key: str, module_path: str, extra_argv: list[str]) -> None:
    mod = importlib.import_module(module_path)
    old_argv = sys.argv[:]
    sys.argv = [module_path] + extra_argv
    try:
        t0 = time.perf_counter()
        mod.main()
        elapsed = time.perf_counter() - t0
        print(f"  [{key}] done in {elapsed:.1f}s")
    finally:
        sys.argv = old_argv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the package-atlas pipeline")
    parser.add_argument(
        "--stages",
        default=",".join(STAGE_MODULES.keys()),
        help="Comma-separated stage keys to run (default: all). "
             f"Available: {', '.join(STAGE_MODULES.keys())}",
    )
    parser.add_argument("--force", action="store_true", help="Bypass cache guards in all stages")
    parser.add_argument("--limit", type=int, default=0, help="Dev subset size (0 = full)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in requested if s not in STAGE_MODULES]
    if unknown:
        parser.error(f"Unknown stage(s): {unknown}. Available: {list(STAGE_MODULES.keys())}")

    extra: list[str] = ["--log-level", args.log_level]
    if args.force:
        extra.append("--force")
    if args.limit:
        extra += ["--limit", str(args.limit)]

    print(f"Running stages: {requested}")
    for key in requested:
        module_path = STAGE_MODULES[key]
        print(f"\n--- Stage {key} ({module_path}) ---")
        try:
            run_stage(key, module_path, extra)
        except NotImplementedError:
            print(f"  [{key}] SKIPPED (not yet implemented)")
        except Exception as exc:
            print(f"\nPipeline failed at stage [{key}]: {exc}", file=sys.stderr)
            sys.exit(1)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
