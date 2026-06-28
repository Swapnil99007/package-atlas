"""Stage 05: Compute within-ecosystem redundancy metrics.

Reads data/embeddings.npy and data/embed_meta.parquet.

Headline metric: share of packages whose description embedding has cosine similarity
>= T to at least one OTHER package in the same ecosystem (within-ecosystem redundancy).
Computed at thresholds [0.90, 0.92, 0.95, 0.97, 0.99].

Since embeddings are L2-normalised, cosine_sim(i,j) = dot(emb_i, emb_j).
Full pairwise matrix is too large for npm (~2.4 GB), so similarity is computed
in chunks and only the per-package max similarity is kept in memory.

Writes:
  data/metrics_summary.json       — headline numbers at all thresholds
  data/near_duplicate_pairs.parquet — pairs with sim >= 0.95 (for inspection)
Appends results to FINDINGS.md.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
FINDINGS = ROOT / "FINDINGS.md"

THRESHOLDS = [0.90, 0.92, 0.95, 0.97, 0.99]
T_COLLECT = 0.95   # only save pairs at this threshold to keep output manageable
CHUNK_SIZE = 500

log = logging.getLogger(__name__)


def compute_eco_metrics(
    eco_emb: np.ndarray,
    chunk_size: int,
    desc: str,
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    """Chunked pairwise similarity within one ecosystem.

    Returns:
        max_sims: float32 array [N] — max cosine sim to any OTHER package
        pairs: list of (i, j, sim) where i < j and sim >= T_COLLECT
    """
    n = len(eco_emb)
    max_sims = np.full(n, -1.0, dtype=np.float32)
    pairs: list[tuple[int, int, float]] = []

    for start in tqdm(range(0, n, chunk_size), desc=desc, unit="chunk"):
        end = min(start + chunk_size, n)
        chunk = eco_emb[start:end]          # (chunk_size, D)
        sims = (chunk @ eco_emb.T).astype(np.float32)  # (chunk_size, N)

        # Zero out self-similarity so it doesn't count as a near-duplicate
        for local_i in range(end - start):
            sims[local_i, start + local_i] = -1.0

        max_sims[start:end] = sims.max(axis=1)

        # Collect near-duplicate pairs (i < j only, to avoid double-counting)
        above = np.argwhere(sims >= T_COLLECT)  # (K, 2) — [local_i, j]
        for local_i, j in above:
            global_i = start + local_i
            if global_i < j:
                pairs.append((int(global_i), int(j), float(sims[local_i, j])))

    return max_sims, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute within-ecosystem redundancy metrics")
    parser.add_argument("--force", action="store_true", help="Rerun even if output exists")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Primary threshold (all thresholds are always computed)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s", stream=sys.stderr)

    out_path = DATA / "metrics_summary.json"
    if out_path.exists() and not args.force:
        print(f"Already done: {out_path}. Use --force to rerun.")
        return

    log.info("Loading embeddings …")
    emb = np.load(DATA / "embeddings.npy")
    meta = pd.read_parquet(DATA / "embed_meta.parquet")
    clean = pd.read_parquet(DATA / "combined_clean.parquet")[["name", "ecosystem", "description"]]

    summary: dict = {"thresholds": THRESHOLDS}
    all_pairs: list[pd.DataFrame] = []

    for eco in ["pypi", "npm"]:
        idx = (meta["ecosystem"] == eco).values
        eco_emb = emb[idx]
        eco_clean = clean[clean["ecosystem"] == eco].reset_index(drop=True)
        n = len(eco_emb)
        log.info("%s: %d packages", eco, n)

        t0 = time.perf_counter()
        max_sims, pairs = compute_eco_metrics(eco_emb, args.chunk_size, desc=f"{eco}")
        elapsed = time.perf_counter() - t0
        log.info("%s done in %.0fs — %d pairs at T=%.2f", eco, elapsed, len(pairs), T_COLLECT)

        redundancy = {
            str(t): float(round((max_sims >= t).mean(), 6))
            for t in THRESHOLDS
        }
        summary[eco] = {
            "n": n,
            "redundancy_rate": redundancy,
            "mean_max_sim": float(round(float(max_sims.mean()), 6)),
            "median_max_sim": float(round(float(np.median(max_sims)), 6)),
            f"near_dup_pairs_{T_COLLECT}": len(pairs),
            "perfect_dups": int((max_sims >= 0.9999).sum()),
        }

        if pairs:
            df_pairs = pd.DataFrame(pairs, columns=["i", "j", "cosine_sim"])
            df_pairs["ecosystem"] = eco
            df_pairs["name_a"] = eco_clean["name"].iloc[df_pairs["i"]].values
            df_pairs["desc_a"] = eco_clean["description"].iloc[df_pairs["i"]].values
            df_pairs["name_b"] = eco_clean["name"].iloc[df_pairs["j"]].values
            df_pairs["desc_b"] = eco_clean["description"].iloc[df_pairs["j"]].values
            df_pairs = df_pairs[["ecosystem", "name_a", "desc_a", "name_b", "desc_b", "cosine_sim"]]
            df_pairs = df_pairs.sort_values("cosine_sim", ascending=False).reset_index(drop=True)
            all_pairs.append(df_pairs)

    # Write outputs
    out_path.write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", out_path)

    pairs_path = DATA / "near_duplicate_pairs.parquet"
    if all_pairs:
        pd.concat(all_pairs, ignore_index=True).to_parquet(pairs_path, index=False)
        log.info("Wrote %s", pairs_path)

    # Append to FINDINGS.md
    date_str = time.strftime("%Y-%m-%d", time.gmtime())
    block = f"\n## Stage 05 (Redundancy Metrics) — {date_str}\n"
    block += f"Model: BAAI/bge-small-en-v1.5 | Embeddings: L2-normalised 384-dim\n"
    block += f"Primary threshold: cosine similarity >= {args.threshold}\n\n"
    block += "| Threshold | PyPI | npm | npm − PyPI |\n|---|---|---|---|\n"
    for t in THRESHOLDS:
        p = summary["pypi"]["redundancy_rate"][str(t)]
        n = summary["npm"]["redundancy_rate"][str(t)]
        diff = n - p
        block += f"| {t} | {100*p:.1f}% | {100*n:.1f}% | +{100*diff:.1f}pp |\n"
    block += f"\nMean max-sim: PyPI {summary['pypi']['mean_max_sim']:.4f} vs npm {summary['npm']['mean_max_sim']:.4f}\n"
    block += f"Near-duplicate pairs (sim >= {T_COLLECT}): PyPI {summary['pypi'][f'near_dup_pairs_{T_COLLECT}']}, npm {summary['npm'][f'near_dup_pairs_{T_COLLECT}']}\n"
    block += f"Perfect duplicates (sim >= 0.9999): PyPI {summary['pypi']['perfect_dups']}, npm {summary['npm']['perfect_dups']}\n"

    with FINDINGS.open("a") as f:
        f.write(block)

    # Print headline
    print("\n=== HEADLINE FINDING: Within-Ecosystem Redundancy ===")
    print(f"{'Threshold':<12} {'PyPI':>8} {'npm':>8} {'npm − PyPI':>12}")
    print("-" * 44)
    for t in THRESHOLDS:
        p = summary["pypi"]["redundancy_rate"][str(t)]
        n_r = summary["npm"]["redundancy_rate"][str(t)]
        marker = " ◀" if t == args.threshold else ""
        print(f"{t:<12} {100*p:>7.1f}% {100*n_r:>7.1f}% {100*(n_r-p):>+11.1f}pp{marker}")
    print()
    print(f"Mean max-sim:    PyPI {summary['pypi']['mean_max_sim']:.4f}  |  npm {summary['npm']['mean_max_sim']:.4f}")
    print(f"Near-dup pairs:  PyPI {summary['pypi'][f'near_dup_pairs_{T_COLLECT}']}  |  npm {summary['npm'][f'near_dup_pairs_{T_COLLECT}']}")
    print(f"Perfect dups:    PyPI {summary['pypi']['perfect_dups']}  |  npm {summary['npm']['perfect_dups']}")


if __name__ == "__main__":
    main()
