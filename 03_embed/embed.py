"""Stage 03: Compute embeddings for package descriptions.

Reads data/combined_clean.parquet.
Encodes every description with BAAI/bge-small-en-v1.5 (33M params, encoder-only).
Runs on MPS (Apple Silicon) — encoder-only models are well-supported by PyTorch MPS.
Writes:
  data/embeddings.npy      — float32 [N, 384], L2-normalized
  data/embed_meta.parquet  — same row order: name, ecosystem, download_count, upload_date

L2 normalization means cosine similarity = dot product in stage 05 (faster, simpler).
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

MODEL_ID = "BAAI/bge-small-en-v1.5"
DIM = 384
DEFAULT_BATCH_SIZE = 256  # encoder-only model, MPS-friendly, large batches are fine

log = logging.getLogger(__name__)


def get_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed package descriptions with Qwen3-Embedding-0.6B")
    parser.add_argument("--force", action="store_true", help="Rerun even if output exists")
    parser.add_argument("--limit", type=int, default=0, help="Dev subset size (0 = full 38k)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s", stream=sys.stderr)

    emb_path = DATA / "embeddings.npy"
    meta_path = DATA / "embed_meta.parquet"
    if emb_path.exists() and meta_path.exists() and not args.force:
        print(f"Already done: {emb_path}. Use --force to rerun.")
        return

    log.info("Loading combined_clean.parquet …")
    df = pd.read_parquet(DATA / "combined_clean.parquet")
    if args.limit:
        df = df.iloc[: args.limit].copy()
        log.info("Dev mode: using first %d rows", args.limit)

    descriptions = df["description"].tolist()
    n = len(descriptions)

    device = get_device()
    log.info("Device: %s | Rows: %d | Model: %s", device, n, MODEL_ID)

    from sentence_transformers import SentenceTransformer

    log.info("Loading model (first run downloads ~1.2 GB) …")
    model = SentenceTransformer(MODEL_ID, device=device)

    log.info("Encoding %d descriptions (batch_size=%d) …", n, args.batch_size)
    t0 = time.perf_counter()
    embeddings = model.encode(
        descriptions,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    elapsed = time.perf_counter() - t0

    assert embeddings.shape == (n, DIM), f"Unexpected shape: {embeddings.shape}"
    embeddings = embeddings.astype(np.float32)

    DATA.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings)
    log.info("Saved %s", emb_path)

    meta = df[["name", "ecosystem", "download_count", "upload_date"]].reset_index(drop=True)
    meta.to_parquet(meta_path, index=False)
    log.info("Saved %s", meta_path)

    norms = np.linalg.norm(embeddings, axis=1)
    print(
        f"\nEmbedding complete: {n} rows × {DIM} dims | device={device} | {elapsed:.0f}s"
        f"\nL2 norm check: min={norms.min():.4f} max={norms.max():.4f} (should be ≈ 1.0)"
        f"\nFiles: {emb_path} ({emb_path.stat().st_size / 1e6:.0f} MB)"
        f"  {meta_path}"
    )


if __name__ == "__main__":
    main()
