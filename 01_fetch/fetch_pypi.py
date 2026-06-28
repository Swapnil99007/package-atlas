"""Stage 01a: Fetch top PyPI packages with descriptions and download counts.

Sources:
  - Top-list: https://hugovk.dev/top-pypi-packages/top-pypi-packages.min.json
    (15k packages by monthly downloads; updated monthly)
  - Per-package: https://pypi.org/pypi/{name}/json

Output: data/pypi_raw.parquet
  Columns: name (str), description (str|null), download_count (int64), upload_date (str|null)

Cache: cache/pypi/{name}.json per package (tombstone {"_status":"not_found"} for 404s)
Sentinel: cache/.fetch_pypi_complete (JSON with run metadata)

NOTE: The hugovk dataset caps at 15,000 packages, not the 25k target in CLAUDE.md.
This is a source constraint; see FINDINGS.md for details.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / "cache" / "pypi"
DATA_DIR = ROOT / "data"
SENTINEL = ROOT / "cache" / ".fetch_pypi_complete"

TOP_LIST_URL = "https://hugovk.dev/top-pypi-packages/top-pypi-packages.min.json"
PYPI_API = "https://pypi.org/pypi/{name}/json"
USER_AGENT = "package-atlas/0.1 (portfolio; swapnilkannojia@gmail.com)"
CONCURRENCY = 20

log = logging.getLogger(__name__)


def fetch_top_list(client: httpx.Client, limit: int) -> list[dict]:
    resp = client.get(TOP_LIST_URL)
    resp.raise_for_status()
    data = resp.json()
    rows = data["rows"]
    log.info("Top-list: %d packages (last_update: %s)", len(rows), data.get("last_update"))
    if limit:
        rows = rows[:limit]
    return rows  # [{"project": str, "download_count": int}, ...]


def get_cached_names(cache_dir: Path) -> set[str]:
    return {p.stem for p in cache_dir.glob("*.json")}


async def fetch_package(
    client: httpx.AsyncClient,
    name: str,
    cache_dir: Path,
    sem: asyncio.Semaphore,
) -> None:
    dest = cache_dir / f"{name}.json"
    async with sem:
        try:
            resp = await client.get(PYPI_API.format(name=name))
        except httpx.RequestError as exc:
            log.warning("Request error for %s: %s", name, exc)
            return

    if resp.status_code == 200:
        # Cache only the fields we need — full PyPI JSON includes README + all release
        # history and can be hundreds of KB; compacting keeps the cache under ~10 MB total.
        raw = resp.json()
        info = raw.get("info", {})
        urls = raw.get("urls", [])
        compact = {
            "summary": info.get("summary"),
            "version": info.get("version"),
            "upload_time": urls[0].get("upload_time") if urls else None,
        }
        dest.write_text(json.dumps(compact))
        log.debug("Fetched %s", name)
    elif resp.status_code == 404:
        dest.write_text(json.dumps({"_status": "not_found"}))
        log.debug("404 tombstone: %s", name)
    else:
        log.warning("HTTP %d for %s (will retry on next run)", resp.status_code, name)


async def fetch_all(to_fetch: list[str], cache_dir: Path) -> None:
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0),
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        tasks = [fetch_package(client, name, cache_dir, sem) for name in to_fetch]
        from tqdm.asyncio import tqdm as atqdm
        await atqdm.gather(*tasks, desc="Fetching PyPI packages")


def build_parquet_from_cache(
    top_rows: list[dict], cache_dir: Path, out_path: Path
) -> tuple[int, int]:
    records = []
    skipped = 0
    dl_map = {r["project"]: r["download_count"] for r in top_rows}

    for name, dl_count in dl_map.items():
        p = cache_dir / f"{name}.json"
        if not p.exists():
            log.debug("No cache entry for %s, skipping", name)
            skipped += 1
            continue
        try:
            data = json.loads(p.read_bytes())
        except json.JSONDecodeError:
            log.warning("Corrupt cache file for %s, skipping", name)
            skipped += 1
            continue

        if data.get("_status") == "not_found":
            skipped += 1
            continue

        # Support both compact format {summary, upload_time} and legacy full API response
        if "info" in data:
            info = data["info"]
            summary = info.get("summary") or None
            urls = data.get("urls", [])
            upload_date = urls[0].get("upload_time") if urls else None
        else:
            summary = data.get("summary") or None
            upload_date = data.get("upload_time")

        if summary and summary.strip().upper() == "UNKNOWN":
            summary = None

        records.append(
            {
                "name": name,
                "description": summary,
                "download_count": dl_count,
                "upload_date": upload_date,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records, columns=["name", "description", "download_count", "upload_date"])
    df["download_count"] = df["download_count"].astype("int64")
    df.to_parquet(out_path, index=False)
    log.info("Wrote %d rows to %s (skipped %d)", len(records), out_path, skipped)
    return len(records), skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch top PyPI packages")
    parser.add_argument("--force", action="store_true", help="Rerun even if output exists")
    parser.add_argument("--limit", type=int, default=0, help="Dev subset size (0 = full)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s", stream=sys.stderr)

    out_path = DATA_DIR / "pypi_raw.parquet"
    if out_path.exists() and not args.force:
        print(f"Already done: {out_path}. Use --force to rerun.")
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
        top_rows = fetch_top_list(client, args.limit)

    log.info("Working set: %d packages", len(top_rows))

    cached = get_cached_names(CACHE_DIR)
    to_fetch = [r["project"] for r in top_rows if r["project"] not in cached]
    log.info("Cache hits: %d, to fetch: %d", len(cached), len(to_fetch))

    if to_fetch:
        asyncio.run(fetch_all(to_fetch, CACHE_DIR))

    count, skipped = build_parquet_from_cache(top_rows, CACHE_DIR, out_path)

    sentinel_data = {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "package_count": count,
        "skipped_count": skipped,
    }
    SENTINEL.parent.mkdir(parents=True, exist_ok=True)
    SENTINEL.write_text(json.dumps(sentinel_data, indent=2))

    print(f"PyPI fetch complete: {count} packages written, {skipped} skipped → {out_path}")


if __name__ == "__main__":
    main()
