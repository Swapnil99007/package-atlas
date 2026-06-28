"""Stage 01b: Fetch top npm packages with descriptions and download counts.

Sources:
  - Top-list: https://raw.githubusercontent.com/evanwashere/top-npm-packages/master/all.json
    Format: [[name, monthly_downloads], ...], sorted by downloads desc. ~45 MB.
    Cached once to cache/npm/npm_all.json.
  - Per-package: https://registry.npmjs.org/{name}/latest

Output: data/npm_raw.parquet
  Columns: name (str), description (str|null), download_count (int64), upload_date (str|null)

Cache: cache/npm/{safe_name}.json per package (tombstone {"_status":"not_found"} for 404s)
Sentinel: cache/.fetch_npm_complete (JSON with run metadata)

Scoped packages: @scope/name stored as @scope__name.json (/ replaced by __)
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from urllib.parse import quote

import httpx
import pandas as pd

ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / "cache" / "npm"
DATA_DIR = ROOT / "data"
ALL_JSON_CACHE = CACHE_DIR / "npm_all.json"
SENTINEL = ROOT / "cache" / ".fetch_npm_complete"

ALL_JSON_URL = (
    "https://raw.githubusercontent.com/evanwashere/top-npm-packages/master/all.json"
)
REGISTRY_URL = "https://registry.npmjs.org/{encoded_name}/latest"
USER_AGENT = "package-atlas/0.1 (portfolio; swapnilkannojia@gmail.com)"
CONCURRENCY = 3          # npm registry rate-limits aggressively; keep concurrency very low
REQUEST_DELAY = 0.2      # seconds held inside semaphore after each request (~15 req/sec max)
DEFAULT_LIMIT = 25_000

log = logging.getLogger(__name__)


def safe_cache_name(name: str) -> str:
    return name.replace("/", "__")


def unsafe_cache_name(safe: str) -> str:
    return safe.replace("__", "/", 1)  # only first occurrence (scope separator)


def fetch_npm_all_list(limit: int) -> list[tuple[str, int]]:
    if ALL_JSON_CACHE.exists():
        log.info("Loading npm top-list from cache: %s", ALL_JSON_CACHE)
        data = json.loads(ALL_JSON_CACHE.read_bytes())
    else:
        log.info("Downloading npm top-list (~45 MB) …")
        with httpx.Client(
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=httpx.Timeout(120.0, connect=30.0),
        ) as client:
            resp = client.get(ALL_JSON_URL)
            resp.raise_for_status()
        ALL_JSON_CACHE.parent.mkdir(parents=True, exist_ok=True)
        ALL_JSON_CACHE.write_bytes(resp.content)
        data = resp.json()
        log.info("Downloaded and cached npm top-list (%d bytes)", len(resp.content))

    entries: list[tuple[str, int]] = [(row[0], row[1]) for row in data]
    entries.sort(key=lambda x: x[1], reverse=True)
    cap = limit if limit else DEFAULT_LIMIT
    return entries[:cap]


def get_cached_names(cache_dir: Path) -> set[str]:
    cached_safe = {p.stem for p in cache_dir.glob("*.json") if p.stem != "npm_all"}
    return {unsafe_cache_name(s) for s in cached_safe}


async def fetch_package(
    client: httpx.AsyncClient,
    name: str,
    cache_dir: Path,
    sem: asyncio.Semaphore,
    retry_budget: int = 6,
) -> None:
    dest = cache_dir / f"{safe_cache_name(name)}.json"
    encoded = quote(name, safe="@")
    url = REGISTRY_URL.format(encoded_name=encoded)

    for attempt in range(1, retry_budget + 1):
        async with sem:
            try:
                resp = await client.get(url)
            except httpx.RequestError as exc:
                log.warning("Request error for %s: %s", name, exc)
                return
            # Hold the slot briefly to stay within npm's rate limit
            await asyncio.sleep(REQUEST_DELAY)

        if resp.status_code == 200:
            # Cache only the fields we need — full /latest responses embed READMEs
            # and can be hundreds of KB each; compacting keeps the cache under ~5 MB total.
            raw = resp.json()
            compact = {
                "description": raw.get("description"),
                "time": raw.get("time"),
                "version": raw.get("version"),
            }
            dest.write_text(json.dumps(compact))
            log.debug("Fetched %s", name)
            return
        elif resp.status_code == 404:
            dest.write_text(json.dumps({"_status": "not_found"}))
            log.debug("404 tombstone: %s", name)
            return
        elif resp.status_code == 429:
            # Exponential backoff: 2s, 4s, 8s, 15s, 30s, 60s
            wait = min(2.0 * (2 ** (attempt - 1)), 60.0)
            log.warning("429 for %s (attempt %d/%d), backing off %.0fs", name, attempt, retry_budget, wait)
            await asyncio.sleep(wait)
        else:
            log.warning("HTTP %d for %s (will retry on next run)", resp.status_code, name)
            return

    log.warning("Gave up on %s after %d attempts", name, retry_budget)


async def fetch_all(to_fetch: list[str], cache_dir: Path) -> None:
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0),
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        tasks = [fetch_package(client, name, cache_dir, sem) for name in to_fetch]
        from tqdm.asyncio import tqdm as atqdm
        await atqdm.gather(*tasks, desc="Fetching npm packages")


def build_parquet_from_cache(
    entries: list[tuple[str, int]], cache_dir: Path, out_path: Path
) -> tuple[int, int]:
    records = []
    skipped = 0
    dl_map = dict(entries)

    for name, dl_count in entries:
        p = cache_dir / f"{safe_cache_name(name)}.json"
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

        description = data.get("description") or None
        if description and description.strip().upper() in ("", "N/A", "UNKNOWN"):
            description = None

        upload_date = None
        t = data.get("time")
        if isinstance(t, str):
            upload_date = t
        elif isinstance(t, dict):
            upload_date = t.get("modified") or t.get("created")

        records.append(
            {
                "name": name,
                "description": description,
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
    parser = argparse.ArgumentParser(description="Fetch top npm packages")
    parser.add_argument("--force", action="store_true", help="Rerun even if output exists")
    parser.add_argument("--limit", type=int, default=0, help="Dev subset size (0 = full 25k)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s", stream=sys.stderr)

    out_path = DATA_DIR / "npm_raw.parquet"
    if out_path.exists() and not args.force:
        print(f"Already done: {out_path}. Use --force to rerun.")
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    entries = fetch_npm_all_list(args.limit)
    log.info("Working set: %d npm packages", len(entries))

    cached = get_cached_names(CACHE_DIR)
    to_fetch = [name for name, _ in entries if name not in cached]
    log.info("Cache hits: %d, to fetch: %d", len(cached), len(to_fetch))

    if to_fetch:
        asyncio.run(fetch_all(to_fetch, CACHE_DIR))

    count, skipped = build_parquet_from_cache(entries, CACHE_DIR, out_path)

    sentinel_data = {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "package_count": count,
        "skipped_count": skipped,
    }
    SENTINEL.parent.mkdir(parents=True, exist_ok=True)
    SENTINEL.write_text(json.dumps(sentinel_data, indent=2))

    print(f"npm fetch complete: {count} packages written, {skipped} skipped → {out_path}")


if __name__ == "__main__":
    main()
