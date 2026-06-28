"""Stage 02: Clean raw package data.

Reads data/pypi_raw.parquet and data/npm_raw.parquet.
Applies four filters in sequence (per ecosystem so drop counts are comparable):
  1. null_empty   — null or empty after strip
  2. single_word  — fewer than 2 words
  3. placeholder  — exact match against known junk strings
  4. non_english  — CJK/Cyrillic characters OR non-ASCII ratio > 0.3

Writes data/combined_clean.parquet and data/clean_report.json.
Appends drop counts to FINDINGS.md — the per-ecosystem rates are themselves a finding.
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
FINDINGS = ROOT / "FINDINGS.md"

log = logging.getLogger(__name__)

PLACEHOLDERS = frozenset([
    "unknown", "todo", "tbd", "fixme", "n/a", "none",
    "description", "no description", "package description",
    "test", "wip",
])

# Matches CJK Unified Ideographs, Hiragana/Katakana, Hangul, and Cyrillic blocks
_NON_LATIN_RE = re.compile(r"[一-鿿぀-ヿ가-힯Ѐ-ӿ]")

NON_ASCII_THRESHOLD = 0.3


def is_non_english(s: str) -> bool:
    if _NON_LATIN_RE.search(s):
        return True
    non_ascii = sum(1 for c in s if ord(c) > 127)
    return (non_ascii / len(s)) > NON_ASCII_THRESHOLD if s else False


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    drops: dict[str, int] = {}

    mask = df["description"].isna() | (df["description"].str.strip() == "")
    drops["null_empty"] = int(mask.sum())
    df = df[~mask].copy()

    mask = df["description"].str.split().str.len() < 2
    drops["single_word"] = int(mask.sum())
    df = df[~mask].copy()

    mask = df["description"].str.strip().str.lower().isin(PLACEHOLDERS)
    drops["placeholder"] = int(mask.sum())
    df = df[~mask].copy()

    mask = df["description"].apply(is_non_english)
    drops["non_english"] = int(mask.sum())
    df = df[~mask].copy()

    return df, drops


def format_report(ecosystem: str, original: int, drops: dict[str, int], kept: int) -> str:
    total_dropped = sum(drops.values())
    pct = 100 * total_dropped / original if original else 0
    lines = [
        f"{ecosystem}: {original} → {kept} kept ({total_dropped} dropped, {pct:.1f}%)",
        "  " + " | ".join(f"{k}: {v}" for k, v in drops.items()),
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw package data")
    parser.add_argument("--force", action="store_true", help="Rerun even if output exists")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s", stream=sys.stderr)

    out_path = DATA / "combined_clean.parquet"
    if out_path.exists() and not args.force:
        print(f"Already done: {out_path}. Use --force to rerun.")
        return

    log.info("Loading raw parquets …")
    pypi = pd.read_parquet(DATA / "pypi_raw.parquet")
    npm = pd.read_parquet(DATA / "npm_raw.parquet")
    pypi["ecosystem"] = "pypi"
    npm["ecosystem"] = "npm"

    report: dict = {}
    clean_frames = []

    for ecosystem, df in [("pypi", pypi), ("npm", npm)]:
        original = len(df)
        clean_df, drops = apply_filters(df)
        kept = len(clean_df)
        clean_frames.append(clean_df)
        report[ecosystem] = {"original": original, "kept": kept, "drops": drops}
        print(format_report(ecosystem.upper(), original, drops, kept))

    combined = pd.concat(clean_frames, ignore_index=True)
    report["combined_total"] = len(combined)

    # Guarantee no nulls leak through
    assert combined["description"].isna().sum() == 0, "BUG: nulls survived cleaning"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    log.info("Wrote %d rows to %s", len(combined), out_path)

    report_path = DATA / "clean_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    # Append to FINDINGS.md
    date_str = time.strftime("%Y-%m-%d", time.gmtime())
    findings_block = f"\n## Stage 02 (Clean) — {date_str}\n"
    for eco in ("pypi", "npm"):
        r = report[eco]
        findings_block += format_report(eco.upper(), r["original"], r["drops"], r["kept"]) + "\n"
    findings_block += f"Combined clean dataset: {len(combined)} rows\n"
    findings_block += f"Non-English threshold: non-ASCII ratio > {NON_ASCII_THRESHOLD} OR CJK/Cyrillic chars present\n"

    with FINDINGS.open("a") as f:
        f.write(findings_block)

    print(f"\nCombined clean dataset: {len(combined)} rows → {out_path}")


if __name__ == "__main__":
    main()
