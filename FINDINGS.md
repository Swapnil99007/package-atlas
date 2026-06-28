# FINDINGS.md — Package Ecosystem Atlas

Running log of computed numbers. Every entry must include the exact method, thresholds, and date computed.

## Data Collection

### PyPI
- Source: hugovk/top-pypi-packages (`https://hugovk.dev/top-pypi-packages/top-pypi-packages.min.json`)
- Coverage: **15,000 packages** (dataset cap; CLAUDE.md targeted 25k — this is a source constraint, not a code bug)
- Download count window: monthly, as of dataset snapshot date
- Last dataset update: TBD (recorded from `last_update` field at fetch time)

### npm
- Source: evanwashere/top-npm-packages `all.json`
- Coverage: **25,000 packages** (top slice by monthly downloads)
- Download count window: monthly
- Last dataset update: April 2026

## Stage 02 (Clean)
*TBD — to be filled in after running clean.py*

## Stage 03 (Embed)
*TBD*

## Stage 05 (Redundancy Metrics)
*TBD — this is the headline finding*

### Headline metric definition
Within-ecosystem redundancy = share of packages whose description cosine-similarity to at least one other package exceeds threshold T.
Threshold T: TBD (to be calibrated on a dev sample).

## Stage 02 (Clean) — 2026-06-28
PYPI: 14995 → 14604 kept (391 dropped, 2.6%)
  null_empty: 280 | single_word: 98 | placeholder: 0 | non_english: 13
NPM: 24954 → 24286 kept (668 dropped, 2.7%)
  null_empty: 358 | single_word: 285 | placeholder: 0 | non_english: 25
Combined clean dataset: 38890 rows
Non-English threshold: non-ASCII ratio > 0.3 OR CJK/Cyrillic chars present
