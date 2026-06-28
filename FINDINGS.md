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

## Stage 05 (Redundancy Metrics) — 2026-06-28
Model: BAAI/bge-small-en-v1.5 | Embeddings: L2-normalised 384-dim
Primary threshold: cosine similarity >= 0.95

| Threshold | PyPI | npm | npm − PyPI |
|---|---|---|---|
| 0.9 | 27.0% | 34.9% | +7.9pp |
| 0.92 | 20.4% | 26.2% | +5.8pp |
| 0.95 | 12.3% | 15.8% | +3.5pp |
| 0.97 | 8.6% | 11.1% | +2.5pp |
| 0.99 | 6.2% | 8.5% | +2.3pp |

Mean max-sim: PyPI 0.8617 vs npm 0.8764
Near-duplicate pairs (sim >= 0.95): PyPI 3940, npm 9855
Perfect duplicates (sim >= 0.9999): PyPI 832, npm 1957
