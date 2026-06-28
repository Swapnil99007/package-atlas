# CLAUDE.md — Package Ecosystem Atlas

## What this project is
An interactive "embedding atlas" that maps PyPI (Python) and npm (JavaScript)
package descriptions into a 2D map of meaning, where similar packages cluster
together. It is a portfolio project intended to demonstrate end-to-end ML
engineering skill and to be shared publicly (LinkedIn). The map is the visual,
but the project's value rests on a **quantified, defensible finding**, not on
the picture looking nice.

## The single most important principle
**Measure first, visualize second.** The headline result must be a number
computed directly from the embeddings (e.g. via cosine similarity), and the map
should *illustrate* that number. Do NOT build a pretty map and then go looking
for a story in it. A reviewer will probe the methodology, so every claim must be
reproducible and the thresholds must be defensible.

## The headline finding we are hunting for
Hypothesis: **npm reinvents the wheel far more than PyPI.** That is, npm should
have many more dense clusters of packages whose self-descriptions are
near-identical (the `left-pad` / `is-odd` / `is-even` phenomenon) than PyPI does.

The key metric is **within-ecosystem redundancy**, measured separately for each
ecosystem — for example, the share of packages whose description sits within a
cosine-similarity threshold of at least one other package, and the count/size of
near-duplicate clusters per ecosystem.

AVOID the trap: simply embedding everything and coloring dots by ecosystem will
produce two separated blobs. That is a boring "different languages use different
words" result. The interesting finding is *within*, not *between*, ecosystems.

Possible secondary findings (nice-to-have, not required): clusters of spam /
typosquatting packages on npm; domain composition contrast (npm = web tooling,
PyPI = data/ML); growth of AI-related packages over time if upload dates are kept.

## Scope (do not exceed without asking)
- Sample the **top ~25,000 packages per registry by download count.** Do NOT try
  to embed all of npm (~3M packages) — it is unnecessary, slow, and full of junk.
- Keep download counts and upload dates as metadata for later filtering/coloring.
- The long tail (full registry) is an optional later "bonus" pass, not the MVP.

## Data sources (verified reachable as of the planning session)
PyPI (the easy one):
- Bulk metadata via the `pypi-data` project / py-code.org (parquet datasets).
- Top-packages-by-downloads list: the community `top-pypi-packages` dataset
  (top ~15k) — use to filter to real, used packages.
- The per-package JSON API (`https://pypi.org/pypi/<name>/json`) has the
  `summary` field; long description / README is also available.

npm (heavier):
- Full metadata via CouchDB replication from `replicate.npmjs.com` — works but is
  slow and can fail partway; respect the 1 req/sec crawler policy.
- Prefer a pre-built research dump (e.g. a recent npm registry dataset on Zenodo)
  to avoid replicating 3M packages by hand.
- Download counts via the npm downloads API.

Always respect robots.txt and rate limits. Cache everything to disk so we never
re-fetch or re-embed.

## Tech stack
- **Language:** Python for the data + ML pipeline.
- **Embeddings — run an open model locally** (this is the skill signal; do not
  just call a hosted API and stop). Good options: `Qwen3-Embedding-0.6B`,
  Google `EmbeddingGemma-300M`, or `Jina v5-text-small`. Pick one, document why.
- **Dimensionality reduction:** UMAP (2D).
- **Clustering:** HDBSCAN (density-based).
- **Cluster labeling:** an LLM pass to name clusters, OR use BERTopic which
  bundles embeddings + UMAP + HDBSCAN + topic labeling for a faster MVP.
- **Visualization:** prototype with Apple's open-source Embedding Atlas or Nomic
  Atlas to validate the finding fast. A custom frontend (deck.gl /
  regl-scatterplot) is an optional week-2 upgrade for a stronger "I built this"
  signal — decide after the finding is confirmed.

## Engineering conventions
- Reproducibility first: pin dependencies, set random seeds (UMAP/HDBSCAN are
  sensitive), and make the whole pipeline runnable end-to-end from one entry point.
- Cache aggressively: raw metadata, cleaned data, and computed embeddings each
  saved to disk (parquet / npy) so reruns are cheap.
- Structure the repo in clear stages: `01_fetch`, `02_clean`, `03_embed`,
  `04_reduce_cluster`, `05_metrics`, `06_visualize`. Each stage reads the previous
  stage's cached output.
- Cleaning matters: drop empty / one-word / placeholder / non-English
  descriptions; record how many were dropped and why (the dropped spam is itself a
  potential finding).
- Keep a running `FINDINGS.md` where the actual computed numbers are recorded as
  they come in, with the exact method and thresholds used.

## Deliverables
1. Reproducible pipeline (the staged scripts above).
2. The interactive map (no signup, loads in one click, has a search box so people
   can find their own favorite package).
3. `README.md` written like a product landing page: screenshot + one-sentence
   finding at the top, methodology and stack below.
4. A short launch writeup (separate file) leading with the finding, not the stack.

## What NOT to do
- Do not embed the entire npm registry.
- Do not present a "two blobs colored by ecosystem" map as the finding.
- Do not skip the redundancy metric in favor of going straight to the visual.
- Do not hardcode secrets/API keys; read from environment variables.
- Do not commit large data files to git; gitignore the data/ and cache/ dirs.

## Workflow preference
Default to **Plan mode** for any new stage: describe the approach and wait for
approval before writing files or running long jobs (especially fetches and
embedding runs, which are slow). Confirm scope before any step that downloads at
scale or embeds.
