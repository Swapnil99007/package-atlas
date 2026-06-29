"""Stage 06: Interactive embedding atlas for GitHub Pages."""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
DOCS = ROOT / "docs"

PYPI_COLOR     = "#4C8EDA"
NPM_COLOR      = "#E8762B"
BG_COLOR       = "#0d0d1a"
MARKER_SIZE    = 3
MARKER_OPACITY = 0.60

TOPIC_LABELS = {
    "@types/node":         "TypeScript Types",
    "schema-utils":        "Webpack / Build Tools",
    "jmespath":            "Data Query & Parsing",
    "boto3-stubs":         "AWS SDK Stubs",
    "python-dateutil":     "Date & Time Utils",
    "lilconfig":           "Config File Loaders",
    "eslint-utils":        "ESLint / Linting",
    "google-auth":         "Auth & Identity",
    "tiktoken":            "AI / LLM Tools",
    "pytest":              "Testing Frameworks",
    "@types/glob":         "File & Path Types",
    "pluggy":              "Plugin Systems",
    "djangorestframework": "Django REST APIs",
    "sglang":              "LLM Inference Servers",
    "python-json-logger":  "Logging & Monitoring",
    "types-requests":      "HTTP Client Stubs",
    "graphql-core":        "GraphQL",
    "zstandard":           "Data Compression",
    "webpack-sources":     "Webpack Ecosystem",
    "pillow-avif-plugin":  "Image Processing",
    "typing-inspection":   "Type System Tools",
}

log = logging.getLogger(__name__)


def format_downloads(n: float) -> str:
    n = int(n)
    if n >= 1_000_000_000: return f"{n/1e9:.1f}B"
    if n >= 1_000_000:     return f"{n/1e6:.0f}M"
    if n >= 1_000:         return f"{n/1e3:.0f}K"
    return str(n)


def top_cluster_labels(df: pd.DataFrame) -> list[dict]:
    clustered = df[df["cluster_id"] >= 0]
    groups = (
        clustered.groupby("cluster_id")
        .agg(n=("name", "count"), cx=("x", "mean"), cy=("y", "mean"))
        .reset_index().sort_values("n", ascending=False).head(21)
    )
    def top_name(cid):
        sub = clustered[clustered["cluster_id"] == cid]
        return sub.loc[sub["download_count"].idxmax(), "name"]
    groups["label"] = groups["cluster_id"].apply(lambda c: TOPIC_LABELS.get(top_name(c), top_name(c)))
    return groups.to_dict(orient="records")


def make_post_script(metrics: dict, n_pypi: int, n_npm: int, n_clusters: int) -> str:
    pypi_r  = metrics["pypi"]["redundancy_rate"]["0.95"]
    npm_r   = metrics["npm"]["redundancy_rate"]["0.95"]
    pypi_p  = metrics["pypi"]["near_dup_pairs_0.95"]
    npm_p   = metrics["npm"]["near_dup_pairs_0.95"]
    total_n = n_pypi + n_npm

    return f"""
(function() {{
  var gd = document.querySelector('.plotly-graph-div');

  var traceNames = [0,1].map(function(ti){{
    return (gd.data[ti].customdata||[]).map(function(d){{return d[0];}});
  }});
  var traceEcos = [0,1].map(function(ti){{
    return (gd.data[ti].customdata||[]).map(function(d){{return d[1];}});
  }});

  gd.on('plotly_click', function(data) {{
    if (!data.points.length) return;
    var pt = data.points[0];
    if (pt.curveNumber > 1) return;
    var name = (traceNames[pt.curveNumber]||[])[pt.pointIndex];
    var eco  = (traceEcos[pt.curveNumber]||[])[pt.pointIndex];
    if (!name) return;
    window.open(eco==='pypi'
      ? 'https://pypi.org/project/'+name+'/'
      : 'https://www.npmjs.com/package/'+name, '_blank');
  }});

  var panel = document.createElement('div');
  panel.style.cssText = (
    'position:fixed;bottom:44px;right:18px;width:280px;max-height:260px;'+
    'background:rgba(18,18,32,0.97);border:1px solid #2a2a4e;border-radius:10px;'+
    'padding:14px 16px;font-family:sans-serif;font-size:12px;color:#ccc;'+
    'box-shadow:0 4px 20px rgba(0,0,0,.6);z-index:600;display:none;overflow-y:auto;line-height:1.5;'
  );
  document.body.appendChild(panel);

  function fmtDl(n){{
    n=parseInt(n)||0;
    return n>=1e9?(n/1e9).toFixed(1)+'B':n>=1e6?Math.round(n/1e6)+'M':n>=1e3?Math.round(n/1e3)+'K':''+n;
  }}

  gd.on('plotly_hover', function(data) {{
    if (!data.points.length) return;
    var pt = data.points[0];
    if (pt.curveNumber > 1) return;
    var cd = (gd.data[pt.curveNumber].customdata||[])[pt.pointIndex]||[];
    var name=cd[0]||'', eco=cd[1]||'', desc=cd[2]||'No description.';
    var dl=fmtDl(cd[3]), cid=cd[4];
    var ec = eco==='pypi' ? '#4C8EDA' : '#E8762B';
    var url = eco==='pypi'
      ? 'https://pypi.org/project/'+name+'/'
      : 'https://www.npmjs.com/package/'+name;
    panel.innerHTML = (
      '<div style="font-size:14px;font-weight:bold;color:#eee;margin-bottom:4px">'+name+'</div>'+
      '<div style="margin-bottom:8px">'+
        '<span style="background:'+ec+';color:#fff;padding:1px 7px;border-radius:4px;font-size:11px">'+eco.toUpperCase()+'</span>'+
        '&nbsp;<span style="color:#555;font-size:11px">'+(cid>=0?'cluster #'+cid:'noise')+'</span>'+
      '</div>'+
      '<div style="color:#aaa;font-size:12px;margin-bottom:10px;line-height:1.6">'+desc+'</div>'+
      '<div style="color:#777;font-size:11px;margin-bottom:10px">⬇ '+dl+' monthly downloads</div>'+
      '<a href="'+url+'" target="_blank" style="display:inline-block;padding:5px 12px;background:'+ec+';color:#fff;border-radius:6px;text-decoration:none;font-size:12px;font-weight:bold">'+
        'View on '+(eco==='pypi'?'PyPI':'npm')+' ↗'+
      '</a>'
    );
    panel.style.display = 'block';
  }});
  gd.on('plotly_unhover', function() {{ panel.style.display = 'none'; }});

  var bar = document.createElement('div');
  bar.style.cssText = 'position:fixed;bottom:0;left:0;right:0;height:34px;background:rgba(13,13,26,0.93);border-top:1px solid #1e1e3e;display:flex;align-items:center;justify-content:center;gap:28px;font-family:monospace;font-size:12px;color:#888;z-index:500;';
  bar.innerHTML = (
    '<span>📦 <b style="color:#ccc">{total_n:,}</b> packages</span>'+
    '<span>🔵 <b style="color:#4C8EDA">{n_pypi:,}</b> PyPI &nbsp;🟠 <b style="color:#E8762B">{n_npm:,}</b> npm</span>'+
    '<span>🗂 <b style="color:#ccc">{n_clusters}</b> clusters</span>'+
    '<span>♻️ npm <b style="color:#E8762B">{100*npm_r:.1f}%</b> vs PyPI <b style="color:#4C8EDA">{100*pypi_r:.1f}%</b> redundant (sim≥0.95)</span>'+
    '<span>🔁 npm near-dup pairs <b style="color:#E8762B">{npm_p:,}</b> vs PyPI <b style="color:#4C8EDA">{pypi_p:,}</b></span>'+
    '<span style="color:#444">Hover for details · Click to open package</span>'
  );
  document.body.appendChild(bar);
  document.body.style.paddingBottom = '34px';
}})();
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s", stream=sys.stderr)

    DOCS.mkdir(exist_ok=True)
    out_path = DOCS / "index.html"
    if out_path.exists() and not args.force:
        print(f"Already done: {out_path}. Use --force to rerun.")
        return

    log.info("Loading data …")
    df = pd.read_parquet(DATA / "clusters.parquet")
    clean = pd.read_parquet(DATA / "combined_clean.parquet")[["name", "ecosystem", "description"]]
    df = df.merge(clean, on=["name", "ecosystem"], how="left")
    df["description"] = df["description"].fillna("")
    metrics = json.loads((DATA / "metrics_summary.json").read_text())

    df["hover"] = [
        f"<b>{r.name}</b> <span style='color:#888'>[{r.ecosystem.upper()}]</span>"
        for r in df.itertuples()
    ]

    labels = top_cluster_labels(df)
    n_pypi     = int((df["ecosystem"] == "pypi").sum())
    n_npm      = int((df["ecosystem"] == "npm").sum())
    n_clusters = int(df["cluster_id"].max()) + 1
    post_script = make_post_script(metrics, n_pypi, n_npm, n_clusters)

    fig = go.Figure()

    for eco, color, label in [("pypi", PYPI_COLOR, "PyPI"), ("npm", NPM_COLOR, "npm")]:
        sub = df[df["ecosystem"] == eco]
        fig.add_trace(go.Scattergl(
            x=sub["x"].values, y=sub["y"].values,
            mode="markers", name=label,
            marker=dict(color=color, size=MARKER_SIZE, opacity=MARKER_OPACITY, line=dict(width=0)),
            text=sub["hover"].values,
            hovertemplate="%{text}<extra></extra>",
            customdata=sub[["name", "ecosystem", "description", "download_count", "cluster_id"]].values,
        ))

    # 21 cluster region labels
    for c in labels:
        fig.add_annotation(
            x=c["cx"], y=c["cy"], text=f"<b>{c['label']}</b>",
            showarrow=False,
            font=dict(family="monospace", color="rgba(255,255,255,0.88)", size=12),
            bgcolor="rgba(13,13,26,0.75)", borderpad=5,
        )

    pypi_r = metrics["pypi"]["redundancy_rate"]["0.95"]
    npm_r  = metrics["npm"]["redundancy_rate"]["0.95"]
    npm_p  = metrics["npm"]["near_dup_pairs_0.95"]
    pypi_p = metrics["pypi"]["near_dup_pairs_0.95"]
    ratio  = npm_p / pypi_p

    fig.add_annotation(
        text=(
            f"<b>npm reinvents the wheel {ratio:.1f}× more than PyPI</b><br>"
            f"<span style='font-size:12px'>At cosine similarity ≥ 0.95<br>"
            f"npm: {100*npm_r:.1f}%  vs  PyPI: {100*pypi_r:.1f}% redundant<br>"
            f"npm near-dup pairs: {npm_p:,}  vs  PyPI: {pypi_p:,}</span>"
        ),
        xref="paper", yref="paper", x=0.01, y=0.99,
        xanchor="left", yanchor="top", showarrow=False,
        bgcolor="rgba(13,13,26,0.90)", bordercolor="#1e1e4e",
        borderwidth=1, borderpad=12,
        font=dict(color="#e0e0e0", size=13), align="left",
    )

    fig.update_layout(
        title=dict(
            text="Package Ecosystem Atlas — PyPI vs npm  •  38,890 packages · 409 clusters",
            font=dict(color="#bbbbbb", size=14), x=0.5,
        ),
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        legend=dict(font=dict(color="#cccccc", size=13),
                    bgcolor="rgba(13,13,26,0.80)", bordercolor="#1e1e3e",
                    borderwidth=1, itemsizing="constant"),
        xaxis=dict(visible=False, fixedrange=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1, fixedrange=False),
        margin=dict(l=10, r=10, t=42, b=44),
        hovermode="closest",
        hoverlabel=dict(bgcolor="#141428", bordercolor="#2a2a5e",
                        font=dict(color="#dddddd", size=12), namelength=0),
        dragmode="pan",
    )

    fig.write_html(
        str(out_path), include_plotlyjs="cdn", post_script=post_script,
        full_html=True,
        config={"scrollZoom": True, "displaylogo": False,
                "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
    )

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nAtlas written: {out_path}  ({size_mb:.1f} MB)")
    print("Open locally:  open docs/index.html")
    print("Live URL:      https://Swapnil99007.github.io/package-atlas/")


if __name__ == "__main__":
    main()
